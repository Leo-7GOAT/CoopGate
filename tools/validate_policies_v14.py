"""Fresh DEVELOPMENT validation in actual SUMO, never a preregistered test.

All requested fixed checkpoints and the unchanged strong baseline execute
the same initials and exogenous disturbance tapes. Results include failures.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from coopgate.disturbance_v9 import DisturbedMergeEnv
from coopgate.ego_v13 import EgoRolloutGate,EgoPolicyGate
from experiment_ego_v13 import run_episode

ROOT=Path(__file__).resolve().parents[1]


def factory(path):
    torch.set_num_threads(1)
    meta=torch.load(path,map_location='cpu',weights_only=True)
    if meta['algorithm']=='rollout_policy_iteration_ego_v14':
        from coopgate.policy_iteration_v14 import load_policy
        return lambda x,m=load_policy(path)[0]:EgoPolicyGate(m)
    if meta['algorithm']=='certificate_feature_policy_iteration_v14':
        from coopgate.certificate_policy_v14 import load_policy,CertificatePolicyGate
        return lambda x,m=load_policy(path)[0]:CertificatePolicyGate(m)
    if meta['algorithm']=='kinematic_feature_policy_iteration_v14':
        from coopgate.kinematic_policy_v14 import load_policy,KinematicPolicyGate
        return lambda x,m=load_policy(path)[0]:KinematicPolicyGate(m)
    raise ValueError('Unrecognized policy')


def run(out,paths,n=64,seed_base=37000001):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    factories={'rollout':lambda x:EgoRolloutGate()}
    for index,path in enumerate(paths):factories[f'policy_{index}']=factory(path)
    protocol=dict(purpose='development validation, not an independent final test',initials=n,seed_base=seed_base,
                  modes=['clean','uniform','bias','pulse'],methods={'rollout':'unchanged EgoRolloutGate',**{f'policy_{i}':str(p) for i,p in enumerate(paths)}},
                  policy_sha256={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
                  evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    warm=np.array([[-80.,12.],[-85.,13.],[-120.,11.]])
    for f in factories.values():f(warm).select(warm,np.zeros(3))
    started=time.perf_counter();rows=[]
    with DisturbedMergeEnv() as env:
        for i in range(n):
            seed=seed_base+i;regime='nominal' if i<n//2 else 'dense'
            for mode in protocol['modes']:
                order=np.random.default_rng(seed+protocol['modes'].index(mode)).permutation(list(factories))
                for name in order:
                    result,trace,initial=run_episode(env,seed,regime,mode,factories[name],i<2)
                    rows.append({**result,'controller':str(name),'initial':initial.tolist()})
                    if trace is not None:(out/f'trace_{i}_{mode}_{name}.json').write_text(json.dumps(trace,indent=2)+'\n')
            if (i+1)%8==0:
                (out/'episodes.json').write_text(json.dumps(rows,indent=2)+'\n')
                print(json.dumps(dict(initials=i+1,sumo_runs=len(rows),elapsed_s=time.perf_counter()-started)),flush=True)
    (out/'episodes.json').write_text(json.dumps(rows,indent=2)+'\n')
    summary={}
    for name in factories:
        r=[x for x in rows if x['controller']==name]
        summary[name]={k:float(np.mean([x[k] for x in r])) for k in ('ego_time','ego_variation','ego_cost','other_time')}
        summary[name].update(runs=len(r),completed=sum(x['completed'] for x in r),
                             collisions=sum(x['collision'] for x in r),aborted=sum(x['aborted'] for x in r))
    result=dict(protocol=protocol,results=summary,sumo_runs=len(rows),elapsed_s=time.perf_counter()-started)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('paths',nargs='+');p.add_argument('--out',required=True)
    p.add_argument('--n',type=int,default=64);p.add_argument('--seed-base',type=int,default=37000001)
    a=p.parse_args();run(a.out,a.paths,a.n,a.seed_base)
