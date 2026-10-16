"""Fresh paired SUMO DEVELOPMENT comparison of trained multi-agent policies.

Both the strong ego planner and learning-free cooperative planner are retained.
No formal acceptance is inferred, and no vehicle is hidden in a team average.
"""
import argparse,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
import torch
from coopgate.cooperative_v15 import load_joint,JointPolicyGate
from coopgate.joint_planner_v15 import JointRolloutGate
from coopgate.ego_v13 import EgoRolloutGate,EgoPolicyGate
from coopgate.policy_iteration_v14 import load_policy
from coopgate.ppo_numerics_v15 import require_complete_training
from coopgate.disturbance_v9 import DisturbedMergeEnv
from experiment_ego_v13 import run_episode
from prerequisite_v15 import require_single

ROOT=Path(__file__).resolve().parents[1]


def run(paths,out,n=128,seed_base=59000001):
    prerequisite=require_single(ROOT)
    if n<4 or n%2:raise ValueError('Use at least four initials and equal fixed regime quotas')
    torch.set_num_threads(1)
    factories={'ego_planner':lambda x:EgoRolloutGate(),'team_planner':lambda x:JointRolloutGate()}
    checkpoints={};teachers=set()
    for path in paths:
        model,meta=load_joint(path)
        require_complete_training(meta)
        kind='mappo' if meta['critic_kind']=='central' else 'ippo';name=f'{kind}_{meta["seed"]}'
        if name in factories:raise ValueError('Repeated trained method name')
        factories[name]=lambda x,m=model:JointPolicyGate(m);teachers.add(meta['ego_seed'])
        checkpoints[name]=dict(path=str(path),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            algorithm=meta['algorithm'],seed=meta['seed'],ego_seed=meta['ego_seed'],steps=meta['steps'],
            actor_parameter_delta_l2=meta['actor_parameter_delta_l2'])
    if not checkpoints:raise ValueError('At least one actual trained joint policy required')
    for seed in sorted(teachers):
        model,_=load_policy(ROOT/f'artifacts/v14/single/seed_{seed}/policy.pt')
        factories[f'single_{seed}']=lambda x,m=model:EgoPolicyGate(m)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    modes=['clean','uniform','bias','pulse'];audit_indices=[0,1,n//2,n//2+1]
    sources=list((ROOT/'src/coopgate').glob('*.py'))+[Path(__file__),ROOT/'tools/experiment_ego_v13.py',ROOT/'tools/prerequisite_v15.py']
    protocol=dict(purpose='Multi-agent DEVELOPMENT evaluation; no formal acceptance verdict',
        n_initials=n,seed_base=seed_base,methods=list(factories),modes=modes,audit_initial_indices=audit_indices,
        expected_runs=n*len(modes)*len(factories),models=checkpoints,prerequisite=prerequisite,
        source_sha256={str(p.resolve().relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for name in protocol['source_sha256']:
        target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    warm=np.array([[-80.,12.],[-85.,13.],[-120.,11.]])
    for factory in factories.values():factory(warm).select(warm,np.zeros(3))
    rows=[];started=time.perf_counter()
    with DisturbedMergeEnv() as env:
        for i in range(n):
            seed=seed_base+i;regime='nominal' if i<n//2 else 'dense'
            for mode_index,mode in enumerate(modes):
                order=np.random.default_rng(seed+mode_index+1515).permutation(list(factories))
                for method in order:
                    result,trace,initial=run_episode(env,seed,regime,mode,factories[method],i in audit_indices)
                    costs=np.array(result['per_vehicle_finish'])+.05*np.array(result['per_vehicle_variation'])
                    result.update(controller=str(method),initial=initial.tolist(),per_vehicle_cost=costs.tolist(),team_cost=float(costs.mean()))
                    rows.append(result)
                    if trace is not None:(out/f'trace_{i}_{mode}_{method}.json').write_text(json.dumps(trace,indent=2)+'\n')
            if (i+1)%8==0:
                (out/'episodes.json').write_text(json.dumps(rows,indent=2)+'\n')
                print(json.dumps(dict(initials=i+1,sumo_runs=len(rows),elapsed_s=time.perf_counter()-started)),flush=True)
    (out/'episodes.json').write_text(json.dumps(rows,indent=2)+'\n')
    summary={}
    for method in factories:
        data=[r for r in rows if r['controller']==method]
        timings=[t for r in data for t in r['controller_ms']]
        summary[method]=dict(runs=len(data),completed=sum(r['completed'] for r in data),
            collisions=sum(r['collision'] for r in data),aborted=sum(r['aborted'] for r in data),
            team_cost=float(np.mean([r['team_cost'] for r in data])),
            individual_cost=np.mean([r['per_vehicle_cost'] for r in data],axis=0).tolist(),
            individual_time=np.mean([r['per_vehicle_finish'] for r in data],axis=0).tolist(),
            individual_variation=np.mean([r['per_vehicle_variation'] for r in data],axis=0).tolist(),
            controller_p95_ms=float(np.percentile(timings,95)) if timings else None)
    record=dict(protocol=protocol,results=summary,sumo_runs=len(rows),elapsed_s=time.perf_counter()-started)
    (out/'summary.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('paths',nargs='+');p.add_argument('--out',required=True)
    p.add_argument('--n',type=int,default=128);p.add_argument('--seed-base',type=int,default=59000001)
    a=p.parse_args();run(a.paths,a.out,a.n,a.seed_base)
