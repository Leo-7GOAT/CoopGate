"""Small, fixed-budget actual SUMO check of stepwise nominal joint admission."""
import argparse,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
import torch
from coopgate.advantage_gate_v18 import JointAdvantageGate
from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.cooperative_v15 import JointPolicyGate
from coopgate.joint_planner_v15 import JointRolloutGate
from coopgate.disturbance_v9 import DisturbedMergeEnv
from experiment_ego_v13 import run_episode
from verify_development_v14 import trace_audit

ROOT=Path(__file__).resolve().parents[1]


def run(path,out,n=16,seed_base=61500001):
    if n<4 or n%2:raise ValueError('Even regime split required')
    torch.set_num_threads(1);path,out=Path(path),Path(out)
    model,meta=load_coupled(path)
    if meta['updates']!=meta['target_updates']:raise ValueError('Incomplete model')
    out.mkdir(parents=True,exist_ok=False)
    factories={'baseline':lambda x:JointRolloutGate(),'full_marl':lambda x:JointPolicyGate(model),
               'nominal_action_gate':lambda x:JointAdvantageGate(model)}
    modes=['clean','uniform','bias','pulse']
    files=list((ROOT/'src/coopgate').glob('*.py'))+[Path(__file__),ROOT/'tools/experiment_ego_v13.py',ROOT/'tools/verify_development_v14.py']
    p=dict(purpose='Stepwise nominal joint admission DEVELOPMENT; not a calibrated disturbance-performance claim',
        model_path=str(path).replace('\\','/'),model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        methods=list(factories),modes=modes,n_initials=n,seed_base=seed_base,expected_sumo_runs=n*len(modes)*len(factories),
        source_sha256={str(f.resolve().relative_to(ROOT)).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    (out/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
    for source in p['source_sha256']:
        target=out/'source'/source;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/source,target)
    rows=[];started=time.perf_counter()
    with DisturbedMergeEnv() as env:
        for i in range(n):
            seed=seed_base+i;regime='nominal' if i<n//2 else 'dense'
            for mi,mode in enumerate(modes):
                for method in np.random.default_rng(seed+mi+1819).permutation(list(factories)):
                    result,trace,initial=run_episode(env,seed,regime,mode,factories[method],True)
                    trace_audit(trace,result)
                    cost=np.array(result['per_vehicle_finish'])+.05*np.array(result['per_vehicle_variation'])
                    result.update(controller=str(method),initial=initial.tolist(),per_vehicle_cost=cost.tolist(),team_cost=float(cost.mean()),
                        learned_action_steps=sum(t.get('nominal_joint_action_admitted',False) for t in trace),control_steps=len(trace),
                        forecast_transitions=sum(t.get('performance_forecast_transitions',0) for t in trace),
                        internal_planning_transitions=sum(t.get('performance_model_transitions',0) for t in trace))
                    rows.append(result);(out/f'trace_{i}_{mode}_{method}.json').write_text(json.dumps(trace,indent=2)+'\n')
            (out/'episodes.json').write_text(json.dumps(rows,indent=2)+'\n')
            print(json.dumps(dict(initials=i+1,sumo_runs=len(rows),elapsed_s=time.perf_counter()-started)),flush=True)
    if len(rows)!=p['expected_sumo_runs']:raise RuntimeError('Incomplete fixed budget')
    lookup={(r['seed'],r['disturbance'],r['controller']):r for r in rows};summaries={}
    for method in factories:
        data=[r for r in rows if r['controller']==method]
        differences=[]
        for seed in range(seed_base,seed_base+n):
            cluster=[]
            for mode in modes:
                row=lookup[seed,mode,method];base=lookup[seed,mode,'baseline']
                delta=np.column_stack((row['per_vehicle_cost'],row['per_vehicle_finish']))-np.column_stack((base['per_vehicle_cost'],base['per_vehicle_finish']))
                if method=='nominal_action_gate' and row['learned_action_steps']==0:
                    np.testing.assert_allclose(delta,0,atol=1e-9,rtol=0)
                if method=='nominal_action_gate' and mode=='clean':
                    assert np.all(delta<=1e-9),'Nominal telescoping guarantee failed'
                cluster.append(delta)
            differences.append(cluster)
        differences=np.array(differences)
        summaries[method]=dict(completed=sum(r['completed'] for r in data),runs=len(data),
            collisions=sum(r['collision'] for r in data),aborted=sum(r['aborted'] for r in data),
            contract_violations=sum(r['contract_violations'] for r in data),
            mean_cost=float(np.mean([r['team_cost'] for r in data])),
            cost_gain_per_vehicle=-differences[:,:,:,0].mean((0,1)),
            harmful_initials=int(np.any(differences>1e-9,axis=(1,2,3)).sum()),
            learned_action_steps=sum(r['learned_action_steps'] for r in data),
            control_steps=sum(r['control_steps'] for r in data),
            p95_controller_ms=float(np.percentile([t for r in data for t in r['controller_ms']],95)))
        summaries[method]['cost_gain_per_vehicle']=summaries[method]['cost_gain_per_vehicle'].tolist()
    result=dict(purpose=p['purpose'],actual_sumo_runs=len(rows),results=summaries,elapsed_s=time.perf_counter()-started)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('path');p.add_argument('--out',required=True)
    p.add_argument('--n',type=int,default=16);p.add_argument('--seed-base',type=int,default=61500001)
    a=p.parse_args();run(a.path,a.out,a.n,a.seed_base)
