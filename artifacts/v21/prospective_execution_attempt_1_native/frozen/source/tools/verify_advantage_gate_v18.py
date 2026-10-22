"""Verify the nominal telescoping identity and reconstruct action admissions."""
import argparse,hashlib,json
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
from verify_single_v12 import equivalent

ROOT=Path(__file__).resolve().parents[1]


def verify(directory,replay=False):
    torch.set_num_threads(1);directory=Path(directory)
    p=json.loads((directory/'protocol.json').read_text());rows=json.loads((directory/'episodes.json').read_text())
    lookup={(r['seed'],r['disturbance'],r['controller']):r for r in rows}
    seeds=range(p['seed_base'],p['seed_base']+p['n_initials'])
    assert len(rows)==len(lookup)==p['expected_sumo_runs']
    assert set(lookup)=={(s,m,k) for s in seeds for m in p['modes'] for k in p['methods']}
    for source,digest in p['source_sha256'].items():
        assert hashlib.sha256((directory/'source'/source).read_bytes()).hexdigest()==digest,source
        assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==digest,source
    path=ROOT/p['model_path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==p['model_sha256']
    model,_=load_coupled(path)
    fs={'baseline':lambda:JointRolloutGate(),'full_marl':lambda:JointPolicyGate(model),'nominal_action_gate':lambda:JointAdvantageGate(model)}
    actions=traces=identities=0
    for i,seed in enumerate(seeds):
        for mode in p['modes']:
            for name,build in fs.items():
                trace=json.loads((directory/f'trace_{i}_{mode}_{name}.json').read_text());row=lookup[seed,mode,name]
                trace_audit(trace,row);traces+=1
                assert row['learned_action_steps']==sum(t.get('nominal_joint_action_admitted',False) for t in trace)
                if name=='nominal_action_gate' and mode=='clean':
                    base=lookup[seed,mode,'baseline']
                    actual=np.column_stack((row['per_vehicle_cost'],row['per_vehicle_finish']))-np.column_stack((base['per_vehicle_cost'],base['per_vehicle_finish']))
                    total=sum((np.array(t['nominal_difference']) for t in trace if t['nominal_joint_action_admitted']),start=np.zeros((3,2)))
                    np.testing.assert_allclose(actual,total,atol=1e-9,rtol=0)
                    assert np.all(actual<=1e-9);identities+=1
                if i in (0,1,p['n_initials']//2,p['n_initials']//2+1):
                    controller=build();x=np.array(row['initial']);prev=np.zeros(3)
                    for item in trace:
                        command,info=controller.select(x,prev)
                        np.testing.assert_allclose(command,item['command'],atol=1e-9,rtol=0)
                        if name=='nominal_action_gate':
                            assert info['nominal_joint_action_admitted']==item['nominal_joint_action_admitted']
                            np.testing.assert_allclose(info['nominal_difference'],item['nominal_difference'],atol=1e-9,rtol=0)
                        x=np.array(item['x']);prev=np.array(item['command']);actions+=1
    replays=0
    if replay:
        with DisturbedMergeEnv() as env:
            for i in (0,p['n_initials']//2):
                seed=p['seed_base']+i;regime='nominal' if i==0 else 'dense'
                for mode in p['modes']:
                    for name,build in fs.items():
                        result,trace,_=run_episode(env,seed,regime,mode,lambda x,b=build:b(),True)
                        for key in ('per_vehicle_finish','per_vehicle_variation','completed','collision','aborted','contract_violations'):
                            equivalent(result[key],lookup[seed,mode,name][key])
                        saved=json.loads((directory/f'trace_{i}_{mode}_{name}.json').read_text())
                        np.testing.assert_allclose([t['command'] for t in trace],[t['command'] for t in saved],atol=1e-9,rtol=0);replays+=1
    r=dict(actual_sumo_runs=len(rows),audited_physical_traces=traces,reconstructed_controls=actions,
        nominal_telescoping_identities=identities,additional_sumo_replays=replays,
        disturbance_performance_certified=False,episodes_sha256=hashlib.sha256((directory/'episodes.json').read_bytes()).hexdigest())
    print(json.dumps(r),flush=True);return r


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--replay',action='store_true');p.add_argument('--out')
    a=p.parse_args();r=verify(a.directory,a.replay)
    if a.out:Path(a.out).write_text(json.dumps(r,indent=2)+'\n')
