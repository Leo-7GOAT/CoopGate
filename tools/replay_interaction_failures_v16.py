"""Actual SUMO reproduction of selected mixed-policy failures, not prevalence."""
import argparse,copy,hashlib,json,shutil
from pathlib import Path
import numpy as np
import torch
from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.cooperative_v15 import JointPolicyGate
from coopgate.joint_planner_v15 import JointRolloutGate
from coopgate.disturbance_v9 import DisturbedMergeEnv
from experiment_ego_v13 import run_episode
from verify_development_v14 import trace_audit

ROOT=Path(__file__).resolve().parents[1]


def run(out):
    torch.set_num_threads(1);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    folder=ROOT/'artifacts/v16/development_models_stage2/coupled_1601'
    model,_=load_coupled(folder/'policy.pt');initial,_=load_coupled(folder/'initial.pt')
    mixed=copy.deepcopy(model);mixed.actors[1]=copy.deepcopy(initial.actors[1])
    ablation_path=ROOT/'artifacts/v16/development_stage2/actor_ablation/results.json'
    ablation=json.loads(ablation_path.read_text())['coupled_1601']
    failures=[r for r in ablation['restore_1']['records'] if not r['complete']]
    if not failures:raise ValueError('No selected failures to reproduce')
    fs={'full':lambda x:JointPolicyGate(model),'restore_1':lambda x:JointPolicyGate(mixed),
        'all_initial':lambda x:JointPolicyGate(initial),'team_planner':lambda x:JointRolloutGate()}
    files=list((ROOT/'src/coopgate').glob('*.py'))+[Path(__file__),ROOT/'tools/experiment_ego_v13.py',ROOT/'tools/verify_development_v14.py']
    protocol=dict(purpose='Post-hoc illustrative failure reproduction; not a random-sample failure-rate estimate',
        cases=failures,methods=list(fs),expected_sumo_runs=len(failures)*len(fs),
        policy_sha256=hashlib.sha256((folder/'policy.pt').read_bytes()).hexdigest(),
        initial_sha256=hashlib.sha256((folder/'initial.pt').read_bytes()).hexdigest(),
        ablation_sha256=hashlib.sha256(ablation_path.read_bytes()).hexdigest(),
        source_sha256={str(p.resolve().relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for name in protocol['source_sha256']:
        target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    rows=[]
    with DisturbedMergeEnv() as env:
        for index,case in enumerate(failures):
            for name,build in fs.items():
                result,trace,state=run_episode(env,case['seed'],case['regime'],case['disturbance'],build,True)
                result.update(controller=name,initial=state.tolist());rows.append(result)
                (out/f'trace_{index}_{name}.json').write_text(json.dumps(trace,indent=2)+'\n')
                trace_audit(trace,result)
                if name in ablation:
                    expected=next(r for r in ablation[name]['records'] if r['seed']==case['seed'] and r['disturbance']==case['disturbance'])
                    np.testing.assert_allclose(result['per_vehicle_finish'],expected['time'],atol=1e-9,rtol=0)
                    np.testing.assert_allclose(result['per_vehicle_variation'],expected['variation'],atol=1e-9,rtol=0)
                    assert result['completed']==expected['complete']
                assert not result['collision'] and not result['aborted'] and result['contract_violations']==0
    (out/'episodes.json').write_text(json.dumps(rows,indent=2)+'\n')
    summary={name:dict(runs=sum(r['controller']==name for r in rows),completed=sum(r['completed'] for r in rows if r['controller']==name)) for name in fs}
    record=dict(actual_sumo_runs=len(rows),results=summary,scope='Selected illustrative cases, no population failure-rate claim')
    (out/'summary.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();run(a.out)
