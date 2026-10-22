"""Actual SUMO runtime-recovery tests; no statistical admission claim.

Initial admission is deliberately forced to exercise recovery from real learned
trajectories. A forced admission is not a calibrated-confidence experiment.
"""
import argparse,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
import torch
from coopgate.profile_recovery_v15 import RecoveringProfile
from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.cooperative_v15 import JointPolicyGate
from coopgate.policy_iteration_v14 import load_policy
from coopgate.ego_v13 import EgoPolicyGate,EgoRolloutGate
from coopgate.joint_planner_v15 import JointRolloutGate
from coopgate.disturbance_v9 import DisturbedMergeEnv
from experiment_ego_v13 import run_episode
from verify_development_v14 import trace_audit

ROOT=Path(__file__).resolve().parents[1]


class InjectedFault:
    def __init__(self,inner,fault,at):
        self.inner=inner;self.fault=fault;self.at=at;self.step=0

    def select(self,state,previous,unused=None):
        self.inner.step=self.step
        if self.step==self.at:
            if self.fault=='exception':raise RuntimeError('Injected policy execution failure')
            if self.fault=='nan':return np.full(3,np.nan),{}
            if self.fault=='invalid_command':return previous+100.,{}
        result=self.inner.select(state,previous);self.step+=1;return result


def run(out,n=8,seed_base=59600001):
    torch.set_num_threads(1);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    single_path=ROOT/'artifacts/v14/single/seed_901/policy.pt'
    multi_path=ROOT/'artifacts/v16/development_models/coupled_1601/policy.pt'
    single,_=load_policy(single_path);multi,_=load_coupled(multi_path)
    families={'single':(lambda:EgoPolicyGate(single),EgoRolloutGate),
              'multi':(lambda:JointPolicyGate(multi),JointRolloutGate)}
    variants=['baseline','admitted_no_fault','exception','nan','invalid_command','rejected']
    modes=['clean','uniform','bias','pulse']
    sources=list((ROOT/'src/coopgate').glob('*.py'))+[Path(__file__),ROOT/'tools/experiment_ego_v13.py',ROOT/'tools/verify_development_v14.py']
    protocol=dict(purpose='Physical runtime recovery with externally forced admission; not confidence calibration',
        statistical_admission_certified=False,n_initials=n,seed_base=seed_base,variants=variants,modes=modes,
        models={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in (single_path,multi_path)},
        expected_sumo_runs=n*len(modes)*len(variants)*len(families),
        fault_steps='4 for even initial index, 8 for odd; before any vehicle can reach the goal',
        source_sha256={str(p.resolve().relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for name in protocol['source_sha256']:
        target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    rows=[];audited_commands=0;started=time.perf_counter()
    with DisturbedMergeEnv() as env:
        for i in range(n):
            seed=seed_base+i;regime='nominal' if i<n//2 else 'dense';at=4+4*(i%2)
            for mode in modes:
                for family,(primary,baseline) in families.items():
                    saved={}
                    for variant in variants:
                        if variant=='baseline':factory=lambda x,b=baseline:b()
                        else:
                            factory=lambda x,p=primary,b=baseline,v=variant,a=at:RecoveringProfile(
                                InjectedFault(p(),v,a),b(),v!='rejected',predicted=np.zeros((3,2)),upper=np.zeros((3,2)))
                        result,trace,initial=run_episode(env,seed,regime,mode,factory,True)
                        result.update(family=family,variant=variant,initial=initial.tolist(),fault_step=at)
                        rows.append(result);saved[variant]=trace
                        (out/f'trace_{i}_{mode}_{family}_{variant}.json').write_text(json.dumps(trace,indent=2)+'\n')
                        trace_audit(trace,result)
                        assert result['completed'] and not result['collision'] and not result['aborted']
                        assert result['contract_violations']==result['outside_steps']==result['uncertified_steps']==0
                        if variant not in ('baseline','rejected'):
                            assert result['accepted'] is True
                        faulty=variant in ('exception','nan','invalid_command')
                        if faulty:
                            assert result['fault_fallback'] is True and len(trace)>at
                            assert all(not t['fault_fallback'] for t in trace[:at])
                            assert all(t['fault_fallback'] and t['performance_admitted'] for t in trace[at:])
                        if faulty or variant=='rejected':
                            first=at if faulty else 0;controller=baseline()
                            state=initial.copy() if first==0 else np.array(trace[first-1]['x'])
                            previous=np.zeros(3) if first==0 else np.array(trace[first-1]['command'])
                            for k in range(first,len(trace)):
                                controller.step=k;command,_=controller.select(state,previous)
                                np.testing.assert_allclose(command,trace[k]['command'],atol=1e-9,rtol=0);audited_commands+=1
                                state=np.array(trace[k]['x']);previous=np.array(trace[k]['command'])
                    np.testing.assert_allclose([t['command'] for t in saved['baseline']],
                                               [t['command'] for t in saved['rejected']],atol=1e-9,rtol=0)
            (out/'episodes.json').write_text(json.dumps(rows,indent=2)+'\n')
            print(json.dumps(dict(initials=i+1,sumo_runs=len(rows),elapsed_s=time.perf_counter()-started)),flush=True)
    summary=dict(actual_sumo_runs=len(rows),completed=sum(r['completed'] for r in rows),
        injected_fault_episodes=sum(r['variant'] in ('exception','nan','invalid_command') for r in rows),
        fallback_command_reconstructions=audited_commands,all_physical_checks_passed=True,
        statistical_admission_certified=False,elapsed_s=time.perf_counter()-started)
    assert len(rows)==protocol['expected_sumo_runs']
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--n',type=int,default=8)
    p.add_argument('--seed-base',type=int,default=59600001);a=p.parse_args();run(a.out,a.n,a.seed_base)
