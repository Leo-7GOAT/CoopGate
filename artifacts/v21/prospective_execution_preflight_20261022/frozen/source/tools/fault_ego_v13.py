"""Engineering fault runs; deliberately separate from calibrated performance."""
import json
from pathlib import Path
import numpy as np
from coopgate.disturbance_v9 import DisturbedMergeEnv
from coopgate.ego_confidence_v13 import EgoConfidenceGate
from coopgate.robust_v9 import robust_certificate
from coopgate.fast_v8 import parameters
from coopgate.merge_v4 import MergeConfig
from experiment_ego_v13 import run_episode
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/v13/ego'


def run():
    cal=json.loads((D/'calibration/calibration.json').read_text());res=json.loads((D/'calibration/residuals.json').read_text())
    # Pick the first admitted calibration initial, never a cherry-picked test.
    item=next(r for r in res if r['policy']==701 and r['predicted'][0]+cal['701']['groups'][r['group']]['q']<=-.05
        and r['predicted'][1]+cal['701']['groups'][r['group']]['q']<=0.)
    rows=[];packed=parameters(MergeConfig())
    def factory(initial,case):
        path=D/'seed_701/policy.pt' if case=='runtime_fault' else D/'intentionally_missing_policy.pt'
        gate=EgoConfidenceGate(path,initial,cal['701'])
        if case=='runtime_fault':
            assert gate.accept
            select=gate.learned.select
            def fail(x,previous,unused=None):
                if gate.step==3:raise RuntimeError('deliberately injected inference failure')
                return select(x,previous,unused)
            gate.learned.select=fail
        return gate
    with DisturbedMergeEnv() as env:
        for case in ('missing_checkpoint','runtime_fault'):
            for mode in ('clean','uniform','bias','pulse'):
                row,trace,initial=run_episode(env,item['seed'],item['regime'],mode,lambda x:factory(x,case),True)
                assert row['completed'] and not row['collision'] and not row['aborted'] and not row['contract_violations']
                if case=='runtime_fault':assert row['fault_fallback'] and all(t['fault_fallback'] for t in trace[3:])
                else:assert not row['accepted'] and row['gate_reason']=='prediction_failed'
                x=initial.copy();previous=np.zeros(3)
                for t in trace:
                    command=np.array(t['command']);assert robust_certificate(x,previous,command,packed,.5)
                    x=np.array(t['x']);previous=command
                rows.append({**row,'case':case})
    result={'purpose':'engineering takeover; not covered by the no-fault performance calibration theorem',
        'actual_sumo_runs':len(rows),'initial_source':'first admitted calibration initial for seed 701','rows':rows}
    def finite(value):
        if isinstance(value,dict):return {k:finite(v) for k,v in value.items()}
        if isinstance(value,list):return [finite(v) for v in value]
        if isinstance(value,float) and not np.isfinite(value):return None
        return value
    (D/'fault_audit.json').write_text(json.dumps(finite(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps({'fault_sumo_runs':len(rows),'all_completed':True}))


if __name__=='__main__':run()
