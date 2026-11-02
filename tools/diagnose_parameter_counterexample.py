"""Privileged OFFLINE baseline-advantage decomposition; never an online input."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from coopgate.actuator_prediction_roundoff_v31 import plant_arrays
from coopgate.actuator_recovery_v34 import class_bounds
from coopgate.disturbance_v9 import disturbance_tape
from coopgate.fast_v8 import parameters
from coopgate.merge_v4 import MergeConfig
from coopgate.memoized_recovery_v37 import make_cache,CAPACITY
from coopgate.scalar_nominal_recovery import forecast_kernel

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/mappo_ctde_20261101'


def diagnose():
    torch.set_num_threads(1);source=BASE/'development/scalar_nominal_sumo_20261102'
    proof=json.loads((source/'verification.json').read_text());native=json.loads((source/'native_receipt.json').read_text())
    if not proof['all_match'] or native['review_actual_native_exit_code']!=0:raise ValueError('Fully reviewed actual SUMO source required')
    raw=(source/'episodes.jsonl').read_bytes();records=[json.loads(s) for s in raw.splitlines()]
    lookup={tuple(r['key']):r for r in records};packed=parameters(MergeConfig());examples=[]
    for key,r in lookup.items():
        if not key[-1].startswith('scalar_'):continue
        row=r['row'];base=lookup[key[:-1]+('baseline',)]['row']
        observed=np.stack((np.array(row['per_vehicle_cost'])-base['per_vehicle_cost'],
                           np.array(row['per_vehicle_finish'])-base['per_vehicle_finish']),axis=-1)
        if not np.any(observed>1e-9):continue
        gain,drag,rho=plant_arrays(row['plant'],.1);memory=np.zeros(3);state=np.array(row['initial']);prev=np.zeros(3)
        tape=disturbance_tape(row['seed'],row['disturbance'],.1,250);decisions=[];total=np.zeros((3,2))
        for step,item in enumerate(r['trace']):
            command=np.array(item['command'])
            if item['actual_learning_used']:
                args=(state,prev,memory,np.array(item['actuator_memory_lower']),np.array(item['actuator_memory_upper']),
                      step,tape,gain,drag,rho,class_bounds(row['plant_name']),packed)
                cache,stats=make_cache();before=forecast_kernel(*args,np.zeros(3),False,cache,stats,CAPACITY)
                cache,stats=make_cache();after=forecast_kernel(*args,command,True,cache,stats,CAPACITY)
                if not before[2] or not after[2]:raise ValueError('This cost decomposition requires complete reference tails')
                delta=after[0]-before[0];total+=delta
                decisions.append(dict(step=step,oracle_baseline_advantage=delta.tolist(),
                    oracle_regression=bool(np.any(delta>1e-9)),
                    selected_nominal_difference=item['selected_bank_differences'][0],
                    all_finite_scenario_differences_nonpositive=bool(np.all(np.array(item['selected_bank_differences'])<=1e-10)),
                    scenario_paths=item['forecast_bank_size']))
            for _ in range(5):memory=rho*memory+(1-rho)*command
            state=np.array(item['x']);prev=command
        np.testing.assert_allclose(total,observed,rtol=0,atol=1e-9)
        examples.append(dict(case=list(key),per_vehicle_cost_and_time_regression=observed.tolist(),
            sum_true_baseline_advantages=total.tolist(),telescoping_identity_residual_max=float(np.abs(total-observed).max()),
            decisions=decisions))
    result=dict(privileged_offline_only=True,uses_true_plant_memory_and_future_tape=True,
        online_controller_changed=False,extra_sumo_executions=0,extra_independent_initials=0,
        independent_adverse_initials=len({tuple(e['case'][:2]) for e in examples}),examples=examples,
        source_episodes_sha256=hashlib.sha256(raw).hexdigest(),
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        mathematical_identity='J_G-J_B = sum_t [c_G,t + V_B(s_G,t+1)-V_B(s_G,t)]; baseline decisions have zero residual. Vector cost AND arrival components.',
        inference='Selected adverse-case mechanism only,not a population risk estimate or fair online comparator.')
    out=BASE/'parameter_counterexample_diagnosis_20261102.json'
    with out.open('x',encoding='utf-8',newline='\n') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':diagnose()
