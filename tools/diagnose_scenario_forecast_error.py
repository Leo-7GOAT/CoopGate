"""Privileged OFFLINE diagnostic; never available to a deployed selector."""
import json
from pathlib import Path
import numpy as np
import torch
from coopgate.actuator_prediction_roundoff_v31 import plant_arrays
from coopgate.actuator_recovery_v34 import class_bounds
from coopgate.disturbance_v9 import disturbance_tape
from coopgate.fast_v8 import parameters
from coopgate.memoized_recovery_v37 import forecast_kernel,make_cache,CAPACITY
from coopgate.merge_v4 import MergeConfig

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/mappo_ctde_20261101/development'


def diagnose():
    torch.set_num_threads(1);lookup={}
    for shard in range(4):
        p=BASE/f'scenario_score_grid_20261102_shard{shard}'
        if not json.loads((p/'verification.json').read_text())['all_match']:raise ValueError('Complete reviewed source required')
        for line in (p/'episodes.jsonl').read_text().splitlines():
            r=json.loads(line);lookup[tuple(r['key'])]=r
    examples=[];packed=parameters(MergeConfig())
    for key,record in lookup.items():
        if key[-1]!='confidence_4705_0':continue
        row=record['row'];base=lookup[key[:-1]+('baseline',)]['row']
        if not (np.any(np.array(row['per_vehicle_cost'])>np.array(base['per_vehicle_cost'])+1e-9)
                or np.any(np.array(row['per_vehicle_finish'])>np.array(base['per_vehicle_finish'])+1e-9)):continue
        gain,drag,rho=plant_arrays(row['plant'],.1);memory=np.zeros(3);previous=np.zeros(3);state=np.array(row['initial'])
        tape=disturbance_tape(row['seed'],row['disturbance'],.1,250)
        for step,item in enumerate(record['trace']):
            command=np.array(item['command'])
            if item['actual_learning_used']:
                args=(state,previous,memory,np.array(item['actuator_memory_lower']),np.array(item['actuator_memory_upper']),
                      step,tape,gain,drag,rho,class_bounds(row['plant_name']),packed)
                cache,stats=make_cache();before=forecast_kernel(*args,np.zeros(3),False,cache,stats,CAPACITY)
                cache,stats=make_cache();after=forecast_kernel(*args,command,True,cache,stats,CAPACITY)
                delta=after[0]-before[0]
                examples.append(dict(case=list(key[:4]),step=step,both_oracle_forecasts_complete=bool(before[2] and after[2]),
                    oracle_candidate_minus_baseline_values=delta.tolist(),
                    original_nominal_differences=item['selected_bank_differences'][0],
                    oracle_detects_regression=bool(before[2] and after[2] and np.any(delta>1e-9))))
            for _ in range(5):memory=rho*memory+(1-rho)*command
            previous=command;state=np.array(item['x'])
    result=dict(privileged_offline_only=True,uses_true_plant_and_realized_future_tape=True,
        online_controller_changed=False,extra_sumo_executions=0,extra_independent_initials=0,
        purpose='Selected adverse-case diagnosis,not population inference or a fair online comparator',examples=examples)
    path=ROOT/'artifacts/mappo_ctde_20261101/scenario_forecast_error_diagnosis_20261102.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(selected_adopted_decisions=len(examples),oracle_regressions=sum(e['oracle_detects_regression'] for e in examples))),flush=True)


if __name__=='__main__':diagnose()
