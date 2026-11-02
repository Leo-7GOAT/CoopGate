"""Offline observed-history observer replay and an explicitly reused prefix probe.

True parameters are used ONLY to audit containment,never for predictions.
Prefix predictions use a fixed zero-innovation tape and observed-history
low/mid/high retention representatives. This is NOT a new closed-loop gate.
"""
from itertools import product
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from coopgate.retention_set_observer import RetentionSetObserver
from coopgate.actuator_prediction_roundoff_v31 import plant_arrays
from coopgate.actuator_recovery_v34 import class_bounds
from coopgate.fast_v8 import parameters
from coopgate.merge_v4 import MergeConfig
from coopgate.memoized_recovery_v37 import make_cache,CAPACITY
from coopgate.scalar_nominal_recovery import forecast_kernel

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/mappo_ctde_20261101'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def probe():
    torch.set_num_threads(1);out=BASE/'retention_observer_probe_20261102';out.mkdir(exist_ok=False)
    sources=[BASE/f'development/parameter_scenario_grid_20261102_shard{i}' for i in range(4)]
    sources.append(BASE/'development/scalar_nominal_sumo_20261102');records=[];hashes={}
    for p in sources:
        proof=json.loads((p/'verification.json').read_text());complete=json.loads((p/'collection_complete.json').read_text())
        if not proof['all_match'] or sha(p/'episodes.jsonl')!=complete['episodes_sha256']:raise ValueError('Reviewed complete unchanged source required')
        hashes[p.name]=sha(p/'episodes.jsonl')
        for line in (p/'episodes.jsonl').read_text().splitlines():
            r=json.loads(line);method=r['row']['method_id']
            if method.startswith('parameter_') or method.startswith('scalar_'):
                if r['row']['plant_name'] in ('lag_mild','lag_strong'):records.append(r)
    files=[Path(__file__),ROOT/'src/coopgate/retention_set_observer.py',ROOT/'tests/test_retention_set_observer.py',ROOT/'docs/retention_set_observer_20261102_protocol.md']
    write(out/'plan.json',dict(source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files},source_data_sha256=hashes,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        observer_records=len(records),cells=32,representatives='All27 Cartesian lower/mid/upper retained-envelope rho points',
        prefix_case=[97110131,'dense','lag_strong','clean'],prediction_innovations='Fixed all-zero tape,NOT realized future',
        true_parameters_used_for_prediction=False,true_parameters_used_for_containment_audit_only=True,
        online_gate_implemented=False,extra_sumo_executions=0,risk_certificate=False))
    checked=0;counts=[];examples=[];packed=parameters(MergeConfig());started=time.perf_counter()
    for r in records:
        row=r['row'];observer=RetentionSetObserver(row['plant_name']);state=np.array(row['initial']);prev=np.zeros(3)
        _,_,actual_rho=plant_arrays(row['plant'],.1);actual_memory=np.zeros(3)
        containing=(observer.lower<=actual_rho[:,None])&(actual_rho[:,None]<=observer.upper)
        for step,item in enumerate(r['trace']):
            status=observer.observe(state,prev)
            if not status['consistent'] or not observer.feasible[containing].all():raise ValueError('Observer eliminated actual parameter cell')
            mem=np.broadcast_to(actual_memory[:,None],observer.lower.shape)
            if np.any(mem[containing]<observer.memory_lower[containing]-1e-10) or np.any(mem[containing]>observer.memory_upper[containing]+1e-10):
                raise ValueError('Actual memory escaped its retained parameter-cell interval')
            checked+=1;counts.append(status['feasible_cells']);command=np.array(item['command'])
            if r['key'][:4]==[97110131,'dense','lag_strong','clean'] and item['actual_learning_used']:
                lower,upper=observer.retention_bounds();points=(lower,.5*(lower+upper),upper);bad=[];complete=0;deltas=[]
                for choices in product(range(3),repeat=3):
                    rho=np.array([points[c][i] for i,c in enumerate(choices)]);memory=np.zeros(3)
                    for past in observer.commands:
                        for _ in range(5):memory=rho*memory+(1-rho)*past
                    mlo=np.array(item['actuator_memory_lower']);mhi=np.array(item['actuator_memory_upper'])
                    if np.any(memory<mlo-1e-12) or np.any(memory>mhi+1e-12):raise ValueError('Online-reconstructed point outside common memory interval')
                    args=(state,prev,memory,mlo,mhi,step,np.zeros((250,3)),np.ones(3),np.zeros(3),rho,class_bounds(row['plant_name']),packed)
                    cache,stats=make_cache();before=forecast_kernel(*args,np.zeros(3),False,cache,stats,CAPACITY)
                    cache,stats=make_cache();after=forecast_kernel(*args,command,True,cache,stats,CAPACITY)
                    valid=bool(before[2] and after[2]);complete+=valid
                    delta=after[0]-before[0];deltas.append(delta)
                    if not valid or np.any(delta>1e-10) or np.any(after[1]-before[1]>.05+1e-10):bad.append(dict(choices=choices,rho=rho.tolist(),complete=valid,differences=delta.tolist()))
                examples.append(dict(case=r['key'],step=step,feasible_cells=status['feasible_cells'],retention_lower=lower.tolist(),retention_upper=upper.tolist(),
                    probe_paths=27,complete_forecast_pairs=complete,violating_paths=len(bad),worst_value_differences=np.max(deltas,axis=0).tolist(),violations=bad))
            for _ in range(5):actual_memory=actual_rho*actual_memory+(1-actual_rho)*command
            state=np.array(item['x']);prev=command
    result=dict(observer_records=len(records),independent_reused_initials=len({tuple(r['key'][:2]) for r in records}),
        decision_boundaries_checked=checked,all_actual_parameter_cells_and_memories_retained=True,
        mean_feasible_cells=np.mean(counts,axis=0).tolist(),examples=examples,
        online_gate_implemented=False,extra_sumo_executions=0,extra_independent_initials=0,risk_certificate=False,
        elapsed_s=time.perf_counter()-started,
        inference='Observed-history representation probe on reused source,not closed-loop repair/generalization. Actual rho/memory used only AFTER observer updates for containment audit; predictions receive no oracle parameters or future tape.')
    write(out/'report.json',result);print(json.dumps({k:v for k,v in result.items() if k!='examples'}),flush=True)


if __name__=='__main__':probe()
