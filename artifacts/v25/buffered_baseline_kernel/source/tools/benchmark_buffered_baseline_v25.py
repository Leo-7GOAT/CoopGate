"""Frozen paired kernel replay on every phase-gate input of one OLD initial.

This is a kernel timing diagnostic, not full gate latency, a new traffic
experiment, or additional independent effectiveness evidence.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np

from coopgate.buffered_baseline_v25 import deduplicated_joint_rollout as buffered
from coopgate.deduplicated_planner_v19 import deduplicated_joint_rollout as reference
from coopgate.episode_store_v21 import encode, observations
from coopgate.fast_v8 import parameters
from coopgate.merge_v4 import MergeConfig
import execute_prospective_v23 as execution

ROOT=execution.ROOT
OLD=execution.OUTPUTS['preflight']
OUTPUT=ROOT/'artifacts/v25/buffered_baseline_kernel'
SOURCES=('src/coopgate/buffered_baseline_v25.py','tools/benchmark_buffered_baseline_v25.py','tests/test_buffered_baseline_v25.py')


def freeze():
    if OUTPUT.exists(): raise FileExistsError('Preserve all earlier kernel measurements')
    old=execution.read(OLD/'frozen/plan.json')
    execution.complete_stage(OLD,old,'preflight')
    path=OLD/'preflight/blocks/00000/observations.sqlite'
    seal=execution.read(Path(str(path)+'.seal.json'))
    unique={};occurrences=0
    for case,row,trace in observations(path,seal['metadata'],seal['cases']):
        if case[-1]!='collection' or not case[2].startswith('phase05_'):continue
        state,previous=row['initial'],[0.,0.,0.]
        for step,item in enumerate(trace):
            values=dict(state=state,previous=previous,step=step)
            key=execution.value_digest(values)
            if key not in unique: unique[key]=dict(**values,occurrences=[])
            unique[key]['occurrences'].append(dict(case=case,step=step))
            occurrences+=1
            state,previous=item['x'],item['command']
    sources={**old['source_sha256'],**{name:execution.digest(ROOT/name) for name in SOURCES}}
    execution.sources_committed([ROOT/name for name in sources])
    for name,expected in sources.items():
        if execution.digest(ROOT/name)!=expected: raise ValueError('Original source changed: '+name)
    OUTPUT.mkdir(parents=True)
    plan=dict(utc=execution.utc(),source_sha256=sources,
        input_sha256={p.relative_to(ROOT).as_posix():execution.digest(p) for p in
            (path,Path(str(path)+'.seal.json'),OLD/'frozen/plan.json',OLD/'preflight/execution/supervisor_exit.json')},
        cases=list(unique.values()),original_decision_occurrences=occurrences,unique_inputs=len(unique),
        repeats=3,order='Alternate reference/buffered order by (input index + repetition) parity',
        configuration=parameters(MergeConfig()).tolist(),sweeps=2,
        compare='Exact command, objective, rule cost and work count; errors preserved before assertions',
        timing='Warm each function before paired timings; save wall and process CPU time under concurrent desktop load',
        full_gate_runtime_measured=False,new_sumo_runs=0,new_independent_initials=0,
        active_calibration_outcomes_read=False,primary_controller_changed=False)
    execution.exclusive_json(OUTPUT/'plan.json',plan)
    for name in SOURCES:
        destination=OUTPUT/'source'/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,destination)
    execution.exclusive_json(OUTPUT/'sha256.json',execution.manifest(OUTPUT))
    print(json.dumps(dict(unique_inputs=len(unique),original_decision_occurrences=occurrences)),flush=True)


def numerical(value):
    return dict(command=value[0].tolist(),objective=float(value[1]),rule_cost=float(value[2]),work=int(value[3]))


def run():
    execution.require_committed([OUTPUT/'plan.json',OUTPUT/'sha256.json'])
    execution.verify_manifest(OUTPUT)
    plan=execution.read(OUTPUT/'plan.json')
    for name,expected in {**plan['source_sha256'],**plan['input_sha256']}.items():
        if execution.digest(ROOT/name)!=expected:raise ValueError('Frozen kernel input changed: '+name)
    destination=OUTPUT/'results';destination.mkdir(exist_ok=False)
    packed=np.array(plan['configuration'])
    records=[]
    for index,case in enumerate(plan['cases']):
        state,previous=np.array(case['state']),np.array(case['previous'])
        expected=numerical(reference(state,previous,case['step'],packed,plan['sweeps']))
        warm=numerical(buffered(state,previous,case['step'],packed,plan['sweeps']))
        execution.exclusive_json(destination/f'warm_{index}.json',dict(reference=expected,buffered=warm))
        if warm!=expected:raise ValueError('Buffered baseline warm result differs')
        for repeat in range(plan['repeats']):
            order=('reference','buffered') if (index+repeat)%2==0 else ('buffered','reference')
            for name in order:
                fn=reference if name=='reference' else buffered
                cpu=time.process_time();wall=time.perf_counter()
                result=fn(state,previous,case['step'],packed,plan['sweeps'])
                wall_ms=(time.perf_counter()-wall)*1000.;cpu_ms=(time.process_time()-cpu)*1000.
                row=dict(index=index,repeat=repeat,implementation=name,wall_ms=wall_ms,cpu_ms=cpu_ms,result=numerical(result))
                execution.exclusive_json(destination/f'input_{index}_repeat_{repeat}_{name}.json',row)
                if row['result']!=expected:raise ValueError('Measured baseline result differs')
                records.append(row)
    totals={name:{key:sum(row[key] for row in records if row['implementation']==name)
                  for key in ('wall_ms','cpu_ms')} for name in ('reference','buffered')}
    reduction={key:100*(1-totals['buffered'][key]/totals['reference'][key]) for key in ('wall_ms','cpu_ms')}
    report=dict(plan_sha256=execution.digest(OUTPUT/'plan.json'),unique_inputs=len(plan['cases']),
        original_decision_occurrences=plan['original_decision_occurrences'],
        timed_calls=len(records),warm_calls=2*len(plan['cases']),all_numerical_results_exact=True,
        totals=totals,reduction_percent=reduction,new_sumo_runs=0,new_independent_initials=0,
        full_gate_speedup_established=False,hard_real_time_established=False,
        active_calibration_outcomes_read=False,primary_controller_changed=False,timing=plan['timing'])
    execution.exclusive_json(destination/'report.json',report)
    execution.exclusive_json(destination/'sha256.json',execution.manifest(destination))
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('freeze','run'))
    args=parser.parse_args();globals()[args.action]()
