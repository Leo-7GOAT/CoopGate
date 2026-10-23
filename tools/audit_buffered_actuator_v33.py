"""Exact actuator-forecast optimization with full old-preflight and failure coverage."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from coopgate.actuator_intervention_v31 import intervention_state
from coopgate.buffered_actuator_v33 import BufferedActuatorGate,forecast as buffered_forecast
from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.disturbance_v9 import disturbance_tape
from coopgate.episode_store_v21 import EpisodeStore,encode,observations
from coopgate.matched_shift_v21 import MatchedShiftedMergeEnv
from coopgate.online_actuator_gate_v32 import ReferenceActuatorGate,forecast
from coopgate.physical_comparison_analysis_v27 import equal_fields,failed
import audit_buffered_scope_v25 as infrastructure
import audit_online_actuator_v32 as previous
from execute_prospective_v23 import check_acknowledgment,exclusive_json,runtime_identity,sources_committed,utc
from experiment_ego_v13 import run_episode
from profile_phase_runtime_v24 import committed,digest
from recover_checkpoint_v19 import atomic_bytes
from recover_interrupted_v21 import compare_replay

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v33/buffered_actuator_audit'
SCRIPT='tools/audit_buffered_actuator_v33.py'
IMPLEMENTATIONS=('original','buffered')
MODELS=('1802','1812','1822')
SOURCE_FILES=('src/coopgate/buffered_actuator_v33.py',SCRIPT,
    'tests/test_buffered_actuator_v33.py','tests/test_buffered_actuator_protocol_v33.py',
    'tools/audit_buffered_scope_v25.py')
PAIR_FIELDS=('seed','scenario','initial','disturbance','radius','plant','external_innovation_radius',
    'physical_error_radius','innovation_tape_sha256')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def audit_groups(integration_cases,failed_baseline_conditions,source_cases):
    source={tuple(k) for k in source_cases}
    selected={tuple(k) for k in integration_cases}
    if len(selected)!=len(integration_cases) or any(k[:-1] not in source or k[-1] not in ('nominal','reference') for k in selected):
        raise ValueError('Unique original integration cases required')
    for condition in failed_baseline_conditions:
        for model in MODELS:
            for kind in ('nominal','reference'):
                case=(*condition,'phase05_'+model,kind)
                if case[:-1] not in source:raise ValueError('Failed baseline condition is outside complete archived scope')
                selected.add(case)
    groups=[]
    for index,case in enumerate(sorted(selected)):
        order=IMPLEMENTATIONS if index%2==0 else IMPLEMENTATIONS[::-1]
        groups.append(dict(block_index=index,case=list(case),
            source='preflight' if case in {tuple(k) for k in integration_cases} else 'all_old_baseline_failures',
            executions=[list(case)+[kind] for kind in order]))
    return groups


def baselines():
    plan=read(previous.BASELINES/'plan.json')
    return {k:r['row'] for k,r,_ in observations(previous.BASELINES/'inputs.sqlite',plan['input_metadata'],plan['input_cases'])}


def old_preflight():
    plan=read(previous.OUT/'plan.json')
    return {k:(r,t) for k,r,t in observations(previous.OUT/'observations.sqlite',
        dict(plan_sha256=digest(previous.OUT/'plan.json'),stage='online_integration'),plan['integration_schedule'])}


def freeze():
    old=previous.inputs()
    report=read(previous.OUT/'analysis/report.json')
    receipt=read(previous.OUT/'completion_verification.json')
    if not report['all_physical_and_online_input_reviews_pass'] or receipt['report_sha256']!=digest(previous.OUT/'analysis/report.json'):
        raise ValueError('Completed online preflight required')
    source=read(previous.INTERVENTIONS/'plan.json')
    failures=sorted(k[:-1] for k,r in baselines().items() if failed(r))
    groups=audit_groups(old['integration_schedule'],failures,source['input_cases'])
    hashes=dict(old['source_sha256']);hashes.update({p:digest(ROOT/p) for p in SOURCE_FILES})
    for name,expected in hashes.items():
        if digest(ROOT/name)!=expected:raise ValueError('Frozen source changed: '+name)
    sources_committed([ROOT/p for p in hashes])
    paths=[previous.OUT/n for n in ('plan.json','observations.sqlite','observations.sqlite.seal.json','analysis/report.json','completion_verification.json')]
    paths += [previous.BASELINES/n for n in ('plan.json','inputs.sqlite','inputs.sqlite.seal.json')]
    OUT.mkdir(parents=True,exist_ok=False)
    exclusive_json(OUT/'plan.json',dict(utc=utc(),source_sha256=hashes,
        inputs_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in paths},runtime=runtime_identity(),
        models=old['models'],kernel_cases=source['admission_cases'],kernel_branches=698,
        groups=groups,blocks=len(groups),actual_sumo_runs=2*len(groups),workers=1,
        complete_preflight_cases=len(old['integration_schedule']),all_old_failed_baseline_conditions=[list(k) for k in failures],
        reference_parameters=old['reference_parameters'],new_independent_initials=0,new_risk_certificate=False,
        active_calibration_outcomes_read=False,ongoing_full_grid_outcomes_read=False,
        change='Only the forecast physical certificate and rule-tail dependency use prevalidated reusable buffers. The original search body, model recurrence, inputs, all candidate/bank decisions and all work counters remain unchanged. No faster-math or pruning change.',
        coverage='Every completed V32 integration case, plus every old V30 failed-baseline condition crossed with all three trained models and both online prediction variants. All 698 old actual intervention branches validate predictor identity first.',
        measurement='Alternate implementation order by fixed pair index; time every complete select call including terminal exceptions. Compare every original non-timing field and command, and independently reconstruct physical/online-input evidence.',
        interpretation='Implementation/timing study on old development initials and explicitly failure-enriched conditions; no efficacy, population-risk, homogeneous-load or hard-real-time inference. Both running frozen studies remain unchanged.'))
    print(json.dumps(dict(kernel_branches=698,paired_cases=len(groups),actual_sumo_runs=2*len(groups),
        failed_baseline_conditions=len(failures),plan_sha256=digest(OUT/'plan.json'))),flush=True)


def inputs():
    committed(OUT/'plan.json');plan=read(OUT/'plan.json')
    for name,expected in {**plan['source_sha256'],**plan['inputs_sha256']}.items():
        if digest(ROOT/name)!=expected:raise ValueError('Frozen audit input changed: '+name)
    sources_committed([ROOT/p for p in plan['source_sha256']])
    if runtime_identity()!=plan['runtime']:raise ValueError('Frozen runtime changed')
    previous.inputs()
    return plan


def kernel():
    plan=inputs();old=previous.archived();directory=OUT/'kernel';directory.mkdir(exist_ok=False)
    cases=[];meta=dict(plan_sha256=digest(previous.INTERVENTIONS/'plan.json'),stage='offline')
    for key,record,traces in observations(previous.INTERVENTIONS/'offline/observations.sqlite',meta,plan['kernel_cases']):
        row,trace=old[key[:-1]];state,command,memory=intervention_state(row,trace,key[-1])
        tape=disturbance_tape(row['seed'],row['disturbance'],row['external_innovation_radius'],250)
        for role in ('baseline_suffix','candidate_then_baseline'):
            first=np.asarray(trace[key[-1]]['command']) if role=='candidate_then_baseline' else None
            args=(state,command,memory,key[-1],tape,row['plant'],first)
            original=forecast(*args,record_trace=True);optimized=buffered_forecast(*args,record_trace=True)
            np.testing.assert_equal(original,optimized)
            replay=previous.compare_kernel(optimized,record['branches'][role],traces[role])
            if not replay['all_match']:raise ValueError('Actual SUMO intervention mismatch')
            cases.append(dict(case=list(key)+[role],all_original_fields_exact=True,actual_replay=replay))
    if len(cases)!=698:raise ValueError('All original intervention branches required')
    report=dict(branches=len(cases),all_original_fields_exact=True,all_actual_replays_match=True,cases=cases,
        actual_sumo_runs=0,new_independent_initials=0,plan_sha256=digest(OUT/'plan.json'))
    exclusive_json(directory/'report.json',report);exclusive_json(directory/'sha256.json',dict(report=digest(directory/'report.json')))
    print(json.dumps(dict(branches=len(cases),all_original_fields_exact=True,all_actual_replays_match=True)),flush=True)


def require_kernel():
    report=read(OUT/'kernel/report.json')
    if (digest(OUT/'kernel/report.json')!=read(OUT/'kernel/sha256.json')['report'] or report['branches']!=698
            or not report['all_original_fields_exact'] or not report['all_actual_replays_match']):
        raise ValueError('Complete predictor identity audit required')


def schedule(plan):return [k for g in plan['groups'] for k in g['executions']]
def metadata():return dict(plan_sha256=digest(OUT/'plan.json'),stage='buffered_online_pairing')


def worker():
    execution=OUT/'execution';startup=execution/'worker.native.json'
    atomic_bytes(startup,encode(dict(native_pid=os.getpid(),parent_pid=os.getppid(),utc=utc())))
    deadline=time.monotonic()+180;ack=execution/'worker.acknowledged.json'
    while not ack.exists():
        if time.monotonic()>deadline:raise TimeoutError('No parent acknowledgment; no simulation started')
        time.sleep(.2)
    proof=read(ack)
    if proof['native_pid']!=os.getpid() or proof['parent_pid']!=os.getppid() or proof['native_start_sha256']!=digest(startup):
        raise ValueError('Native acknowledgment mismatch')
    plan=inputs();require_kernel();previous.execution_priority();torch.set_num_threads(1)
    models={s:load_coupled(ROOT/m['path'])[0] for s,m in plan['models'].items()}
    old,baseline,prior=previous.archived(),baselines(),old_preflight()
    for profile in previous.PLANTS:
        for kind in ('nominal','reference'):
            for cls in (ReferenceActuatorGate,BufferedActuatorGate):
                cls(models['1802'],profile,kind).select(np.array([[-80.,12.],[-85.,13.],[-120.,11.]]),np.zeros(3))
    store=EpisodeStore(OUT/'observations.sqlite',metadata());completed=0
    try:
        with ExitStack() as stack:
            envs={p:stack.enter_context(MatchedShiftedMergeEnv(plant=p)) for p in previous.PLANTS}
            for group in plan['groups']:
                pair={}
                for key in group['executions']:
                    source=old[tuple(key[:-2])][0];model=models[key[5].rsplit('_',1)[1]]
                    cls=ReferenceActuatorGate if key[-1]=='original' else BufferedActuatorGate
                    wrapped=[]
                    def factory(_):
                        gate=infrastructure.TimedGate(cls(model,key[4],key[-2]));wrapped.append(gate);return gate
                    started=time.perf_counter()
                    row,trace,initial=run_episode(envs[key[4]],source['seed'],source['scenario'],source['disturbance'],factory,True)
                    costs=np.asarray(row['per_vehicle_finish'])+.05*np.asarray(row['per_vehicle_variation'])
                    row.update(initial=initial.tolist(),plant_name=key[4],controller=key[-2]+'_'+key[5].rsplit('_',1)[1],
                        per_vehicle_cost=costs.tolist(),team_cost=float(costs.mean()))
                    review=previous.review_online(row,trace,key[-2],key[4])
                    record=dict(row=row,review=review,calls=wrapped[0].calls,episode_wall_seconds=time.perf_counter()-started)
                    store.append(key,record,trace)
                    equal_fields(row,baseline[(*key[:-3],'baseline')],PAIR_FIELDS)
                    infrastructure.check_calls(row,trace,record['calls'])
                    if tuple(key[:-1]) in prior:
                        prior_row,prior_trace=prior[tuple(key[:-1])]
                        compare_replay(prior_row['row'],row,prior_trace,trace)
                    pair[key[-1]]=(record,trace);completed+=1
                a,b=pair['original'],pair['buffered']
                compare_replay(a[0]['row'],b[0]['row'],a[1],b[1])
                np.testing.assert_equal([c['error'] for c in a[0]['calls']],[c['error'] for c in b[0]['calls']])
                atomic_bytes(execution/'progress.json',encode(dict(utc=utc(),native_pid=os.getpid(),completed_pairs=group['block_index']+1,
                    expected_pairs=len(plan['groups']),actual_sumo_runs=completed,expected_sumo_runs=plan['actual_sumo_runs'])))
                print(json.dumps(dict(pairs=group['block_index']+1,actual_sumo_runs=completed)),flush=True)
        store.seal(schedule(plan))
    finally:store.close()
    exclusive_json(execution/'collection_complete.json',dict(plan_sha256=digest(OUT/'plan.json'),native_pid=os.getpid(),
        actual_sumo_runs=completed,blocks=len(plan['groups']),database_sha256=digest(OUT/'observations.sqlite'),
        seal_sha256=digest(OUT/'observations.sqlite.seal.json'),utc=utc()))


def watch():
    inputs();require_kernel()
    # Configure only this separate watcher's I/O paths; frozen numerical code
    # and the other native processes are untouched.
    infrastructure.OUT=OUT;infrastructure.SCRIPT=SCRIPT
    infrastructure.watch()


def times(calls):
    values=[c['wall_ms'] for c in calls]
    return dict(calls=len(values),sum_ms=sum(values),p50_p95_p99_max_ms=np.percentile(values,[50,95,99,100]).tolist(),
        over_500ms=sum(t>500 for t in values),terminal_calls=sum(c['error'] is not None for c in calls))


def summarize():
    plan=inputs();execution=OUT/'execution'
    exit,launch,done=[read(execution/n) for n in ('exit.json','launch.json','collection_complete.json')]
    infrastructure.validate_completion(plan,digest(OUT/'plan.json'),exit,launch,done)
    check_acknowledgment(execution,'worker',launch['launcher_pid'],launch['native_pid'])
    if digest(OUT/'observations.sqlite')!=done['database_sha256'] or digest(OUT/'observations.sqlite.seal.json')!=done['seal_sha256']:
        raise ValueError('Complete unchanged data required')
    baseline,prior=baselines(),old_preflight();lookup={};calls={k:[] for k in IMPLEMENTATIONS}
    for key,record,trace in observations(OUT/'observations.sqlite',metadata(),schedule(plan)):
        review=previous.review_online(record['row'],trace,key[-2],key[4])
        if review!=record['review']:raise ValueError('Independent online trace review differs')
        equal_fields(record['row'],baseline[(*key[:-3],'baseline')],PAIR_FIELDS)
        infrastructure.check_calls(record['row'],trace,record['calls'])
        if key[:-1] in prior:compare_replay(prior[key[:-1]][0]['row'],record['row'],prior[key[:-1]][1],trace)
        lookup[key]=(record,trace);calls[key[-1]].extend(record['calls'])
    pairs=[]
    for group in plan['groups']:
        key=tuple(group['case']);a,b=[lookup[(*key,k)] for k in IMPLEMENTATIONS]
        compare_replay(a[0]['row'],b[0]['row'],a[1],b[1])
        np.testing.assert_equal([c['error'] for c in a[0]['calls']],[c['error'] for c in b[0]['calls']])
        pairs.append(dict(case=list(key),source=group['source'],all_non_timing_fields_exact=True,
            commands=len(a[1]),aborted=a[0]['row']['aborted'],completed=a[0]['row']['completed'],
            over_error_bound=a[0]['row']['error_contract_exceeded_substeps']>0,
            adopted=a[0]['review']['actual_admitted_steps']>0,original=times(a[0]['calls']),buffered=times(b[0]['calls'])))
    summaries={k:times(calls[k]) for k in IMPLEMENTATIONS}
    cells=[]
    for profile in previous.PLANTS:
        for kind in ('nominal','reference'):
            for origin in ('preflight','all_old_baseline_failures'):
                selected=[g for g in plan['groups'] if g['case'][4]==profile and g['case'][-1]==kind and g['source']==origin]
                if not selected:continue
                cell=dict(plant=profile,kind=kind,source=origin,paired_cases=len(selected))
                for impl in IMPLEMENTATIONS:
                    cell[impl]=times([c for g in selected for c in lookup[(*g['case'],impl)][0]['calls']])
                cells.append(cell)
    analysis=OUT/'analysis';analysis.mkdir(exist_ok=False)
    report=dict(plan_sha256=digest(OUT/'plan.json'),actual_sumo_runs=plan['actual_sumo_runs'],paired_cases=len(pairs),
        all_non_timing_fields_exact=True,pairs=pairs,cells=cells,summary=summaries,
        total_time_reduction_percent=100*(1-summaries['buffered']['sum_ms']/summaries['original']['sum_ms']),
        new_independent_initials=0,new_risk_certificate=False,interpretation=plan['interpretation'])
    exclusive_json(analysis/'report.json',report);exclusive_json(analysis/'sha256.json',dict(report=digest(analysis/'report.json')))
    print(json.dumps(dict(actual_sumo_runs=report['actual_sumo_runs'],paired_cases=len(pairs),summary=summaries,
        total_time_reduction_percent=report['total_time_reduction_percent'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('freeze','kernel','watch','worker','summarize'))
    globals()[parser.parse_args().action]()
