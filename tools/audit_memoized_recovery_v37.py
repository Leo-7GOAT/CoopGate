"""Complete matched SUMO timing/equivalence for both frozen recovery performance gates."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.episode_store_v21 import EpisodeStore, encode, observations
from coopgate.matched_shift_v21 import MatchedShiftedMergeEnv
from coopgate.memoized_recovery_v37 import CAPACITY, MemoizedRecoveryGate
from coopgate.recovery_gate_audit_v35 import review_joint_trace
from coopgate.recovery_joint_gate_v35 import RecoveryJointGate
import audit_buffered_scope_v25 as infrastructure
import study_recovery_gate_v35 as source
from execute_prospective_v23 import check_acknowledgment, exclusive_json, runtime_identity, sources_committed, utc
from experiment_ego_v13 import run_episode
from profile_phase_runtime_v24 import committed, digest
from recover_checkpoint_v19 import atomic_bytes

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/v37/memoized_recovery_audit'
SCRIPT = 'tools/audit_memoized_recovery_v37.py'
RULES = ('nominal_performance','bank_performance')
IMPLEMENTATIONS = ('original','memoized')
EXTRA = {'proposal_cache','proposal_cache_entries','proposal_cache_capacity','planning_counter_scope'}
SOURCES = ('src/coopgate/memoized_recovery_v37.py','tests/test_memoized_recovery_v37.py',
           SCRIPT,'tests/test_memoized_recovery_audit_v37.py')


def groups(cases):
    selected = sorted(tuple(k) for k in cases if k[-2] in RULES)
    if (len(selected) != 180 or len(set(selected)) != 180
            or len({k[:5] for k in selected}) != 30):
        raise ValueError('All 180 original performance-gate episodes are required')
    combinations = {(r,m) for r in RULES for m in source.MODELS}
    for condition in {k[:5] for k in selected}:
        if {k[-2:] for k in selected if k[:5] == condition} != combinations:
            raise ValueError('Both gates and all three fixed models required in every condition')
    return [dict(block_index=i,case=list(k),executions=[list(k)+[v] for v in
            (IMPLEMENTATIONS if i%2 == 0 else IMPLEMENTATIONS[::-1])]) for i,k in enumerate(selected)]


def archive():
    plan = source.read(source.OUT/'plan.json')
    result = {}
    for group in plan['groups']:
        path = source.OUT/'blocks'/f"{group['block_index']:05d}"/'observations.sqlite'
        for key,record,trace in observations(path,source.metadata(group),group['cases']):
            if key[-2] in RULES:
                result[key] = record,trace
    return result


def non_timing(record,trace):
    return dict(row={k:v for k,v in record['row'].items() if k not in ('setup_ms','controller_ms')},
        calls=[c['error'] for c in record['calls']],
        trace=[{k:v for k,v in t.items() if k != 'gate_ms' and k not in EXTRA} for t in trace])


def cache_work(trace):
    total = dict(requests=0,hits=0,misses=0,successful_misses=0,failed_misses=0,
        logical_successful_transitions=0,computed_successful_transitions=0,capacity_skips=0)
    maximum_entries = 0
    for item in trace:
        values = item['proposal_cache']
        if (set(values) != set(total) or any(type(v) is not int or v < 0 for v in values.values())
                or values['requests'] != values['hits']+values['misses']
                or values['misses'] != values['successful_misses']+values['failed_misses']
                or values['logical_successful_transitions'] != item['completed_planning_model_transitions']
                or values['computed_successful_transitions'] > values['logical_successful_transitions']
                or item['proposal_cache_capacity'] != CAPACITY
                or not 0 <= item['proposal_cache_entries'] <= CAPACITY
                or not item['planning_counter_scope'].startswith('Original completed_planning_model_transitions retains logical reference work;')):
            raise ValueError('Cache diagnostics must distinguish logical and actually computed work')
        for name,value in values.items(): total[name] += value
        maximum_entries = max(maximum_entries,item['proposal_cache_entries'])
    return dict(**total,maximum_entries=maximum_entries,
        complete_planning_work=all(t['planning_work_complete'] for t in trace))


def freeze():
    prior = source.inputs()
    report = source.read(source.OUT/'analysis/report.json')
    proof = source.read(source.OUT/'completion_verification.json')
    if (report['actual_sumo_runs'] != 390 or not report['all_physical_and_full_gate_replays_match']
            or proof['summary_exit_code'] != 0 or proof['worker_exit_code'] != 0
            or proof['report_sha256'] != digest(source.OUT/'analysis/report.json')):
        raise ValueError('Complete independently reviewed source integration required')
    schedule = groups([k for g in prior['groups'] for k in g['cases']])
    hashes = dict(prior['source_sha256'])
    hashes.update({n:digest(ROOT/n) for n in SOURCES})
    for name,expected in hashes.items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen dependency changed: '+name)
    sources_committed([ROOT/n for n in hashes])
    files = [source.OUT/n for n in ('plan.json','analysis/report.json','completion_verification.json')]
    for group in prior['groups']:
        directory = source.OUT/'blocks'/f"{group['block_index']:05d}"
        files += [directory/n for n in ('observations.sqlite','observations.sqlite.seal.json','complete.json')]
    inputs = dict(prior['inputs_sha256'])
    inputs.update({p.relative_to(ROOT).as_posix():digest(p) for p in files})
    OUT.mkdir(parents=True,exist_ok=False)
    exclusive_json(OUT/'plan.json',dict(utc=utc(),runtime=runtime_identity(),source_sha256=hashes,
        inputs_sha256=inputs,models=prior['models'],groups=schedule,blocks=180,actual_sumo_runs=360,
        implementations=IMPLEMENTATIONS,cache_capacity=CAPACITY,new_independent_initials=0,new_risk_certificate=False,
        change='Only exact memoization of successful pure nominal planning proposals within one decision. Key contains all float64 state/previous-command/configuration bits plus global clock. All actuator-memory checks still execute. Exceptions are never stored; a full cache computes normally. No candidate, path, ranking, threshold, horizon or physics change.',
        scope='All 180 nominal/bank performance episodes from the complete 390-run integration, all 30 conditions and all three fixed models. Includes every old baseline-failure condition. Old 15 initials only; no effect selection or independent efficacy evidence.',
        order='One worker; fixed lexicographic case order, alternating original/memoized order by pair index. All rules, models and actuator classes warmed on the same declared state before timing.',
        checks='Store actual outcomes before checking. Every non-timing original field, command, forecast-evidence field and terminal identity must match both its fresh pair and the fully replayed V35 archive. Independently reconstruct physical and baseline evidence for every actual trace.',
        work='Original planning counters remain logical reference work. Additional cache counters report hits/misses, capacity skips and successfully computed transitions. Work inside failed planner calls remains uncounted; full callback times retain it.',
        timing='External full select callbacks including terminal exceptions; warmup excluded. Sum, mean, p95/max and >500ms counts, with logical calls and physical outcome equality checked. Concurrent desktop measurements, not hard-real-time or cross-study speed claims.',
        publication='All outcomes, capacity pressure, failures and slow calls retained. No reduced adoption or weakened checks to improve runtime. Existing performance-risk calibration and both running full grids remain unchanged.',
        active_calibration_changed=False,active_full_grids_changed=False,partial_formal_effectiveness_read=False))
    print(json.dumps(dict(pairs=180,actual_sumo_runs=360,plan_sha256=digest(OUT/'plan.json'))),flush=True)


def inputs():
    committed(OUT/'plan.json')
    plan = source.read(OUT/'plan.json')
    for name,expected in {**plan['source_sha256'],**plan['inputs_sha256']}.items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen memoization source/input changed: '+name)
    sources_committed([ROOT/n for n in plan['source_sha256']])
    if runtime_identity() != plan['runtime']:
        raise ValueError('Frozen runtime changed')
    prior = source.read(source.OUT/'plan.json')
    if groups([k for g in prior['groups'] for k in g['cases']]) != plan['groups']:
        raise ValueError('All completed original performance cases must remain fixed')
    for item in plan['models'].values():
        if digest(ROOT/item['path']) != item['sha256']:
            raise ValueError('Frozen model changed')
    return plan


def metadata(group):
    return dict(plan_sha256=digest(OUT/'plan.json'),block_index=group['block_index'],stage='exact_memoization')


def verify(key,record,trace,old,model):
    infrastructure.check_calls(record['row'],trace,record['calls'])
    if non_timing(record,trace) != non_timing(*old):
        raise ValueError('Actual execution differs from fully reviewed V35 archive')
    review = review_joint_trace(record['row'],trace,model)
    work = cache_work(trace) if key[-1] == 'memoized' else None
    return dict(case=list(key),all_original_non_timing_fields_exact=True,physical_review=review,cache_work=work)


def worker():
    execution = OUT/'execution'
    startup,ack = execution/'worker.native.json',execution/'worker.acknowledged.json'
    atomic_bytes(startup,encode(dict(native_pid=os.getpid(),parent_pid=os.getppid(),utc=utc())))
    deadline = time.monotonic()+180.
    while not ack.exists():
        if time.monotonic() > deadline:
            raise TimeoutError('No native parent acknowledgment; no simulation started')
        time.sleep(.2)
    proof = source.read(ack)
    if (proof['native_pid'] != os.getpid() or proof['parent_pid'] != os.getppid()
            or proof['native_start_sha256'] != digest(startup)):
        raise ValueError('Native acknowledgment mismatch')
    plan = inputs()
    source.recovery.execution_priority()
    torch.set_num_threads(1)
    models = {s:load_coupled(ROOT/m['path'])[0] for s,m in plan['models'].items()}
    old = archive()
    classes = dict(original=RecoveryJointGate,memoized=MemoizedRecoveryGate)
    warm = np.array([[-140.,12.],[-220.,10.],[-190.,12.]])
    for profile in source.recovery.PROFILES:
        for model in models.values():
            for rule in RULES:
                for cls in classes.values(): cls(model,profile,rule).select(warm,np.zeros(3))
    (OUT/'blocks').mkdir(exist_ok=False)
    completed = 0
    with ExitStack() as stack:
        envs = {p:stack.enter_context(MatchedShiftedMergeEnv(plant=p)) for p in source.recovery.PROFILES}
        for group in plan['groups']:
            directory = OUT/'blocks'/f"{group['block_index']:05d}"
            directory.mkdir()
            store = EpisodeStore(directory/'observations.sqlite',metadata(group))
            reviews,paired = [],[]
            try:
                for key in group['executions']:
                    prior,_ = old[tuple(key[:-1])]
                    reference = prior['row']
                    model = models[key[-2]]
                    wrapped = []
                    def factory(_):
                        gate = infrastructure.TimedGate(classes[key[-1]](model,key[4],key[-3]))
                        wrapped.append(gate)
                        return gate
                    start = time.perf_counter()
                    row,trace,initial = run_episode(envs[key[4]],reference['seed'],reference['scenario'],reference['disturbance'],factory,True)
                    costs = np.asarray(row['per_vehicle_finish'])+.05*np.asarray(row['per_vehicle_variation'])
                    row.update(initial=initial.tolist(),plant_name=key[4],rule=key[-3],model=key[-2],
                        controller=key[-3]+'_'+key[-2],certificate_contract='bounded_actuator_memory_stopping',
                        per_vehicle_cost=costs.tolist(),team_cost=float(costs.mean()),control_steps=len(trace),
                        learned_action_steps=sum(bool(t['actual_learning_used']) for t in trace))
                    record = dict(row=row,calls=wrapped[0].calls,episode_wall_seconds=time.perf_counter()-start)
                    store.append(key,record,trace)
                    reviews.append(verify(key,record,trace,old[tuple(key[:-1])],model))
                    paired.append(non_timing(record,trace))
                    completed += 1
                if paired[0] != paired[1]:
                    raise ValueError('Fresh original/memoized executions differ')
                store.seal(group['executions'])
            finally:
                store.close()
            exclusive_json(directory/'complete.json',dict(**metadata(group),reviews=reviews,
                database_sha256=digest(directory/'observations.sqlite'),seal_sha256=digest(directory/'observations.sqlite.seal.json')))
            atomic_bytes(execution/'progress.json',encode(dict(utc=utc(),native_pid=os.getpid(),completed_runs=completed,
                expected_runs=360,completed_blocks=group['block_index']+1,expected_blocks=180)))
    exclusive_json(execution/'collection_complete.json',dict(plan_sha256=digest(OUT/'plan.json'),native_pid=os.getpid(),
        actual_sumo_runs=completed,blocks=180,utc=utc()))
    print(json.dumps(dict(actual_sumo_runs=completed)),flush=True)


def watch():
    inputs()
    infrastructure.OUT,infrastructure.SCRIPT,infrastructure.inputs = OUT,SCRIPT,inputs
    infrastructure.watch()


def summarize():
    plan = inputs()
    execution = OUT/'execution'
    receipt,launch,done = [source.read(execution/n) for n in ('exit.json','launch.json','collection_complete.json')]
    infrastructure.validate_completion(plan,digest(OUT/'plan.json'),receipt,launch,done)
    check_acknowledgment(execution,'worker',launch['launcher_pid'],launch['native_pid'])
    torch.set_num_threads(1)
    models = {s:load_coupled(ROOT/m['path'])[0] for s,m in plan['models'].items()}
    old,cases = archive(),[]
    for group in plan['groups']:
        directory = OUT/'blocks'/f"{group['block_index']:05d}"
        proof = source.read(directory/'complete.json')
        if (proof['database_sha256'] != digest(directory/'observations.sqlite')
                or proof['seal_sha256'] != digest(directory/'observations.sqlite.seal.json')):
            raise ValueError('Complete paired block changed')
        paired = []
        for index,(key,record,trace) in enumerate(observations(directory/'observations.sqlite',metadata(group),group['executions'])):
            review = verify(key,record,trace,old[key[:-1]],models[key[-2]])
            if review != proof['reviews'][index]:
                raise ValueError('Independent physical/equivalence review differs')
            cases.append(dict(case=list(key),row=record['row'],calls=record['calls'],review=review))
            paired.append(non_timing(record,trace))
        if paired[0] != paired[1]: raise ValueError('Fresh matched pair differs')
    if len(cases) != 360: raise ValueError('All 360 actual executions required')
    cells = []
    for rule in (None,*RULES):
        for plant in (None,*source.recovery.PROFILES):
            for implementation in IMPLEMENTATIONS:
                selected = [c for c in cases if c['case'][-1] == implementation
                    and (rule is None or c['row']['rule'] == rule) and (plant is None or c['row']['plant_name'] == plant)]
                times = [t['wall_ms'] for c in selected for t in c['calls']]
                work = [c['review']['cache_work'] for c in selected] if implementation == 'memoized' else []
                cells.append(dict(rule=rule,plant=plant,implementation=implementation,runs=len(selected),
                    calls=len(times),terminal_calls=sum(t['error'] is not None for c in selected for t in c['calls']),
                    total_ms=sum(times),mean_ms=float(np.mean(times)),p95_ms=float(np.quantile(times,.95)),
                    max_ms=max(times),over_500ms=sum(t > 500. for t in times),
                    cache_requests=sum(w['requests'] for w in work),cache_hits=sum(w['hits'] for w in work),
                    cache_capacity_skips=sum(w['capacity_skips'] for w in work),
                    computed_successful_transitions=sum(w['computed_successful_transitions'] for w in work),
                    logical_successful_transitions=sum(w['logical_successful_transitions'] for w in work)))
    target = OUT/'analysis'
    target.mkdir(exist_ok=False)
    report = dict(plan_sha256=digest(OUT/'plan.json'),actual_sumo_runs=360,pairs=180,
        all_original_non_timing_fields_exact=True,all_independent_physical_reviews_pass=True,
        cases=cases,cells=cells,new_independent_initials=0,new_risk_certificate=False,
        scope=plan['scope'],timing_scope=plan['timing'],work_scope=plan['work'])
    exclusive_json(target/'report.json',report)
    exclusive_json(target/'sha256.json',dict(report=digest(target/'report.json')))
    print(json.dumps(dict(actual_sumo_runs=360,pairs=180,all_exact=True,
        totals=[c for c in cells if c['rule'] is None and c['plant'] is None])),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('freeze','worker','watch','summarize'))
    globals()[parser.parse_args().command]()
