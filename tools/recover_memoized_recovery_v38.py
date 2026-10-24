"""Explicit continuation of the interrupted V37 paired memoization audit.

The 74 sealed V37 blocks remain the authoritative prefix.  The empty,
unsealed block 74 is preserved byte-for-byte as interruption evidence and is
never opened for writing.  A separately frozen execution reruns original
block indices 74..179.  Only after that execution exits successfully does the
summarizer combine the two sealed ranges and call V37's unchanged verifier on
all 180 pairs.
"""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.episode_store_v21 import EpisodeStore, encode, inventory, observations
from coopgate.matched_shift_v21 import MatchedShiftedMergeEnv
import audit_buffered_scope_v25 as infrastructure
import audit_memoized_recovery_v37 as source
import study_recovery_gate_v35 as study
from execute_prospective_v23 import (check_acknowledgment, exclusive_json,
    native_snapshot, runtime_identity, sources_committed, utc)
from experiment_ego_v13 import run_episode
from profile_phase_runtime_v24 import committed, digest
from recover_checkpoint_v19 import atomic_bytes


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT/'artifacts/v37/memoized_recovery_audit'
OUT = ROOT/'artifacts/v38/memoized_recovery_recovery_20261024'
SCRIPT = 'tools/recover_memoized_recovery_v38.py'
TEST = 'tests/test_recover_memoized_recovery_v38.py'
SOURCES = (SCRIPT, TEST)
RETAINED_BLOCKS = 74
PARTIAL_INDEX = 74
TOTAL_BLOCKS = 180
RECOVERY_BLOCKS = TOTAL_BLOCKS-RETAINED_BLOCKS
RECOVERY_RUNS = 2*RECOVERY_BLOCKS
EVIDENCE_FILES = ('watcher_launch.json','execution/launch.json',
    'execution/spawn.json',
    'execution/worker.native.json','execution/worker.acknowledged.json',
    'execution/progress.json','watcher.stdout.log','watcher.stderr.log',
    'execution/worker.stdout.log','execution/worker.stderr.log')
FAILED_ATTEMPT = None
RECOVERY_SEGMENT = 'recovered_v38'
FAILED_EVIDENCE_FILES = (
    'watcher_launch.json','execution/spawn.json','execution/launch.json',
    'execution/worker.native.json','execution/worker.acknowledged.json',
    'execution/exit.json','execution/worker.stdout.log','execution/worker.stderr.log',
    'recovery_watcher.stdout.log','recovery_watcher.stderr.log')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_hashes(directory):
    directory = Path(directory)
    return {p.relative_to(directory).as_posix():digest(p)
            for p in sorted(directory.rglob('*')) if p.is_file()}


def process_chain(launch, watcher):
    values = (watcher['launcher_pid'], launch['watcher_pid'],
              launch['launcher_pid'], launch['native_pid'])
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError('Interrupted process identities are incomplete')
    return tuple(dict.fromkeys(values))


def process_descendants(processes, roots):
    """Return every currently visible descendant of the frozen process roots."""
    descendants,frontier = set(),set(roots)
    while frontier:
        children = {int(process['ProcessId']) for process in processes
                    if process.get('ProcessId') is not None
                    and process.get('ParentProcessId') is not None
                    and int(process['ParentProcessId']) in frontier}
        children -= descendants
        children -= set(roots)
        if not children:
            break
        descendants.update(children)
        frontier = children
    return descendants


def require_interrupted_processes_absent(prior=PRIOR, processes=None):
    """Require an unknown exit with the exact old native chain absent now."""
    prior = Path(prior)
    execution = prior/'execution'
    if (execution/'exit.json').exists() or (execution/'collection_complete.json').exists():
        raise ValueError('A terminal V37 receipt exists; recovery is not justified')
    launch, watcher = read(execution/'launch.json'), read(prior/'watcher_launch.json')
    ids = process_chain(launch, watcher)
    processes = native_snapshot() if processes is None else processes
    descendants = process_descendants(processes,ids)
    for process in processes:
        command = process.get('CommandLine') or ''
        if (process.get('ProcessId') in ids or process.get('ProcessId') in descendants
                or 'audit_memoized_recovery_v37.py' in command):
            raise ValueError('The V37 execution may still be live; do not recover it')
    return dict(observed_utc=utc(),old_process_ids=list(ids),
        descendant_processes_checked=True,target_execution_processes_absent=True,
        original_exit_status='unknown',
        exit_receipt_present=False,collection_complete_present=False)


def failed_attempt_identity(directory=None,processes=None):
    """Re-verify an earlier failed retry without consuming any result outcomes."""
    directory = FAILED_ATTEMPT if directory is None else Path(directory)
    if directory is None:
        return None
    directory = Path(directory)
    verification = directory/'failed_attempt_verification.json'
    committed(verification)
    proof = read(verification)
    if (proof.get('schema') != 'coopgate.memoized_recovery_failed_attempt.v1'
            or proof.get('status') != 'failed_before_simulation'
            or proof.get('results_eligible') is not False
            or proof.get('restart_in_place_permitted') is not False
            or proof.get('simulation_boundary',{}).get('new_sumo_runs') != 0
            or proof.get('exit',{}).get('exit_code') != 1
            or digest(directory/'plan.json') != proof.get('plan_sha256')):
        raise ValueError('The previous failed-attempt boundary is invalid')
    evidence_hashes = proof.get('execution_evidence_sha256',{})
    if set(evidence_hashes) != set(FAILED_EVIDENCE_FILES):
        raise ValueError('The previous failed-attempt evidence set is incomplete')
    for name,expected in evidence_hashes.items():
        path = directory/name
        if not path.is_file() or digest(path) != expected:
            raise ValueError('Previous failed-attempt evidence changed: '+name)
    watcher = read(directory/'watcher_launch.json')
    spawn = read(directory/'execution/spawn.json')
    launch = read(directory/'execution/launch.json')
    native = read(directory/'execution/worker.native.json')
    acknowledgment = read(directory/'execution/worker.acknowledged.json')
    exit_receipt = read(directory/'execution/exit.json')
    identities = proof['process_identity']
    plan_sha = proof['plan_sha256']
    if (watcher.get('launcher_pid') != identities.get('outer_launcher_pid')
            or watcher.get('plan_sha256') != plan_sha
            or spawn.get('watcher_pid') != identities.get('watcher_pid')
            or spawn.get('launcher_pid') != identities.get('worker_launcher_pid')
            or spawn.get('plan_sha256') != plan_sha
            or launch.get('watcher_pid') != identities.get('watcher_pid')
            or launch.get('launcher_pid') != identities.get('worker_launcher_pid')
            or launch.get('native_pid') != identities.get('native_worker_pid')
            or launch.get('parent_pid') != identities.get('worker_launcher_pid')
            or launch.get('plan_sha256') != plan_sha
            or native.get('native_pid') != identities.get('native_worker_pid')
            or native.get('parent_pid') != identities.get('worker_launcher_pid')
            or acknowledgment.get('native_pid') != identities.get('native_worker_pid')
            or acknowledgment.get('parent_pid') != identities.get('worker_launcher_pid')
            or acknowledgment.get('launcher_pid') != identities.get('worker_launcher_pid')
            or acknowledgment.get('native_start_sha256') != digest(
                directory/'execution/worker.native.json')
            or any(exit_receipt.get(name) != launch.get(name) for name in
                   ('launcher_pid','native_pid','parent_pid','native_creation_date','command_line'))):
        raise ValueError('The previous failed-attempt process identity chain changed')
    if (exit_receipt.get('exit_code') != 1
            or exit_receipt.get('plan_sha256') != plan_sha
            or (directory/'blocks').exists()
            or (directory/'execution/progress.json').exists()
            or (directory/'execution/collection_complete.json').exists()
            or (directory/'analysis').exists()):
        raise ValueError('The previous retry is not a pre-simulation failure')
    roots = tuple(identities[name] for name in ('outer_launcher_pid','watcher_pid',
        'worker_launcher_pid','native_worker_pid'))
    processes = native_snapshot() if processes is None else processes
    descendants = process_descendants(processes,roots)
    if any(process.get('ProcessId') in roots or process.get('ProcessId') in descendants
           for process in processes):
        raise ValueError('The previous failed retry may still have a live descendant')
    return dict(path=directory.relative_to(ROOT).as_posix(),
        verification_sha256=digest(verification),plan_sha256=proof['plan_sha256'],
        execution_evidence_sha256=evidence_hashes,
        exit_code=1,new_sumo_runs=0,
        all_recorded_processes_and_visible_descendants_absent=True)


def block_identity(directory, expected_metadata, expected_cases, *, complete):
    """Inspect identities and checksums only; never read partial outcome blobs."""
    directory = Path(directory)
    database = directory/'observations.sqlite'
    state = inventory(database)
    if state['metadata'] != expected_metadata:
        raise ValueError('Interrupted block metadata changed')
    expected = [list(case) for case in expected_cases]
    markers = [(directory/name).exists() for name in
               ('observations.sqlite.seal.json','complete.json')]
    if complete:
        if not all(markers) or state['cases'] != expected:
            raise ValueError('Every retained block must be completely sealed')
        seal = read(directory/'observations.sqlite.seal.json')
        proof = read(directory/'complete.json')
        proof_metadata = {name:proof.get(name) for name in expected_metadata}
        if (seal['database_sha256'] != digest(database)
                or {k:seal[k] for k in ('metadata','cases','row_sha256','trace_sha256')} != state
                or proof_metadata != expected_metadata
                or proof['database_sha256'] != digest(database)
                or proof['seal_sha256'] != digest(directory/'observations.sqlite.seal.json')
                or len(proof.get('reviews',())) != len(expected)
                or [item.get('case') for item in proof['reviews']] != expected):
            raise ValueError('Retained block proof or exact schedule changed')
    else:
        if any(markers) or state['cases'] != expected[:len(state['cases'])]:
            raise ValueError('Partial block must be an unsealed schedule prefix')
    return dict(records=len(state['cases']),complete=bool(complete),
        inventory=state,files=file_hashes(directory))


def partition_groups(plan):
    groups = plan.get('groups')
    if (not isinstance(groups,list) or len(groups) != TOTAL_BLOCKS
            or [group.get('block_index') for group in groups] != list(range(TOTAL_BLOCKS))
            or any(len(group.get('executions',())) != 2 for group in groups)):
        raise ValueError('The exact contiguous 180-pair V37 schedule is required')
    return groups[:RETAINED_BLOCKS],groups[RETAINED_BLOCKS:]


def interrupted_inventory(plan, prior=PRIOR):
    prior = Path(prior)
    retained, missing = partition_groups(plan)
    root = prior/'blocks'
    names = sorted(path.name for path in root.iterdir() if path.is_dir())
    expected_names = [f'{index:05d}' for index in range(PARTIAL_INDEX+1)]
    if names != expected_names:
        raise ValueError('Interrupted V37 blocks are not the exact contiguous prefix 0..74')
    blocks = []
    for group in retained:
        directory = root/f"{group['block_index']:05d}"
        blocks.append(dict(block_index=group['block_index'],
            **block_identity(directory,source.metadata(group),group['executions'],complete=True)))
    partial_group = missing[0]
    partial = dict(block_index=PARTIAL_INDEX,
        **block_identity(root/f'{PARTIAL_INDEX:05d}',source.metadata(partial_group),
                         partial_group['executions'],complete=False))
    progress = read(prior/'execution/progress.json')
    if (progress.get('completed_blocks') != RETAINED_BLOCKS
            or progress.get('completed_runs') != 2*RETAINED_BLOCKS
            or progress.get('expected_blocks') != TOTAL_BLOCKS
            or progress.get('expected_runs') != 2*TOTAL_BLOCKS):
        raise ValueError('Last durable progress does not match the sealed prefix')
    return dict(retained=blocks,partial=partial,progress=progress,
        retained_blocks=RETAINED_BLOCKS,retained_runs=2*RETAINED_BLOCKS,
        missing_blocks=RECOVERY_BLOCKS,missing_runs=RECOVERY_RUNS,
        partial_records_engineering_only=partial['records'])


def freeze():
    if OUT.exists():
        raise FileExistsError('Preserve the original V38 recovery attempt')
    prior_plan = source.inputs()
    if digest(PRIOR/'plan.json') != source.digest(source.OUT/'plan.json'):
        raise ValueError('Configured V37 source directory changed')
    absence = require_interrupted_processes_absent()
    previous_failed = failed_attempt_identity()
    interrupted = interrupted_inventory(prior_plan)
    original_evidence_hashes = {}
    for name in EVIDENCE_FILES:
        path = PRIOR/name
        if not path.exists():
            raise ValueError('Missing interruption evidence: '+name)
        original_evidence_hashes[name] = digest(path)
    hashes = dict(prior_plan['source_sha256'])
    hashes.update({name:digest(ROOT/name) for name in SOURCES})
    for name,expected in hashes.items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen recovery dependency changed: '+name)
    sources_committed([ROOT/name for name in hashes])
    _, missing = partition_groups(prior_plan)
    OUT.mkdir(parents=True,exist_ok=False)
    evidence = OUT/'interruption_evidence'
    shutil.copytree(PRIOR/'blocks'/f'{PARTIAL_INDEX:05d}',evidence/f'block_{PARTIAL_INDEX:05d}')
    for name in EVIDENCE_FILES:
        source_path = PRIOR/name
        target = evidence/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source_path,target)
    evidence_hashes = file_hashes(evidence)
    exclusive_json(OUT/'plan.json',dict(utc=utc(),runtime=runtime_identity(),
        original_plan_sha256=digest(PRIOR/'plan.json'),source_sha256=hashes,
        prior_source_sha256=prior_plan['source_sha256'],models=prior_plan['models'],
        retained_blocks=RETAINED_BLOCKS,retained_actual_sumo_runs=2*RETAINED_BLOCKS,
        missing_groups=missing,blocks=RECOVERY_BLOCKS,actual_sumo_runs=RECOVERY_RUNS,
        aggregate_blocks=TOTAL_BLOCKS,aggregate_actual_sumo_runs=2*TOTAL_BLOCKS,
        interruption=interrupted,interruption_presence=absence,
        previous_failed_attempt=previous_failed,
        original_execution_evidence_sha256=original_evidence_hashes,
        interruption_evidence_sha256=evidence_hashes,
        partial_block_reused_for_results=False,partial_records_engineering_only=True,
        partial_record_count=interrupted['partial_records_engineering_only'],
        original_exit_status='unknown',new_independent_initials=0,new_risk_certificate=False,
        recovery_rule='Keep V37 sealed blocks 0..73 unchanged; preserve unsealed block 74 byte-for-byte; rerun only complete original pairs 74..179 in a separate directory.',
        verification='Every recovered record is saved before V37 unchanged verify/non_timing checks. Final reporting rechecks every old and recovered sealed pair with the same V37 verifier and requires exact 180-pair coverage.',
        effects='No candidate, policy, path, threshold, plant, ordering or outcome selection changes. Partial records are engineering-only and excluded from effect and timing aggregation.',
        timing='Paired orders remain those frozen in V37. Aggregate timing spans two desktop epochs after an interruption and is not a hard-real-time or single-session benchmark.',
        active_calibration_changed=False,active_full_grids_changed=False,
        partial_effectiveness_outcomes_read=False))
    print(json.dumps(dict(retained_blocks=RETAINED_BLOCKS,missing_blocks=RECOVERY_BLOCKS,
        actual_sumo_runs=RECOVERY_RUNS,plan_sha256=digest(OUT/'plan.json'))),flush=True)


def recovery_metadata(group):
    return dict(plan_sha256=digest(OUT/'plan.json'),
        original_plan_sha256=digest(PRIOR/'plan.json'),block_index=group['block_index'],
        stage='exact_memoization_explicit_recovery')


def inputs():
    committed(OUT/'plan.json')
    prior_plan = source.inputs()
    plan = read(OUT/'plan.json')
    if (plan['original_plan_sha256'] != digest(PRIOR/'plan.json')
            or plan['runtime'] != runtime_identity()
            or plan['prior_source_sha256'] != prior_plan['source_sha256']):
        raise ValueError('V37 scientific or runtime identity changed')
    for name,expected in plan['source_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen V38 source changed: '+name)
    sources_committed([ROOT/name for name in plan['source_sha256']])
    require_interrupted_processes_absent()
    if failed_attempt_identity() != plan.get('previous_failed_attempt'):
        raise ValueError('Previous failed-attempt identity changed')
    current = interrupted_inventory(prior_plan)
    if current != plan['interruption']:
        raise ValueError('Original V37 interrupted prefix changed')
    for name,expected in plan['original_execution_evidence_sha256'].items():
        if digest(PRIOR/name) != expected:
            raise ValueError('Original V37 execution evidence changed: '+name)
    if file_hashes(OUT/'interruption_evidence') != plan['interruption_evidence_sha256']:
        raise ValueError('Copied interruption evidence changed')
    _, missing = partition_groups(prior_plan)
    if (plan['missing_groups'] != missing or plan['blocks'] != RECOVERY_BLOCKS
            or plan['actual_sumo_runs'] != RECOVERY_RUNS):
        raise ValueError('Recovery scope changed')
    return plan


def worker():
    execution = OUT/'execution'
    startup,ack = execution/'worker.native.json',execution/'worker.acknowledged.json'
    atomic_bytes(startup,encode(dict(native_pid=os.getpid(),parent_pid=os.getppid(),utc=utc())))
    deadline = time.monotonic()+180.
    while not ack.exists():
        if time.monotonic() > deadline:
            raise TimeoutError('No native parent acknowledgment; no simulation started')
        time.sleep(.2)
    proof = read(ack)
    if (proof['native_pid'] != os.getpid() or proof['parent_pid'] != os.getppid()
            or proof['native_start_sha256'] != digest(startup)):
        raise ValueError('Native acknowledgment mismatch')
    plan = inputs()
    study.recovery.execution_priority()
    torch.set_num_threads(1)
    models = {name:load_coupled(ROOT/item['path'])[0] for name,item in plan['models'].items()}
    archived = source.archive()
    classes = dict(original=source.RecoveryJointGate,memoized=source.MemoizedRecoveryGate)
    warm = np.array([[-140.,12.],[-220.,10.],[-190.,12.]])
    for profile in study.recovery.PROFILES:
        for model in models.values():
            for rule in source.RULES:
                for cls in classes.values():
                    cls(model,profile,rule).select(warm,np.zeros(3))
    (OUT/'blocks').mkdir(exist_ok=False)
    completed = 0
    with ExitStack() as stack:
        envs = {profile:stack.enter_context(MatchedShiftedMergeEnv(plant=profile))
                for profile in study.recovery.PROFILES}
        for local_index,group in enumerate(plan['missing_groups']):
            directory = OUT/'blocks'/f"{group['block_index']:05d}"
            directory.mkdir()
            store = EpisodeStore(directory/'observations.sqlite',recovery_metadata(group))
            reviews,paired = [],[]
            try:
                for key in group['executions']:
                    prior,_ = archived[tuple(key[:-1])]
                    reference = prior['row']
                    model = models[key[-2]]
                    wrapped = []
                    def factory(_):
                        gate = infrastructure.TimedGate(classes[key[-1]](model,key[4],key[-3]))
                        wrapped.append(gate)
                        return gate
                    started = time.perf_counter()
                    row,trace,initial = run_episode(envs[key[4]],reference['seed'],
                        reference['scenario'],reference['disturbance'],factory,True)
                    costs = np.asarray(row['per_vehicle_finish'])+.05*np.asarray(row['per_vehicle_variation'])
                    row.update(initial=initial.tolist(),plant_name=key[4],rule=key[-3],model=key[-2],
                        controller=key[-3]+'_'+key[-2],certificate_contract='bounded_actuator_memory_stopping',
                        per_vehicle_cost=costs.tolist(),team_cost=float(costs.mean()),control_steps=len(trace),
                        learned_action_steps=sum(bool(item['actual_learning_used']) for item in trace))
                    record = dict(row=row,calls=wrapped[0].calls,
                        episode_wall_seconds=time.perf_counter()-started)
                    store.append(key,record,trace)
                    reviews.append(source.verify(key,record,trace,archived[tuple(key[:-1])],model))
                    paired.append(source.non_timing(record,trace))
                    completed += 1
                if paired[0] != paired[1]:
                    raise ValueError('Fresh original/memoized recovery executions differ')
                store.seal(group['executions'])
            finally:
                store.close()
            exclusive_json(directory/'complete.json',dict(**recovery_metadata(group),reviews=reviews,
                database_sha256=digest(directory/'observations.sqlite'),
                seal_sha256=digest(directory/'observations.sqlite.seal.json')))
            atomic_bytes(execution/'progress.json',encode(dict(utc=utc(),native_pid=os.getpid(),
                completed_runs=completed,expected_runs=RECOVERY_RUNS,completed_blocks=local_index+1,
                expected_blocks=RECOVERY_BLOCKS,aggregate_completed_blocks=RETAINED_BLOCKS+local_index+1,
                aggregate_expected_blocks=TOTAL_BLOCKS)))
    if completed != RECOVERY_RUNS:
        raise ValueError('Every missing V37 pair must be recovered')
    exclusive_json(execution/'collection_complete.json',dict(plan_sha256=digest(OUT/'plan.json'),
        native_pid=os.getpid(),actual_sumo_runs=completed,blocks=RECOVERY_BLOCKS,
        aggregate_actual_sumo_runs=2*TOTAL_BLOCKS,aggregate_blocks=TOTAL_BLOCKS,utc=utc()))
    print(json.dumps(dict(recovered_actual_sumo_runs=completed,aggregate_actual_sumo_runs=360)),flush=True)


def watch():
    inputs()
    infrastructure.OUT,infrastructure.SCRIPT,infrastructure.inputs = OUT,SCRIPT,inputs
    infrastructure.watch()


def require_full_coverage(cases, groups):
    expected = [tuple(key) for group in groups for key in group['executions']]
    actual = [tuple(case['case']) for case in cases]
    if len(actual) != 2*TOTAL_BLOCKS or len(set(actual)) != len(actual) or set(actual) != set(expected):
        raise ValueError('Final report requires each of the 180 frozen pairs exactly once')
    return True


def summarize():
    plan = inputs()
    execution = OUT/'execution'
    receipt,launch,done = [read(execution/name) for name in
                           ('exit.json','launch.json','collection_complete.json')]
    infrastructure.validate_completion(plan,digest(OUT/'plan.json'),receipt,launch,done)
    check_acknowledgment(execution,'worker',launch['launcher_pid'],launch['native_pid'])
    expected_recovery_names = [f"{group['block_index']:05d}" for group in plan['missing_groups']]
    actual_recovery_names = sorted(path.name for path in (OUT/'blocks').iterdir()
                                   if path.is_dir())
    if actual_recovery_names != expected_recovery_names:
        raise ValueError('Recovery output must contain exactly the frozen missing block range')
    receipt_paths = ('watcher_launch.json','execution/spawn.json','execution/launch.json',
        'execution/worker.native.json','execution/worker.acknowledged.json',
        'execution/exit.json','execution/collection_complete.json')
    execution_receipts = {name:digest(OUT/name) for name in receipt_paths}
    torch.set_num_threads(1)
    models = {name:load_coupled(ROOT/item['path'])[0] for name,item in plan['models'].items()}
    archived,cases,block_hashes = source.archive(),[],{}
    original_plan = read(PRIOR/'plan.json')
    for group in original_plan['groups']:
        retained = group['block_index'] < RETAINED_BLOCKS
        root = PRIOR if retained else OUT
        directory = root/'blocks'/f"{group['block_index']:05d}"
        expected_metadata = source.metadata(group) if retained else recovery_metadata(group)
        proof = read(directory/'complete.json')
        proof_metadata = {name:proof.get(name) for name in expected_metadata}
        if (proof_metadata != expected_metadata
                or proof['database_sha256'] != digest(directory/'observations.sqlite')
                or proof['seal_sha256'] != digest(directory/'observations.sqlite.seal.json')):
            raise ValueError('A combined complete pair changed')
        block_hashes[f"{group['block_index']:05d}"] = {
            name:digest(directory/name) for name in
            ('observations.sqlite','observations.sqlite.seal.json','complete.json')}
        paired = []
        loaded = list(observations(directory/'observations.sqlite',expected_metadata,group['executions']))
        if len(loaded) != 2 or len(proof.get('reviews',())) != 2:
            raise ValueError('Every combined block must contain its exact pair')
        for index,(key,record,trace) in enumerate(loaded):
            review = source.verify(key,record,trace,archived[key[:-1]],models[key[-2]])
            if review != proof['reviews'][index]:
                raise ValueError('Independent V37 verification differs in combined review')
            cases.append(dict(case=list(key),row=record['row'],calls=record['calls'],review=review,
                execution_segment='retained_v37' if retained else RECOVERY_SEGMENT))
            paired.append(source.non_timing(record,trace))
        if paired[0] != paired[1]:
            raise ValueError('A combined original/memoized pair differs')
    require_full_coverage(cases,original_plan['groups'])
    cells = []
    for rule in (None,*source.RULES):
        for plant in (None,*study.recovery.PROFILES):
            for implementation in source.IMPLEMENTATIONS:
                selected = [case for case in cases if case['case'][-1] == implementation
                    and (rule is None or case['row']['rule'] == rule)
                    and (plant is None or case['row']['plant_name'] == plant)]
                times = [call['wall_ms'] for case in selected for call in case['calls']]
                work = [case['review']['cache_work'] for case in selected] if implementation == 'memoized' else []
                cells.append(dict(rule=rule,plant=plant,implementation=implementation,runs=len(selected),
                    calls=len(times),terminal_calls=sum(call['error'] is not None for case in selected for call in case['calls']),
                    total_ms=sum(times),mean_ms=float(np.mean(times)),p95_ms=float(np.quantile(times,.95)),
                    max_ms=max(times),over_500ms=sum(value > 500. for value in times),
                    cache_requests=sum(item['requests'] for item in work),cache_hits=sum(item['hits'] for item in work),
                    cache_capacity_skips=sum(item['capacity_skips'] for item in work),
                    computed_successful_transitions=sum(item['computed_successful_transitions'] for item in work),
                    logical_successful_transitions=sum(item['logical_successful_transitions'] for item in work)))
    target = OUT/'analysis'
    target.mkdir(exist_ok=False)
    report = dict(plan_sha256=digest(OUT/'plan.json'),original_plan_sha256=digest(PRIOR/'plan.json'),
        actual_sumo_runs=2*TOTAL_BLOCKS,retained_actual_sumo_runs=2*RETAINED_BLOCKS,
        recovered_actual_sumo_runs=RECOVERY_RUNS,pairs=TOTAL_BLOCKS,
        all_original_non_timing_fields_exact=True,all_independent_physical_reviews_pass=True,
        exact_full_180_pair_coverage=True,cases=cases,cells=cells,new_independent_initials=0,
        new_risk_certificate=False,original_exit_status='unknown',
        completion_native_pid=launch['native_pid'],completion_exit_code=receipt['exit_code'],
        execution_receipts_sha256=execution_receipts,
        excluded_partial_records=plan['partial_record_count'],
        excluded_partial_records_are_engineering_only=True,scope=read(PRIOR/'plan.json')['scope'],
        timing_scope=plan['timing'],work_scope=read(PRIOR/'plan.json')['work'])
    exclusive_json(target/'report.json',report)
    exclusive_json(target/'completion_verification.json',dict(
        plan_sha256=digest(OUT/'plan.json'),original_plan_sha256=digest(PRIOR/'plan.json'),
        native_pid=launch['native_pid'],exit_code=receipt['exit_code'],
        actual_sumo_runs=2*TOTAL_BLOCKS,retained_blocks=RETAINED_BLOCKS,
        recovered_blocks=RECOVERY_BLOCKS,aggregate_blocks=TOTAL_BLOCKS,
        exact_full_coverage=True,execution_receipts_sha256=execution_receipts,
        block_files_sha256=block_hashes,report_sha256=digest(target/'report.json')))
    exclusive_json(target/'sha256.json',dict(report=digest(target/'report.json'),
        completion_verification=digest(target/'completion_verification.json')))
    print(json.dumps(dict(actual_sumo_runs=360,pairs=180,all_exact=True,
        totals=[cell for cell in cells if cell['rule'] is None and cell['plant'] is None])),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('freeze','worker','watch','summarize'))
    globals()[parser.parse_args().command]()
