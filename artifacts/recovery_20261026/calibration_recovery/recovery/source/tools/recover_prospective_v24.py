"""Explicit continuation of the interrupted first prospective calibration.

The scientific plan and all controllers stay byte-identical. Complete blocks
are copied unchanged; partial blocks retain their exact SQLite record blobs in
new append-only stores. Every interrupted boundary must replay exactly before
any missing role is collected. Original exits remain unknown.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

import execute_prospective_v23 as original
from coopgate.episode_store_v21 import EpisodeStore, encode, inventory, readonly, unpack
from coopgate.observable_initial_v19 import ObservableInitialMergeEnv
from experiment_ego_v13 import run_episode

ROOT = original.ROOT
PRIOR = original.OUTPUTS['prospective']
OUTPUT = ROOT/'artifacts/v24/prospective_calibration_recovery_20261022'
RECOVERY = OUTPUT/'recovery'
EVIDENCE = ROOT/'artifacts/v24/prospective_calibration_interruption_20261022'
SCRIPT = 'tools/recover_prospective_v24.py'
SOURCES = (SCRIPT, 'tests/test_recover_prospective_v24.py')


def prefix_length(state, metadata, expected):
    present = state['cases']
    n = len(present)
    if (state['metadata'] != metadata or not 0 < n < len(expected)
            or present != [list(c) for c in expected[:n]]
            or len({tuple(c) for c in present}) != n
            or len(state['row_sha256']) != n or len(state['trace_sha256']) != n):
        raise ValueError('Interrupted block must be an exact nonempty proper prefix')
    return n


def prefix_records(path, metadata, expected, expected_sha):
    if original.digest(path) != expected_sha:
        raise ValueError('Original partial database changed')
    state = inventory(path)
    prefix_length(state, metadata, expected)
    with readonly(path) as connection:
        records = list(connection.execute('SELECT position,case_key,row_blob,trace_blob,row_sha256,trace_sha256 '
                                          'FROM episodes ORDER BY position'))
    # Check stored blobs too, not merely the checksum columns.
    for _, _, row, trace, rh, th in records:
        unpack(row, rh)
        unpack(trace, th)
    if original.digest(path) != expected_sha:
        raise ValueError('Original partial database changed during read')
    return state, records


def copy_prefix(path, destination, metadata, expected, expected_sha):
    if Path(path).resolve() == Path(destination).resolve():
        raise ValueError('Never reopen an original interrupted database for writing')
    state, records = prefix_records(path, metadata, expected, expected_sha)
    store = EpisodeStore(destination, metadata)
    try:
        with store.connection:
            store.connection.executemany('INSERT INTO episodes VALUES(?,?,?,?,?,?)', records)
        store.position = len(records)
        if inventory(destination) != state:
            raise ValueError('Copied prefix identities or checksums changed')
    except BaseException:
        store.close()
        raise
    return store, records


def previous_files(inventory_record):
    directory = PRIOR/'calibration'
    expected = inventory_record['original_calibration_files']
    current = {p.relative_to(directory).as_posix():original.digest(p)
               for p in directory.rglob('*') if p.is_file()}
    if current != expected:
        raise ValueError('Original interrupted files differ from the immutable snapshot')
    return expected


def freeze():
    if OUTPUT.exists():
        raise FileExistsError('A new explicit protocol is required after another interruption')
    original.require_project_stopped(original.native_snapshot())
    _, plan = original.load('prospective')
    original.sources_committed([ROOT/name for name in SOURCES])
    original.require_committed([EVIDENCE/'inventory.json', EVIDENCE/'original_calibration.zip'])
    prior = original.read(EVIDENCE/'inventory.json')
    if (prior['original_plan_sha256'] != original.digest(PRIOR/'frozen/plan.json')
            or prior['original_exit_status'] != 'unknown'
            or prior['original_supervisor_exit_receipt_present']
            or prior['partial_effectiveness_outcomes_read']
            or prior['risk_attempt_unchanged'] != plan['attempt']
            or prior['allocated_error_unchanged'] != plan['allocated_error']
            or original.digest(EVIDENCE/'original_calibration.zip') != prior['archive_sha256']):
        raise ValueError('Interruption evidence or original statistical attempt changed')
    previous_files(prior)
    spec = original.stage_spec(plan, 'calibration')
    blocks = []
    for item in prior['blocks']:
        index = item['initial_index']
        directory = PRIOR/'calibration/blocks'/f'{index:05d}'
        state = inventory(directory/'observations.sqlite')
        if state != item['inventory']:
            raise ValueError('Preserved block inventory changed')
        complete = all(item[k] for k in ('sealed', 'complete', 'manifest'))
        expected = original.schedule(spec, index)
        metadata = original.block_metadata(PRIOR, 'calibration', index, None)
        if complete:
            original.verify_manifest(directory)
            if (state['metadata'] != metadata or state['cases'] != [list(c) for c in expected]
                    or original.read(directory/'observations.sqlite.seal.json')['database_sha256']
                    != original.digest(directory/'observations.sqlite')):
                raise ValueError('Completed prefix is inconsistent')
        else:
            prefix_length(state, metadata, expected)
            if any(item[k] for k in ('sealed', 'complete', 'manifest')):
                raise ValueError('Unexpected partially sealed state requires explicit inspection')
            if (directory/'decision.json').exists() != (len(state['cases']) >= 10):
                raise ValueError('Prediction receipt does not match committed prefix boundary')
        blocks.append(dict(initial_index=index, records=len(state['cases']), complete=complete,
            database_sha256=original.digest(directory/'observations.sqlite'),
            decision_sha256=original.digest(directory/'decision.json') if (directory/'decision.json').exists() else None))
    for shard in range(spec['shards']):
        actual = [b['initial_index'] for b in blocks if b['initial_index'] % spec['shards'] == shard]
        if actual != original.assigned(spec, shard)[:len(actual)]:
            raise ValueError('Original worker records are not a contiguous scheduled prefix')
    retained = sum(b['records'] for b in blocks)
    if (len(blocks) != prior['initial_blocks_started'] or retained != prior['committed_observation_roles']
            or sum(b['complete'] for b in blocks) != prior['fully_completed_initials']):
        raise ValueError('Interrupted compute inventory changed')
    RECOVERY.mkdir(parents=True)
    (OUTPUT/'frozen').mkdir()
    shutil.copy2(PRIOR/'frozen/plan.json', OUTPUT/'frozen/plan.json')
    recovery = dict(utc=original.utc(), original_plan_sha256=original.digest(OUTPUT/'frozen/plan.json'),
        inventory_sha256=original.digest(EVIDENCE/'inventory.json'), archive_sha256=prior['archive_sha256'],
        source_sha256={name:original.digest(ROOT/name) for name in SOURCES},
        original_exit_status='unknown', blocks=blocks, retained_roles=retained,
        missing_roles=spec['total_sumo_runs']-retained, boundary_replays=sum(not b['complete'] for b in blocks),
        initial_count=spec['n_initials'], confirmation_initial_count=plan['stages']['confirmation']['n_initials'],
        original_attempt=plan['attempt'], allocated_error=plan['allocated_error'],
        schedule_models_thresholds_unchanged=True, statistical_sample_size_unchanged=True,
        boundary_rule='All three last committed roles must replay exactly before collecting any missing role; save observations before assertions; no extra statistical N',
        recovery_rule='Copy sealed blocks byte-for-byte, copy partial SQLite row and trace blobs exactly, append only missing original roles; never reopen prior database',
        timing_rule='Preserve original per-role and existing prediction timings; unknown interrupted full-block/full-prediction wall time stays null; report resumed segment separately',
        analysis_rule='Original calibration and confirmation functions only after full independent review and actual successful native exits; confirmation still executes if primary calibration fails',
        repeated_interruption_rule='Preserve all new records; never automatically restart this recovery')
    original.exclusive_json(RECOVERY/'plan.json', recovery)
    for name in SOURCES:
        target = RECOVERY/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
    original.exclusive_json(RECOVERY/'sha256.json', original.manifest(RECOVERY))
    print(json.dumps({k:recovery[k] for k in ('retained_roles','missing_roles','boundary_replays','initial_count')}), flush=True)


def inputs():
    _, plan = original.load('prospective')
    original.verify_manifest(RECOVERY)
    original.require_committed([RECOVERY/'plan.json', RECOVERY/'sha256.json', OUTPUT/'frozen/plan.json'])
    recovery = original.read(RECOVERY/'plan.json')
    if (original.digest(OUTPUT/'frozen/plan.json') != original.digest(PRIOR/'frozen/plan.json')
            or original.digest(OUTPUT/'frozen/plan.json') != recovery['original_plan_sha256']
            or original.digest(EVIDENCE/'inventory.json') != recovery['inventory_sha256']
            or original.digest(EVIDENCE/'original_calibration.zip') != recovery['archive_sha256']
            or recovery['original_attempt'] != plan['attempt'] or recovery['allocated_error'] != plan['allocated_error']):
        raise ValueError('Frozen recovery provenance or statistical identity changed')
    for name, expected in recovery['source_sha256'].items():
        if original.digest(ROOT/name) != expected:
            raise ValueError('Recovery implementation changed: '+name)
    previous_files(original.read(EVIDENCE/'inventory.json'))
    return recovery, plan


def boundary(entry, plan, env, fs):
    index = entry['initial_index']
    spec = plan['stages']['calibration']
    metadata = original.block_metadata(OUTPUT, 'calibration', index, None)
    _, records = prefix_records(PRIOR/'calibration/blocks'/f'{index:05d}'/'observations.sqlite',
                               metadata, original.schedule(spec,index), entry['database_sha256'])
    _, key, row_blob, trace_blob, rh, th = records[-1]
    case, old_row, old_trace = tuple(json.loads(key)), unpack(row_blob,rh), unpack(trace_blob,th)
    directory = OUTPUT/'calibration/boundary'/f'{index:05d}'
    directory.mkdir(parents=True, exist_ok=False)
    if case[-1] == 'observed_shadow':
        with ObservableInitialMergeEnv(np.asarray(old_row['initial'])) as observed_env:
            row, trace, observed = run_episode(observed_env, 1, 'unused', 'clean', fs[case[2]], True)
        row.update(controller=case[2], initial=observed.tolist())
    else:
        row, trace = original.execute(env, case[:3], spec, fs[case[2]])
    original.exclusive_json(directory/'observation.json', dict(case=case, row=row, trace=trace))
    original.compare_replay(old_row, row, old_trace, trace)
    if case[-1] != 'observed_shadow':
        original.verify_record(row, trace, spec)
        reviewer = original.IndependentReview(dict(spec=spec, models=plan['models']))
        reviewer.check(row, trace)
    original.exclusive_json(directory/'verified.json', dict(case=case, exact_non_timing_match=True,
        original_database_sha256=entry['database_sha256'], actual_sumo_runs=1, new_independent_initials=0))
    original.exclusive_json(directory/'sha256.json', original.manifest(directory))


def complete_partial(entry, plan, env, fs):
    index = entry['initial_index']
    spec = plan['stages']['calibration']
    expected = original.schedule(spec, index)
    directory = OUTPUT/'calibration/blocks'/f'{index:05d}'
    directory.mkdir(parents=True, exist_ok=False)
    previous = PRIOR/'calibration/blocks'/f'{index:05d}'
    metadata = original.block_metadata(OUTPUT, 'calibration', index, None)
    started = time.perf_counter()
    store, records = copy_prefix(previous/'observations.sqlite', directory/'observations.sqlite',
                                 metadata, expected, entry['database_sha256'])
    n = len(records)
    try:
        rows = {json.loads(key)[2]:unpack(row,rh) for _,key,row,_,rh,_ in records[:10]}
        env.reset(spec['seed_base']+index, original.regime_for(spec,index))
        observed = env.x.copy()
        if n < 10:
            shadow_started = time.perf_counter()
            with ObservableInitialMergeEnv(observed) as shadow_env:
                for case in expected[n:10]:
                    row, trace, current = run_episode(shadow_env, 1, 'unused', 'clean', fs[case[2]], True)
                    row.update(controller=case[2], initial=current.tolist())
                    store.append(case, row, trace)
                    rows[case[2]] = row
            prediction = original.prediction_from_rows(observed, plan['family'], rows)
            prediction.update(full_prediction_wall_ms=None, actual_prediction_runs=10,
                resumed_prediction_wall_ms=(time.perf_counter()-shadow_started)*1000,
                original_prediction_wall_time_unknown=True)
            if inventory(directory/'observations.sqlite')['cases'] != [list(c) for c in expected[:10]]:
                raise ValueError('Observed predictions must precede all real collection records')
            selection_started = time.perf_counter()
            choices, owners = None, None  # Original calibration has no fitted selector yet.
            selection_wall = (time.perf_counter()-selection_started)*1000
            original.exclusive_json(directory/'decision.json', dict(prediction=prediction, choices=choices, owners=owners,
                calibration_sha256=None, selection_wall_ms=selection_wall, selection_performed=False,
                actual_main_records_before_selection=0, observed_shadow_records_before_selection=10,
                recovery_plan_sha256=original.digest(RECOVERY/'plan.json'), utc=original.utc()))
        else:
            if original.digest(previous/'decision.json') != entry['decision_sha256']:
                raise ValueError('Previously completed observable prediction changed')
            shutil.copy2(previous/'decision.json', directory/'decision.json')
            saved = original.read(directory/'decision.json')
            rebuilt = original.prediction_from_rows(observed, plan['family'], rows)
            for key, value in rebuilt.items():
                np.testing.assert_equal(value, saved['prediction'][key])
            if saved['choices'] is not None or saved['owners'] is not None:
                raise ValueError('Calibration cannot select using its own outcomes')
        for case in expected[max(n,10):]:
            row, trace = original.execute(env, case[:3], spec, fs[case[2]])
            store.append(case, row, trace)
            original.verify_record(row, trace, spec)
        store.seal(expected)
        preserved = inventory(directory/'observations.sqlite')
        initial = inventory(previous/'observations.sqlite')
        for key in ('cases','row_sha256','trace_sha256'):
            if preserved[key][:n] != initial[key]:
                raise ValueError('Continuation altered a retained record')
        original.exclusive_json(directory/'complete.json', dict(initial_index=index, actual_sumo_runs=len(expected),
            retained_sumo_roles=n, new_sumo_roles=len(expected)-n, full_initial_wall_ms=None,
            original_full_initial_wall_time_unknown=True, resumed_segment_wall_ms=(time.perf_counter()-started)*1000,
            plan_sha256=original.digest(OUTPUT/'frozen/plan.json'), calibration_sha256=None,
            recovery_plan_sha256=original.digest(RECOVERY/'plan.json'), original_database_sha256=entry['database_sha256']))
        original.exclusive_json(directory/'sha256.json', original.manifest(directory))
    finally:
        store.close()


def phases(stage):
    return ('boundary','collect','review') if stage == 'calibration' else ('collect','review')


def await_parent_ack(stage, role):
    directory = OUTPUT/stage/'execution'
    path = directory/f'{role}.native.json'
    if path.exists():
        raise FileExistsError('Preserve native startup identity')
    original.atomic_bytes(path, encode(dict(native_pid=os.getpid(), parent_pid=os.getppid(),
                                            stage=stage, role=role, utc=original.utc())))
    ack = directory/f'{role}.acknowledged.json'
    deadline = time.monotonic()+180
    while not ack.exists():
        if time.monotonic() > deadline:
            raise TimeoutError('Parent did not acknowledge the native identity; no study work started')
        time.sleep(.2)
    proof = original.read(ack)
    if (proof['native_pid'] != os.getpid() or proof['parent_pid'] != os.getppid()
            or proof['native_start_sha256'] != original.digest(path)):
        raise ValueError('Parent acknowledgment differs from this native process')


def calibration_for(plan, stage):
    if stage == 'calibration':
        return None, None
    complete_stage(plan, 'calibration')
    value, fingerprint = original.calibration_for(OUTPUT, plan, stage)
    if value['recovery_plan_sha256'] != original.digest(RECOVERY/'plan.json'):
        raise ValueError('Final test requires the complete recovered calibration')
    return value, fingerprint


def worker(stage, phase, shard):
    await_parent_ack(stage, f'{phase}_{shard}')
    torch.set_num_threads(1)
    recovery, plan = inputs()
    spec = original.stage_spec(plan, stage)
    if phase not in phases(stage):
        raise ValueError('Unregistered recovery phase')
    ids = original.assigned(spec, shard)
    calibration, cal_sha = calibration_for(plan, stage)
    entries = {b['initial_index']:b for b in recovery['blocks']} if stage == 'calibration' else {}
    counts, extras = Counter(), {}
    if phase == 'review':
        reviewer = original.IndependentReview(dict(spec=spec, models=plan['models']))
        for number,index in enumerate(ids,1):
            counts.update(original.review_initial(OUTPUT,plan,stage,index,calibration,cal_sha,reviewer))
            progress(stage, phase, shard, index, number, len(ids))
        extras = dict(role_counts=dict(counts), review_counts=reviewer.counts, retained_error_cases=reviewer.error_cases)
    else:
        fs = original.factories(dict(spec=spec, models=plan['models']))
        original.warm_factories(fs)
        with original.DisturbedMergeEnv() as env:
            for number,index in enumerate(ids,1):
                entry = entries.get(index)
                if phase == 'boundary':
                    if entry is not None and not entry['complete']:
                        boundary(entry, plan, env, fs)
                        counts['boundary_replays'] += 1
                elif entry is not None and entry['complete']:
                    previous = PRIOR/'calibration/blocks'/f'{index:05d}'
                    destination = OUTPUT/'calibration/blocks'/f'{index:05d}'
                    shutil.copytree(previous, destination)
                    if original.manifest(previous) != original.manifest(destination):
                        raise ValueError('Completed original block changed during copy')
                    original.load_block(OUTPUT,plan,stage,index,None)
                    counts['retained_roles'] += entry['records']
                elif entry is not None:
                    complete_partial(entry, plan, env, fs)
                    counts['retained_roles'] += entry['records']
                    counts['new_roles'] += len(original.schedule(spec,index))-entry['records']
                else:
                    original.collect_initial(OUTPUT,plan,stage,index,env,fs,calibration,cal_sha)
                    counts['new_roles'] += len(original.schedule(spec,index))
                if phase != 'boundary':
                    progress(stage,phase,shard,index,number,len(ids))
        extras = dict(compute_counts=dict(counts))
    original.exclusive_json(OUTPUT/stage/'execution'/f'{phase}_{shard}.complete.json', dict(pid=os.getpid(),
        initial_indices=ids, stage=stage, phase=phase, calibration_sha256=cal_sha,
        plan_sha256=original.digest(OUTPUT/'frozen/plan.json'),
        recovery_plan_sha256=original.digest(RECOVERY/'plan.json'), **extras))


def progress(stage, phase, shard, index, number, total):
    value = dict(utc=original.utc(), stage=stage, phase=phase, shard=shard,
                 completed_index=index, completed_initials=number, assigned_initials=total)
    original.atomic_bytes(OUTPUT/stage/'execution'/f'{phase}_{shard}.progress.json', encode(value))
    print(json.dumps(value), flush=True)


def supervise(stage):
    await_parent_ack(stage, 'supervisor')
    recovery, plan = inputs()
    spec = original.stage_spec(plan, stage)
    calibration_for(plan, stage)
    execution = OUTPUT/stage/'execution'
    ph, rh = original.digest(OUTPUT/'frozen/plan.json'), original.digest(RECOVERY/'plan.json')
    review_counts, roles, compute, receipts = Counter(), Counter(), Counter(), {}
    for phase in phases(stage):
        processes, handles, identities = [], [], []
        try:
            for shard in range(spec['shards']):
                stdout = (execution/f'{phase}_{shard}.stdout.log').open('x',encoding='utf-8')
                stderr = (execution/f'{phase}_{shard}.stderr.log').open('x',encoding='utf-8')
                handles.extend((stdout,stderr))
                process = subprocess.Popen([sys.executable,'-X','utf8',SCRIPT,'worker',stage,phase,str(shard)],
                    cwd=ROOT,stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                processes.append(process)
                identities.append(original.acknowledge_child(process,execution,f'{phase}_{shard}'))
            original.exclusive_json(execution/f'{phase}_launch.json', dict(utc=original.utc(),supervisor_pid=os.getpid(),
                worker_pids=[p.pid for p in processes],worker_native_pids=[p['native_pid'] for p in identities],
                native_identities=identities,plan_sha256=ph,stage=stage,recovery_plan_sha256=rh))
            while True:
                codes = [p.poll() for p in processes]
                original.atomic_bytes(execution/'state.json',encode(dict(utc=original.utc(),phase=phase,
                    exit_codes=codes,worker_pids=[p.pid for p in processes],
                    worker_native_pids=[i['native_pid'] for i in identities],stage=stage,plan_sha256=ph,recovery_plan_sha256=rh)))
                if all(code is not None for code in codes):
                    break
                time.sleep(10)
            original.exclusive_json(execution/f'{phase}_exits.json', dict(utc=original.utc(),
                workers=[dict(pid=p.pid,exit_code=p.returncode) for p in processes]))
            if any(p.returncode != 0 for p in processes):
                raise RuntimeError('Recovery worker failed; preserve prefixes, no automatic restart')
            for shard,process in enumerate(processes):
                path = execution/f'{phase}_{shard}.complete.json'
                receipt = original.read(path)
                if (receipt['pid'] != identities[shard]['native_pid'] or receipt['initial_indices'] != original.assigned(spec,shard)
                        or receipt['plan_sha256'] != ph or receipt['recovery_plan_sha256'] != rh
                        or receipt['stage'] != stage or receipt['phase'] != phase):
                    raise ValueError('Recovery worker receipt does not match live process identity')
                receipts[path.name] = original.digest(path)
                if phase == 'review':
                    review_counts.update(receipt['review_counts'])
                    roles.update(receipt['role_counts'])
                else:
                    compute.update(receipt['compute_counts'])
        finally:
            for handle in handles:
                handle.close()
    retained = recovery['retained_roles'] if stage == 'calibration' else 0
    boundaries = recovery['boundary_replays'] if stage == 'calibration' else 0
    if (roles['main'] != spec['main_runs'] or roles['observed_shadow'] != spec['observed_shadow_runs']
            or roles['deployment'] != spec['deployment_runs'] or review_counts['physical_traces'] != spec['main_runs']
            or compute['retained_roles'] != retained or compute['new_roles'] != spec['total_sumo_runs']-retained
            or compute['boundary_replays'] != boundaries):
        raise ValueError('Complete recovery compute budget or physical review differs')
    hashes = {f'{i:05d}':original.digest(OUTPUT/stage/'blocks'/f'{i:05d}'/'sha256.json') for i in range(spec['n_initials'])}
    original.exclusive_json(execution/'completed.json', dict(utc=original.utc(), stage=stage, plan_sha256=ph,
        recovery_plan_sha256=rh, original_exit_status='unknown', role_counts=dict(roles),review_counts=dict(review_counts),
        compute_counts=dict(compute),receipt_sha256=receipts,block_manifest_sha256=hashes,
        boundary_manifest_sha256={f"{b['initial_index']:05d}":original.digest(
            OUTPUT/stage/'boundary'/f"{b['initial_index']:05d}"/'sha256.json')
            for b in recovery['blocks'] if stage=='calibration' and not b['complete']},
        actual_sumo_runs=spec['total_sumo_runs'], new_independent_initials=spec['n_initials'],
        recovered_schedule_not_extra_independent_initials=True))


def watch(stage):
    _, plan = inputs()
    original.stage_spec(plan,stage)
    original.require_project_stopped(original.native_snapshot())
    calibration_for(plan,stage)
    execution = OUTPUT/stage/'execution'
    execution.mkdir(parents=True,exist_ok=False)
    with (execution/'supervisor.stdout.log').open('x',encoding='utf-8') as out, (execution/'supervisor.stderr.log').open('x',encoding='utf-8') as err:
        process = subprocess.Popen([sys.executable,'-X','utf8',SCRIPT,'supervise',stage],cwd=ROOT,
            stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        native = original.acknowledge_child(process,execution,'supervisor')
        identity = dict(stage=stage,plan_sha256=original.digest(OUTPUT/'frozen/plan.json'),supervisor_pid=process.pid,
                        recovery_plan_sha256=original.digest(RECOVERY/'plan.json'))
        original.exclusive_json(execution/'watcher_launch.json',dict(utc=original.utc(),watcher_pid=os.getpid(),
            supervisor_native_pid=native['native_pid'],native_identity=native,**identity))
        code = process.wait()
        original.exclusive_json(execution/'supervisor_exit.json',dict(utc=original.utc(),exit_code=code,**identity))
    if code != 0:
        raise RuntimeError('Native recovery supervisor failed; retain all records')


def complete_stage(plan, stage):
    recovery, _ = inputs()
    result = original.complete_stage(OUTPUT,plan,stage)
    rh = original.digest(RECOVERY/'plan.json')
    if result['recovery_plan_sha256'] != rh or result['original_exit_status'] != 'unknown':
        raise ValueError('Recovery completion does not preserve original provenance')
    total = plan['stages'][stage]['total_sumo_runs']
    retained = recovery['retained_roles'] if stage=='calibration' else 0
    expected_compute = dict(new_roles=total-retained)
    if stage=='calibration':
        expected_compute.update(retained_roles=retained,boundary_replays=recovery['boundary_replays'])
    if result['compute_counts'] != expected_compute:
        raise ValueError('Retained, new and boundary execution budgets changed')
    if stage == 'calibration':
        execution = OUTPUT/stage/'execution'
        watch = original.read(execution/'watcher_launch.json')
        launch = original.read(execution/'boundary_launch.json')
        exits = original.read(execution/'boundary_exits.json')
        if (launch['supervisor_pid'] != watch['supervisor_native_pid'] or launch['recovery_plan_sha256'] != rh
                or launch['stage'] != stage or launch['plan_sha256'] != recovery['original_plan_sha256']
                or len(launch['worker_pids']) != plan['stages'][stage]['shards']
                or len(set(launch['worker_pids'])) != len(launch['worker_pids'])
                or len(set(launch['worker_native_pids'])) != len(launch['worker_pids'])
                or [r['pid'] for r in exits['workers']] != launch['worker_pids']
                or any(r['exit_code'] != 0 for r in exits['workers'])):
            raise ValueError('All actual boundary workers must have exited successfully')
        for shard,(launcher,native) in enumerate(zip(launch['worker_pids'],launch['worker_native_pids'])):
            original.check_acknowledgment(execution,f'boundary_{shard}',launcher,native)
        for entry in recovery['blocks']:
            index = entry['initial_index']
            previous = PRIOR/stage/'blocks'/f'{index:05d}'
            current = OUTPUT/stage/'blocks'/f'{index:05d}'
            if entry['complete']:
                if original.manifest(previous) != original.manifest(current):
                    raise ValueError('A complete retained block no longer has its original bytes')
            else:
                before,after = inventory(previous/'observations.sqlite'),inventory(current/'observations.sqlite')
                for key in ('cases','row_sha256','trace_sha256'):
                    if before[key] != after[key][:entry['records']]:
                        raise ValueError('A retained partial record changed')
                if (entry['decision_sha256'] is not None
                        and original.digest(current/'decision.json') != entry['decision_sha256']):
                    raise ValueError('A completed original prediction changed')
            if not entry['complete']:
                directory = OUTPUT/stage/'boundary'/f"{entry['initial_index']:05d}"
                original.verify_manifest(directory)
                if original.digest(directory/'sha256.json') != result['boundary_manifest_sha256'][f'{index:05d}']:
                    raise ValueError('Reviewed boundary manifest changed')
                evidence = original.read(directory/'verified.json')
                if not evidence['exact_non_timing_match'] or evidence['original_database_sha256'] != entry['database_sha256']:
                    raise ValueError('Interrupted boundary proof changed')
    return result


def analyze(stage):
    recovery, plan = inputs()
    spec = original.stage_spec(plan,stage)
    original.require_project_stopped(original.native_snapshot())
    complete = complete_stage(plan,stage)
    calibration, cal_sha = calibration_for(plan,stage)
    timing,decisions = [],[]
    def rows():
        for index in range(spec['n_initials']):
            directory,records = original.load_block(OUTPUT,plan,stage,index,cal_sha)
            decision = original.read(directory/'decision.json')
            receipt = original.read(directory/'complete.json')
            timing.append(dict(initial_index=index, full_prediction_wall_ms=decision['prediction']['full_prediction_wall_ms'],
                selection_wall_ms=decision['selection_wall_ms'],full_initial_wall_ms=receipt['full_initial_wall_ms'],
                resumed_segment_wall_ms=receipt.get('resumed_segment_wall_ms'),
                resumed_prediction_wall_ms=decision['prediction'].get('resumed_prediction_wall_ms')))
            decisions.append(decision['choices'])
            for key,row,_ in records:
                if key[-1] == 'collection':
                    yield row
    data = original.paired_observations(rows(),profiles=original.profile_names(plan['family']),
        independent_initials=spec['n_initials'],seed_base=spec['seed_base'],methods=spec['methods'])
    destination = OUTPUT/stage/'analysis'
    destination.mkdir(exist_ok=False)
    if stage == 'calibration':
        result = original.calibrate_observations(data['names'],data['adopted'],data['harmful'],data['differences'],
            data['failed'],family=plan['family'],seed_base=spec['seed_base'],attempt=plan['attempt'])
        result.update(execution_plan_sha256=original.digest(OUTPUT/'frozen/plan.json'),
            recovery_plan_sha256=original.digest(RECOVERY/'plan.json'),original_exit_status='unknown',
            complete_execution_sha256=original.digest(OUTPUT/stage/'execution/completed.json'),
            native_exit_sha256=original.digest(OUTPUT/stage/'execution/supervisor_exit.json'))
        original.exclusive_json(destination/'calibration.json',result)
        passed = result['primary_certificate']['all_fixed_models_both_targets_certified']
    else:
        result = original.evaluate_adapters(data,calibration,family=plan['family'],seeds=original.MODELS,
            seed_base=spec['seed_base'],mode_stream=22117,bootstrap_seed=22123,bootstrap_samples=5000)
        for selection in result['selections']:
            if decisions[selection['initial_index']][selection['adapter']][selection['training_seed']] != selection['profile']:
                raise ValueError('Runtime and offline adapter choices differ')
        result.update(actual_deployment_replays_still_required=False,actual_deployment_runs=spec['deployment_runs'],
            calibration_sha256=cal_sha,execution_plan_sha256=original.digest(OUTPUT/'frozen/plan.json'),
            recovery_plan_sha256=original.digest(RECOVERY/'plan.json'),
            all_observed_runtime_offline_choices_exact=True,independent_review_counts=complete['review_counts'])
        acceptance = original.final_acceptance(calibration,result,plan['family'])
        original.exclusive_json(destination/'comparison.json',result)
        original.exclusive_json(destination/'acceptance.json',acceptance)
        passed = acceptance['complete_primary_success']
    original.exclusive_json(destination/'runtime_accounting.json',dict(per_initial=timing,
        observed_shadow_runs=spec['observed_shadow_runs'],deployment_runs=spec['deployment_runs'],
        independent_initials_added_by_extra_roles=0,real_time_claim=False,
        unknown_historical_full_times_preserved_as_null=True))
    original.exclusive_json(destination/'sha256.json',original.manifest(destination))
    print(json.dumps(dict(stage=stage,primary_complete_success=passed,actual_sumo_runs=spec['total_sumo_runs'])),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest='command',required=True)
    subs.add_parser('freeze')
    for command in ('watch','supervise','worker','analyze'):
        item = subs.add_parser(command)
        item.add_argument('stage',choices=('calibration','confirmation'))
        if command == 'worker':
            item.add_argument('phase',choices=('boundary','collect','review'))
            item.add_argument('shard',type=int)
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze()
    elif args.command == 'worker':
        worker(args.stage,args.phase,args.shard)
    else:
        globals()[args.command](args.stage)
