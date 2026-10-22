"""Preserve 99 committed integration roles and complete the fixed 144-role plan.

This explicit recovery reads interrupted SQLite blocks without modifying them.
Only the registered suffix is executed; boundary replays never add statistical N.
"""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

from coopgate.disturbance_v9 import DisturbedMergeEnv
from coopgate.episode_store_v21 import EpisodeStore, inventory, observations, readonly, unpack
from coopgate.observable_initial_v19 import ObservableInitialMergeEnv
from experiment_ego_v13 import run_episode
from probe_disturbed_gate_v19 import digest, save
from recover_checkpoint_v19 import atomic_bytes
import recover_interrupted_v21 as recovery
import preflight_prospective_runtime_v21 as original
from prospective_runtime_v21 import IndependentReview, execute, verify_record
from resume_frozen_risk_v19 import build_record
from start_final_v19 import require_committed, verify_manifest
from study_gate_v19 import regime_for
from study_matched_transfer_v21 import factories
from study_repair_v20 import trace_metadata

ROOT = original.ROOT
OUTPUT = ROOT / 'artifacts/v21/runtime_preflight_recovery_20261022'
FROZEN = OUTPUT / 'frozen'
EXECUTION = OUTPUT / 'execution'
SOURCES = ('tools/recover_runtime_preflight_v21.py', 'tests/test_runtime_preflight_recovery_v21.py')


def role_schedule(cases):
    return [(*case, role) for case in cases for role in
            (('collection', 'legacy_reference', 'observed_shadow') if case[1] == 'clean'
             else ('collection', 'legacy_reference'))]


def completed_pair_prefix(cases, present):
    """This recovery admits exact completed pairs/triples, never hidden holes."""
    cases = list(map(tuple, cases))
    present = list(map(tuple, present))
    expected = role_schedule(cases)
    if present != expected[:len(present)] or len(set(present)) != len(present):
        raise ValueError('Interrupted roles are not an exact registered prefix')
    boundaries = [len(role_schedule(cases[:i])) for i in range(len(cases) + 1)]
    if len(present) not in boundaries:
        raise ValueError('This recovery requires a complete pair/triple boundary')
    count = boundaries.index(len(present))
    return cases[count:], cases[count - 1] if count and count < len(cases) else None


def original_metadata(shard):
    return dict(role='integration_only', plan_sha256=digest(original.FROZEN / 'plan.json'), shard=shard)


def read_prefix(path, metadata, cases, expected_sha256):
    """Read unsealed prior data only under a frozen explicit recovery identity."""
    if digest(path) != expected_sha256:
        raise ValueError('Interrupted database changed')
    state = inventory(path)
    if state['metadata'] != metadata or state['cases'] != [list(c) for c in cases]:
        raise ValueError('Interrupted database identity/schedule changed')
    with readonly(path) as connection:
        records = [(tuple(json.loads(key)), unpack(row, rh), unpack(trace, th))
                   for key, row, trace, rh, th in connection.execute(
                       'SELECT case_key,row_blob,trace_blob,row_sha256,trace_sha256 FROM episodes ORDER BY position')]
    if digest(path) != expected_sha256:
        raise ValueError('Interrupted database changed during the read')
    return records


def assert_previous_stopped():
    processes = recovery.native_snapshot()
    relevant = [p for p in processes if any(name in (p.get('CommandLine') or '') for name in
                ('tools/preflight_prospective_runtime_v21.py', 'tools/recover_runtime_preflight_v21.py'))]
    recovery.assert_no_existing_workers(relevant)
    return processes


def freeze():
    if OUTPUT.exists():
        raise FileExistsError('Preserve existing integration recovery')
    processes = assert_previous_stopped()
    old = original.inputs()
    exit_record = recovery.read(original.EXECUTION / 'supervisor_exit.json')
    if exit_record['exit_code'] == 0:
        raise ValueError('This recovery is for the interrupted original execution')
    require_committed([ROOT / name for name in SOURCES])
    groups = []
    previous_files = {}
    for shard in range(old['shards']):
        directory = original.OUTPUT / f'shard_{shard}'
        path = directory / 'observations.sqlite'
        if any(directory.glob('observations.sqlite-*')):
            raise ValueError('Inspect unresolved SQLite journal before recovery')
        state = inventory(path)
        if state['metadata'] != original_metadata(shard):
            raise ValueError('Different integration block')
        missing, boundary = completed_pair_prefix(original.cases(old, shard), state['cases'])
        if missing and boundary is None:
            raise ValueError('A retained boundary is required for continuation parity')
        groups.append(dict(shard=shard, existing_roles=state['cases'], missing_cases=missing,
                           missing_roles=role_schedule(missing), boundary_case=boundary,
                           database_sha256=digest(path)))
        for file in directory.iterdir():
            if file.is_file():
                previous_files[file.relative_to(ROOT).as_posix()] = digest(file)
    for file in original.EXECUTION.iterdir():
        if file.is_file():
            previous_files[file.relative_to(ROOT).as_posix()] = digest(file)
    FROZEN.mkdir(parents=True)
    for name in previous_files:
        target = FROZEN / 'previous' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    plan = dict(utc=recovery.utc(), groups=groups, previous_files=previous_files,
                previous_native_snapshot=processes, original_supervisor_exit=exit_record,
                original_plan_sha256=digest(original.FROZEN / 'plan.json'),
                original_planned_roles=old['actual_sumo_runs'],
                retained_roles=sum(len(g['existing_roles']) for g in groups),
                missing_roles=sum(len(g['missing_roles']) for g in groups),
                boundary_replays=sum(bool(g['missing_cases']) for g in groups),
                new_independent_initials=0, new_calibration_data=False,
                sources={name:digest(ROOT / name) for name in SOURCES})
    save(FROZEN / 'plan.json', plan)
    for name in SOURCES:
        target = FROZEN / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    save(FROZEN / 'sha256.json', original.manifest(FROZEN))
    print(json.dumps({k:plan[k] for k in ('retained_roles', 'missing_roles', 'boundary_replays')}), flush=True)


def inputs():
    verify_manifest(FROZEN)
    plan = recovery.read(FROZEN / 'plan.json')
    require_committed([FROZEN / 'plan.json', FROZEN / 'sha256.json', *[ROOT / name for name in SOURCES]])
    old = original.inputs()
    if digest(original.FROZEN / 'plan.json') != plan['original_plan_sha256']:
        raise ValueError('Original integration protocol changed')
    for name, expected in {**plan['previous_files'], **plan['sources']}.items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen integration input changed: ' + name)
    return plan, old


def old_records(group):
    return read_prefix(original.OUTPUT / f"shard_{group['shard']}" / 'observations.sqlite',
                       original_metadata(group['shard']), group['existing_roles'], group['database_sha256'])


def new_metadata(shard):
    return dict(role='integration_continuation', shard=shard, plan_sha256=digest(FROZEN / 'plan.json'))


def compare_shadow(actual, trace, shadow, shadow_trace):
    for field in ('per_vehicle_finish', 'per_vehicle_variation', 'completed', 'collision',
                  'aborted', 'contract_violations'):
        np.testing.assert_equal(shadow[field], actual[field])
    np.testing.assert_allclose(shadow['initial'], actual['initial'], atol=1e-8, rtol=0)
    if len(shadow_trace) != len(trace):
        raise ValueError('Shadow trajectory length differs')
    for a, b in zip(shadow_trace, trace):
        np.testing.assert_array_equal(a['command'], b['command'])
        np.testing.assert_allclose(a['x'], b['x'], atol=1e-8, rtol=0)


def continue_group(group, old):
    shard = group['shard']
    spec, protocol = old['protocol']['spec'], old['protocol']
    out = OUTPUT / 'suffix' / f'shard_{shard}'
    out.mkdir(parents=True, exist_ok=False)
    retained = {key:(row, trace) for key, row, trace in old_records(group)}
    fs = factories(protocol)
    warm = np.array([[-80., 12.], [-85., 13.], [-120., 11.]])
    for factory in fs.values():
        factory(warm).select(warm, np.zeros(3))
    store = EpisodeStore(out / 'observations.sqlite', new_metadata(shard))
    count = 0
    try:
        with ExitStack() as stack:
            env = stack.enter_context(DisturbedMergeEnv())
            reference_env = stack.enter_context(DisturbedMergeEnv())
            boundary = tuple(group['boundary_case'])
            row, trace = execute(env, boundary, spec, fs[boundary[-1]])
            save(out / 'boundary_row.json', row)
            save(out / 'boundary_trace.json', trace)
            verify_record(row, trace, spec)
            previous, old_trace = retained[(*boundary, 'collection')]
            recovery.compare_replay(previous, row, old_trace, trace)
            for case in map(tuple, group['missing_cases']):
                index, mode, name = case
                actual, trace = execute(env, case, spec, fs[name])
                store.append((*case, 'collection'), actual, trace)
                verify_record(actual, trace, spec)
                raw, reference, initial = run_episode(reference_env, spec['seed_base'],
                    regime_for(spec, index), mode, fs[name], True)
                legacy, _, _, _ = build_record(raw, reference, initial, index, name, spec['methods'][name], spec)
                legacy.update(trace_metadata(reference))
                store.append((*case, 'legacy_reference'), legacy, reference)
                verify_record(legacy, reference, spec)
                recovery.compare_replay(legacy, actual, reference, trace)
                count += 2
                if mode == 'clean':
                    with ObservableInitialMergeEnv(np.asarray(actual['initial'])) as shadow_env:
                        shadow, shadow_trace, observed = run_episode(shadow_env, 1, 'unused', 'clean', fs[name], True)
                    shadow.update(initial=observed.tolist(), controller=name)
                    store.append((*case, 'observed_shadow'), shadow, shadow_trace)
                    compare_shadow(actual, trace, shadow, shadow_trace)
                    count += 1
                atomic_bytes(EXECUTION / 'state.json', (json.dumps(dict(utc=recovery.utc(), stage='suffix',
                    shard=shard, completed_roles=count, expected_roles=len(group['missing_roles']), pid=os.getpid()))+'\n').encode())
        store.seal(group['missing_roles'])
        save(out / 'complete.json', dict(pid=os.getpid(), roles=count, boundary_replays=1,
                                        plan_sha256=digest(FROZEN / 'plan.json')))
        save(out / 'sha256.json', original.manifest(out))
    finally:
        store.close()


def full_audit(plan, old):
    review = IndependentReview(old['protocol'])
    spec = old['protocol']['spec']
    metadata = dict(role='complete_recovered_integration', plan_sha256=digest(FROZEN / 'plan.json'))
    store = EpisodeStore(OUTPUT / 'complete.sqlite', metadata)
    expected = []
    reference_count = shadow_count = 0
    try:
        for group in plan['groups']:
            records = old_records(group)
            if group['missing_roles']:
                directory = OUTPUT / 'suffix' / f"shard_{group['shard']}"
                verify_manifest(directory)
                records.extend(observations(directory / 'observations.sqlite', new_metadata(group['shard']), group['missing_roles']))
            keys = [key for key, _, _ in records]
            if keys != role_schedule(original.cases(old, group['shard'])):
                raise ValueError('Recovered integration is not complete')
            lookup = {key:(row, trace) for key, row, trace in records}
            for case in original.cases(old, group['shard']):
                actual, trace = lookup[(*case, 'collection')]
                legacy, legacy_trace = lookup[(*case, 'legacy_reference')]
                review.check(actual, trace)
                verify_record(legacy, legacy_trace, spec)
                recovery.compare_replay(legacy, actual, legacy_trace, trace)
                reference_count += 1
                if case[1] == 'clean':
                    shadow, shadow_trace = lookup[(*case, 'observed_shadow')]
                    compare_shadow(actual, trace, shadow, shadow_trace)
                    shadow_count += 1
            for key, row, trace in records:
                store.append(key, row, trace)
            expected.extend(keys)
            print(json.dumps(dict(stage='review', verified_pairs=reference_count, shadows=shadow_count)), flush=True)
        if len(expected) != old['actual_sumo_runs'] or reference_count != 64 or shadow_count != 16:
            raise ValueError('Incomplete integration budget')
        store.seal(expected)
        if sum(1 for _ in observations(OUTPUT / 'complete.sqlite', metadata, expected)) != len(expected):
            raise ValueError('Complete recovered storage does not reload')
    finally:
        store.close()
    inputs()
    save(OUTPUT / 'audit.json', dict(unique_original_planned_roles=len(expected),
        original_retained_roles=plan['retained_roles'], new_suffix_runs=plan['missing_roles'],
        new_boundary_replays=plan['boundary_replays'], collection_reference_pairs=reference_count,
        observed_shadow_runs=shadow_count, all_reference_pairs_exact=True,
        all_observed_shadow_commands_and_outcomes_match=True, review_counts=review.counts,
        retained_error_cases=review.error_cases, new_independent_initials=0, new_risk_certified=False,
        original_supervisor_exit=plan['original_supervisor_exit'],
        plan_sha256=digest(FROZEN / 'plan.json'), complete_database_sha256=digest(OUTPUT / 'complete.sqlite')))


def run():
    torch.set_num_threads(1)
    plan, old = inputs()
    try:
        for group in plan['groups']:
            if group['missing_cases']:
                continue_group(group, old)
        full_audit(plan, old)
        save(EXECUTION / 'completed.json', dict(utc=recovery.utc(), audit_sha256=digest(OUTPUT / 'audit.json')))
    except Exception as error:
        save(EXECUTION / 'failure.json', dict(utc=recovery.utc(), type=type(error).__name__, error=str(error)))
        raise


def watch():
    inputs()
    assert_previous_stopped()
    EXECUTION.mkdir(exist_ok=False)
    command = [sys.executable, '-X', 'utf8', 'tools/recover_runtime_preflight_v21.py', 'run']
    with (EXECUTION / 'supervisor.stdout.log').open('x') as out, (EXECUTION / 'supervisor.stderr.log').open('x') as err:
        process = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        save(EXECUTION / 'watcher_launch.json', dict(utc=recovery.utc(), watcher_pid=os.getpid(),
            supervisor_pid=process.pid, command=command, plan_sha256=digest(FROZEN / 'plan.json')))
        code = process.wait()
        save(EXECUTION / 'supervisor_exit.json', dict(utc=recovery.utc(), supervisor_pid=process.pid,
                                                   exit_code=code, plan_sha256=digest(FROZEN / 'plan.json')))
    if code:
        raise SystemExit(code)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['freeze', 'watch', 'run'])
    globals()[parser.parse_args().command]()
