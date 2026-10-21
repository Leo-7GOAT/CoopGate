"""Append-only recovery of the two interrupted, frozen V21 SUMO studies.

Original exits are unknown, never reconstructed. Original observations stay in
place and in immutable zip snapshots. Only the frozen missing suffix is run;
new process receipts and a complete independent review justify recovered data.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np
import torch

from coopgate.disturbance_v9 import DisturbedMergeEnv
from coopgate.matched_shift_v21 import MatchedShiftedMergeEnv
from experiment_ego_v13 import run_episode
from probe_disturbed_gate_v19 import digest, save
from recover_checkpoint_v19 import atomic_bytes
from start_final_v19 import require_committed, verify_manifest
from study_gate_v19 import read_protocol
import study_phase_repair_v21 as phase
import study_matched_transfer_v21 as transfer

ROOT = Path(__file__).resolve().parents[1]
RECOVERY = ROOT / 'artifacts/v21/interrupted_recovery_20261021'
FROZEN = RECOVERY / 'frozen'
EXECUTION = RECOVERY / 'execution'
STUDIES = {'phase': phase.STUDY, 'transfer': transfer.STUDY}
MODULES = {'phase': phase, 'transfer': transfer}
SOURCES = ('tools/recover_interrupted_v21.py', 'tests/test_interrupted_recovery_v21.py')
ROW_TIMINGS = {'setup_ms', 'controller_ms'}
TRACE_TIMINGS = {'gate_ms'}


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def manifest(directory):
    return {p.relative_to(directory).as_posix(): digest(p)
            for p in sorted(directory.rglob('*')) if p.is_file() and p.name != 'sha256.json'}


def native_snapshot():
    command = ("Get-CimInstance Win32_Process | Where-Object { "
               "$_.Name -match '^(python|sumo)' } | Select-Object "
               "ProcessId,ParentProcessId,Name,CreationDate,CommandLine | ConvertTo-Json -Depth 3")
    result = subprocess.run(['powershell', '-NoProfile', '-Command', command],
                            text=True, capture_output=True, check=True)
    data = json.loads(result.stdout) if result.stdout.strip() else []
    return data if isinstance(data, list) else [data]


def assert_no_existing_workers(processes, own_pid=None, parent_pid=None):
    own_pid = os.getpid() if own_pid is None else own_pid
    parent_pid = os.getppid() if parent_pid is None else parent_pid
    ancestors = {own_pid, parent_pid}
    while True:
        extra = {p['ParentProcessId'] for p in processes if p['ProcessId'] in ancestors}
        if extra <= ancestors:
            break
        ancestors |= extra
    # Conservatively decline while any other Python or SUMO process is alive.
    other = [p for p in processes if p['ProcessId'] not in ancestors]
    if other:
        raise ValueError('Existing Python/SUMO process requires inspection before recovery')


def identity(study, row):
    if study == 'phase':
        return row['initial_index'], row['disturbance'], row['controller']
    return transfer.identity(row)


def trace_name(study, case):
    return (f'trace_{case[0]}_{case[1]}_{case[2]}.json' if study == 'phase'
            else f'trace_{case[0]}_{case[4]}_{case[5]}.json')


def row_name(study, case):
    return trace_name(study, case).replace('trace_', 'row_', 1)


def prefix_layout(expected, observed, trace_files, name_fn):
    """Accept a complete schedule prefix and at most its next orphan trace."""
    expected, observed = list(map(tuple, expected)), list(map(tuple, observed))
    if observed != expected[:len(observed)] or len(set(observed)) != len(observed):
        raise ValueError('Saved observations are not an exact frozen schedule prefix')
    required = {name_fn(case) for case in observed}
    if not required <= set(trace_files):
        raise ValueError('Saved observation has no physical trace')
    orphan = set(trace_files) - required
    permitted = {name_fn(expected[len(observed)])} if len(observed) < len(expected) else set()
    if not orphan <= permitted:
        raise ValueError('Unexpected trace outside the prefix boundary')
    return expected[len(observed):], sorted(orphan)


def replay_cases(study, prefixes, spec):
    """Outcome-blind coverage: first controller/plant and both shard boundaries."""
    selected, seen_names, seen_plants = [], set(), set()
    for shard, rows in prefixes.items():
        for i, row in enumerate(rows):
            name = row['controller']
            plant = row.get('plant_name', 'additive')
            if name not in seen_names or plant not in seen_plants or i in (0, len(rows) - 1):
                selected.append(dict(shard=shard, case=list(identity(study, row))))
            seen_names.add(name)
            seen_plants.add(plant)
    if seen_names != set(spec['methods']):
        raise ValueError('Prefix cannot provide replay coverage of every controller')
    return selected


def compare_replay(old_row, new_row, old_trace, new_trace):
    def remove(value, keys):
        return {k: v for k, v in value.items() if k not in keys}
    if old_row is not None:
        np.testing.assert_equal(remove(old_row, ROW_TIMINGS), remove(new_row, ROW_TIMINGS))
    np.testing.assert_equal([remove(t, TRACE_TIMINGS) for t in old_trace],
                            [remove(t, TRACE_TIMINGS) for t in new_trace])


def freeze():
    if RECOVERY.exists():
        raise FileExistsError('Preserve the original recovery record')
    processes = native_snapshot()
    assert_no_existing_workers(processes)
    protocols = {name: read_protocol(path)[1] for name, path in STUDIES.items()}
    require_committed([ROOT / name for name in SOURCES])
    FROZEN.mkdir(parents=True)
    studies = {}
    for study, directory in STUDIES.items():
        spec = protocols[study]['spec']
        prefixes, shards = {}, {}
        for shard in range(spec['shards']):
            original = directory / f'shard_{shard}'
            rows = read(original / 'episodes.json')
            expected = MODULES[study].schedule(spec, shard)
            missing, orphan = prefix_layout(expected, [identity(study, r) for r in rows],
                [p.name for p in original.glob('trace_*.json')], lambda c: trace_name(study, c))
            if (original / 'complete.json').exists():
                complete = read(original / 'complete.json')
                if (missing or complete['actual_sumo_runs'] != len(rows)
                        or complete['episodes_sha256'] != digest(original / 'episodes.json')
                        or complete['protocol_sha256'] != digest(directory / 'protocol.json')):
                    raise ValueError('Existing complete receipt contradicts prefix')
            files = {p.name: digest(p) for p in sorted(original.iterdir()) if p.is_file()}
            archive = FROZEN / f'{study}_shard_{shard}.zip'
            with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as output:
                for name in files:
                    output.write(original / name, name)
            if archive.stat().st_size >= 95_000_000:
                raise ValueError('Snapshot archive exceeds public repository file budget')
            with zipfile.ZipFile(archive) as snapshot:
                if snapshot.testzip() is not None or set(snapshot.namelist()) != set(files):
                    raise ValueError('Snapshot archive failed integrity verification')
            shards[str(shard)] = dict(prefix_count=len(rows), expected_count=len(expected),
                missing_cases=missing, orphan_traces=orphan, original_files=files,
                archive=archive.name, archive_sha256=digest(archive))
            prefixes[shard] = rows
        studies[study] = dict(directory=directory.relative_to(ROOT).as_posix(),
            protocol_sha256=digest(directory / 'protocol.json'), shards=shards,
            replay_cases=replay_cases(study, prefixes, spec))
    plan = dict(utc=utc(), purpose='Complete interrupted frozen development without replacing observations',
        original_exit_status='unknown; original native processes are absent and temporary logs are missing',
        original_exit_receipts_reconstructed=False, original_process_snapshot=processes,
        case_selection='Exact missing schedule suffix only; no outcome selection or new initial IDs',
        orphan_policy='Archive in place; rerun the next uncommitted case and compare every non-timing trace field when parseable',
        replay_policy='First existing case per controller and plant, plus first and last per shard; all non-timing fields must match',
        independent_review='All original and resumed cases, using the unchanged frozen reviewers and analysis',
        timing_scope='Original timings retained; new timings tagged by epoch. No isolated deployment speed claim.',
        original_files_modified=False, policy_or_statistical_rules_changed=False,
        partial_effectiveness_or_risk_results_read=False, studies=studies,
        source_sha256={name: digest(ROOT / name) for name in SOURCES})
    save(FROZEN / 'plan.json', plan)
    for name in SOURCES:
        destination = FROZEN / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    save(FROZEN / 'sha256.json', manifest(FROZEN))
    print(json.dumps({name: dict(retained=sum(s['prefix_count'] for s in p['shards'].values()),
        missing=sum(len(s['missing_cases']) for s in p['shards'].values()),
        parity_replays=len(p['replay_cases'])) for name, p in studies.items()}), flush=True)


def inputs(check_original=False):
    verify_manifest(FROZEN)
    plan = read(FROZEN / 'plan.json')
    require_committed([FROZEN / 'plan.json', FROZEN / 'sha256.json', *[ROOT / name for name in SOURCES]])
    for name, expected in plan['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen recovery implementation changed')
    protocols = {}
    for study, item in plan['studies'].items():
        directory = ROOT / item['directory']
        if digest(directory / 'protocol.json') != item['protocol_sha256']:
            raise ValueError('Original protocol changed')
        protocols[study] = read_protocol(directory)[1]
        if check_original:
            for shard, entry in item['shards'].items():
                original = directory / f'shard_{shard}'
                actual_names = {p.name for p in original.iterdir() if p.is_file()}
                if actual_names != set(entry['original_files']):
                    raise ValueError('Original directory changed after recovery freeze')
                for name, expected in entry['original_files'].items():
                    if digest(original / name) != expected:
                        raise ValueError('Original observation changed after freeze')
    return plan, protocols


def original_case(study, case, plan):
    shard = case[0] % 3
    item = plan['studies'][study]
    directory = ROOT / item['directory'] / f'shard_{shard}'
    entry = item['shards'][str(shard)]
    if digest(directory / 'episodes.json') != entry['original_files']['episodes.json']:
        raise ValueError('Original checkpoint changed')
    lookup = {identity(study, row): row for row in read(directory / 'episodes.json')}
    path = directory / trace_name(study, case)
    if digest(path) != entry['original_files'][path.name]:
        raise ValueError('Original trace changed')
    return lookup.get(tuple(case)), read(path)


def execute(study, case, spec, factory, env):
    if study == 'phase':
        index, mode, name = case
        row, trace, initial = run_episode(env, spec['seed_base'] + index,
            phase.regime_for(spec, index), mode, factory, True)
        row, _, _, _ = phase.build_record(row, trace, initial, index, name, spec['methods'][name], spec)
        row.update(phase.trace_metadata(trace))
    else:
        _, scenario, _, seed, _, name = case
        row, trace, initial = run_episode(env, seed, scenario, spec['mode'], factory, True)
        row = transfer.record(row, trace, initial, case, spec)
    if identity(study, row) != tuple(case):
        raise ValueError('Recovery executed a different scheduled case')
    return row, trace


def verify_execution(study, row, trace, spec):
    method = spec['methods'][row['controller']]
    if study == 'transfer':
        transfer.verify_record(trace, row, method)
        return
    rebuilt, adopted, checks, failures = phase.build_record(dict(row), trace, np.asarray(row['initial']),
        row['initial_index'], row['controller'], method, spec)
    for key, value in rebuilt.items():
        np.testing.assert_equal(value, row[key])
    phase.verify_trace(trace, row, method, adopted, checks, failures)
    for item in trace:
        phase.phase_record_check(item, method)


def require_stage(stage):
    launch = read(EXECUTION / f'{stage}_launch.json')
    exits = read(EXECUTION / f'{stage}_exits.json')['workers']
    phase.validate_worker_exits(exits, launch, len(launch['worker_pids']))
    return exits


def worker(stage, shard):
    if stage not in ('preflight', 'suffix') or shard not in range(3):
        raise ValueError('Unknown recovery worker')
    if stage == 'suffix':
        require_stage('preflight')
    plan, protocols = inputs()
    torch.set_num_threads(1)
    out = RECOVERY / stage / f'shard_{shard}'
    out.mkdir(parents=True, exist_ok=False)
    counts = {}
    for study, protocol in protocols.items():
        item, spec = plan['studies'][study], protocol['spec']
        cases = ([r['case'] for r in item['replay_cases'] if r['shard'] == shard] if stage == 'preflight'
                 else item['shards'][str(shard)]['missing_cases'])
        target = out / study
        target.mkdir()
        fs = MODULES[study].factories(protocol)
        warm = np.array([[-80., 12.], [-85., 13.], [-120., 11.]])
        for factory in fs.values():
            factory(warm).select(warm, np.zeros(3))
        with ExitStack() as stack:
            envs = ({'additive': stack.enter_context(DisturbedMergeEnv())} if study == 'phase' else
                    {p: stack.enter_context(MatchedShiftedMergeEnv(plant=p)) for p in spec['plants']})
            for number, case in enumerate(cases):
                name = case[-1]
                save(target / row_name(study, case).replace('row_', 'start_', 1),
                     dict(case=case, stage=stage, utc=utc(), pid=os.getpid()))
                row, trace = execute(study, case, spec, fs[name], envs['additive' if study == 'phase' else case[4]])
                # Persist the actual observation before checking it, including
                # any mismatch or failure. Never replace an original outcome.
                for filename, value in ((trace_name(study, case), trace), (row_name(study, case), row)):
                    path = target / filename
                    if path.exists():
                        raise FileExistsError('Never overwrite an executed recovery case')
                    atomic_bytes(path, (json.dumps(value, indent=2) + '\n').encode())
                verify_execution(study, row, trace, spec)
                evidence = dict(case=case, stage=stage, recovered_at=utc(), original_observation_replaced=False)
                if stage == 'preflight':
                    old_row, old_trace = original_case(study, case, plan)
                    compare_replay(old_row, row, old_trace, trace)
                    evidence['non_timing_fields_exact'] = True
                elif trace_name(study, case) in item['shards'][str(shard)]['orphan_traces']:
                    try:
                        _, old_trace = original_case(study, case, plan)
                    except json.JSONDecodeError:
                        evidence['orphan_trace_parseable'] = False
                    else:
                        compare_replay(None, row, old_trace, trace)
                        evidence.update(orphan_trace_parseable=True, orphan_non_timing_fields_exact=True)
                save(target / row_name(study, case).replace('row_', 'receipt_', 1), evidence)
                print(json.dumps(dict(stage=stage, shard=shard, study=study,
                    executed=number + 1, expected=len(cases), pid=os.getpid())), flush=True)
        counts[study] = len(cases)
    save(out / 'complete.json', dict(pid=os.getpid(), stage=stage, counts=counts,
        plan_sha256=digest(FROZEN / 'plan.json')))
    save(out / 'sha256.json', manifest(out))


def assembled_rows(study, plan, protocol):
    rows, sources = [], {}
    item, spec = plan['studies'][study], protocol['spec']
    for shard in range(3):
        original = ROOT / item['directory'] / f'shard_{shard}'
        prefix = read(original / 'episodes.json')
        resumed = RECOVERY / 'suffix' / f'shard_{shard}' / study
        extra = [read(resumed / row_name(study, case)) for case in item['shards'][str(shard)]['missing_cases']]
        combined = prefix + extra
        if [identity(study, r) for r in combined] != MODULES[study].schedule(spec, shard):
            raise ValueError('Recovered grid differs from the original full schedule')
        for index, row in enumerate(combined):
            key = identity(study, row)
            is_old = index < len(prefix)
            path = (original if is_old else resumed) / trace_name(study, key)
            sources[key] = dict(path=path.relative_to(ROOT).as_posix(), sha256=digest(path),
                                timing_epoch='original' if is_old else 'recovery')
        rows.extend(combined)
    return rows, sources


def finalize(study):
    require_stage('preflight')
    require_stage('suffix')
    plan, protocols = inputs(check_original=True)
    for stage in ('preflight', 'suffix'):
        for shard in range(3):
            directory = RECOVERY / stage / f'shard_{shard}'
            verify_manifest(directory)
            receipt = read(directory / 'complete.json')
            if receipt['plan_sha256'] != digest(FROZEN / 'plan.json') or receipt['stage'] != stage:
                raise ValueError('Recovery completion belongs to a different plan')
            for name, item in plan['studies'].items():
                expected = (sum(r['shard'] == shard for r in item['replay_cases']) if stage == 'preflight'
                            else len(item['shards'][str(shard)]['missing_cases']))
                if receipt['counts'][name] != expected:
                    raise ValueError('Incomplete recovery stage')
                planned = ([r['case'] for r in item['replay_cases'] if r['shard'] == shard] if stage == 'preflight'
                           else item['shards'][str(shard)]['missing_cases'])
                target = directory / name
                if {p.name for p in target.glob('row_*.json')} != {row_name(name, c) for c in planned}:
                    raise ValueError('Recovery files differ from the exact execution schedule')
                for case in planned:
                    audit = read(target / row_name(name, case).replace('row_', 'receipt_', 1))
                    if audit['case'] != case or audit['stage'] != stage:
                        raise ValueError('Execution receipt identity differs')
                    if stage == 'preflight':
                        old_row, old_trace = original_case(name, case, plan)
                        compare_replay(old_row, read(target / row_name(name, case)), old_trace,
                                       read(target / trace_name(name, case)))
    protocol = protocols[study]
    rows, provenance = assembled_rows(study, plan, protocol)
    if len(rows) != protocol['spec']['expected_actual_sumo_runs']:
        raise ValueError('Incomplete full study')
    result = RECOVERY / 'results' / study
    result.mkdir(parents=True, exist_ok=False)
    if study == 'phase':
        # The frozen phase reviewer reads STUDY/shard_*/trace_*.json. Give it a
        # separate immutable-file view, without modifying original observations.
        view = ROOT / 'runs/v21_interrupted_recovery_review'
        view.mkdir(parents=True, exist_ok=False)
        for key, source in provenance.items():
            path = view / f'shard_{key[0] % 3}' / trace_name(study, key)
            path.parent.mkdir(exist_ok=True)
            os.link(ROOT / source['path'], path)
        old_study = phase.STUDY
        try:
            phase.STUDY = view
            reviewed = phase.independent_review(rows, protocol)
        finally:
            phase.STUDY = old_study
        paired = phase.paired_checks(rows, protocol['spec'])
        results = phase.summarize(rows, protocol['spec'])
        recounted = {name: phase.outcome_counts(rows, protocol['spec'], name) for name in protocol['spec']['methods']}
        for name, counts in recounted.items():
            for key, value in counts.items():
                np.testing.assert_allclose(value, results[name][key], atol=1e-12, rtol=1e-12)
        summary = dict(independent_initials=protocol['spec']['n_initials'], results=results,
                       paired_results=paired, independently_recounted_outcomes=recounted)
    else:
        reviewed = transfer.independent_review(rows, protocol,
            lambda row: read(ROOT / provenance[identity(study, row)]['path']))
        summary = dict(independent_initials=len(protocol['spec']['scenarios']) * protocol['spec']['n_initials_per_scenario'],
                       **transfer.summarize(rows, protocol['spec']))
    for source in provenance.values():
        if digest(ROOT / source['path']) != source['sha256']:
            raise ValueError('Trace changed during complete independent review')
    rows.sort(key=lambda row: identity(study, row))
    save(result / 'episodes.json', rows)
    save(result / 'provenance.json', [dict(case=list(key), **entry) for key, entry in provenance.items()])
    save(result / 'summary.json', dict(**summary, actual_sumo_runs=len(rows), independent_review_counts=reviewed,
        protocol_sha256=plan['studies'][study]['protocol_sha256'], episodes_sha256=digest(result / 'episodes.json'),
        recovery_plan_sha256=digest(FROZEN / 'plan.json'), original_exit_status='unknown',
        recovery_execution_verified=True, risk_certified=False, hard_real_time_certified=False,
        original_rows_retained=sum(s['prefix_count'] for s in plan['studies'][study]['shards'].values()),
        resumed_rows=sum(len(s['missing_cases']) for s in plan['studies'][study]['shards'].values())))
    save(result / 'sha256.json', manifest(result))
    print(json.dumps(dict(study=study, completed=True, actual_sumo_runs=len(rows))), flush=True)


def launch_stage(stage, commands):
    processes, handles = [], []
    try:
        for i, command in enumerate(commands):
            stdout = (EXECUTION / f'{stage}_{i}.stdout.log').open('x', encoding='utf-8')
            stderr = (EXECUTION / f'{stage}_{i}.stderr.log').open('x', encoding='utf-8')
            handles.extend((stdout, stderr))
            processes.append(subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)))
        save(EXECUTION / f'{stage}_launch.json', dict(utc=utc(), stage=stage, supervisor_pid=os.getpid(),
            worker_pids=[p.pid for p in processes], commands=commands, recovery_plan_sha256=digest(FROZEN / 'plan.json')))
        while True:
            codes = [p.poll() for p in processes]
            atomic_bytes(EXECUTION / 'state.json', (json.dumps(dict(utc=utc(), stage=stage,
                worker_pids=[p.pid for p in processes], exit_codes=codes)) + '\n').encode())
            if all(code is not None for code in codes):
                break
            time.sleep(10)
        save(EXECUTION / f'{stage}_exits.json', dict(utc=utc(), workers=[dict(pid=p.pid, exit_code=p.returncode) for p in processes]))
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError('Recovery stage failed; retain all evidence, do not advance')
    finally:
        for handle in handles:
            handle.close()


def run():
    snapshot = native_snapshot()
    assert_no_existing_workers(snapshot)
    inputs(check_original=True)
    EXECUTION.mkdir(parents=True, exist_ok=False)
    save(EXECUTION / 'native_before_launch.json', dict(utc=utc(), processes=snapshot))
    base = [sys.executable, '-X', 'utf8', 'tools/recover_interrupted_v21.py']
    try:
        for stage in ('preflight', 'suffix'):
            launch_stage(stage, [base + ['worker', stage, str(shard)] for shard in range(3)])
        launch_stage('review', [base + ['finalize', study] for study in STUDIES])
        save(EXECUTION / 'completed.json', dict(utc=utc(), original_exit_status='unknown',
            recovered_summary_sha256={study: digest(RECOVERY / 'results' / study / 'summary.json') for study in STUDIES}))
    except Exception as error:
        save(EXECUTION / 'failure.json', dict(utc=utc(), type=type(error).__name__, error=str(error)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('freeze')
    sub.add_parser('run')
    work = sub.add_parser('worker')
    work.add_argument('stage', choices=['preflight', 'suffix'])
    work.add_argument('shard', type=int)
    finish = sub.add_parser('finalize')
    finish.add_argument('study', choices=list(STUDIES))
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze()
    elif args.command == 'run':
        run()
    elif args.command == 'worker':
        worker(args.stage, args.shard)
    else:
        finalize(args.study)
