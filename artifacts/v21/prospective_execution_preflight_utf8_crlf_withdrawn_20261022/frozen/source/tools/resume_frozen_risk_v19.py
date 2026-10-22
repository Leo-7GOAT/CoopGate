"""Resume terminated V19 workers without changing any frozen experimental rule.

The interrupted prefix is snapshotted before preparation. Each shard replays its
last saved case, retains every old record and trace, then executes the exact
missing suffix of the original schedule. No partial risk estimate is computed.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np
import torch

from probe_disturbed_gate_v19 import digest, save
from start_final_v19 import require_committed, verify_manifest
from study_gate_v19 import DisturbedMergeEnv, factories, read_protocol, regime_for, run_episode, trace_audit

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'artifacts/v19/risk_validation'
RECOVERY = ROOT / 'artifacts/v19/risk_recovery_20261019'
WALL_FIELDS = {'setup_ms', 'controller_ms', 'admission_ms'}


def schedule(spec, shard):
    if not 0 <= shard < spec['shards']:
        raise ValueError('Unknown shard')
    result = []
    for index in range(shard, spec['n_initials'], spec['shards']):
        seed = spec['seed_base'] + index
        for mi, mode in enumerate(spec['modes']):
            for method in np.random.default_rng(seed + mi + 1919).permutation(list(spec['methods'])):
                result.append((index, mode, str(method)))
    return result


def validate_prefix(rows, spec, shard):
    expected = schedule(spec, shard)
    identities = [(r['initial_index'], r['disturbance'], r['controller']) for r in rows]
    if len(rows) > len(expected) or identities != expected[:len(rows)]:
        raise ValueError('Records are not the exact original execution prefix; do not filter or reorder them')
    for row in rows:
        if row['seed'] != spec['seed_base'] + row['initial_index'] or row['scenario'] != regime_for(spec, row['initial_index']):
            raise ValueError('Initial seed or regime differs from the original frozen schedule')
    return expected


def encoded_rows(rows, newline):
    if newline not in ('\n', '\r\n'):
        raise ValueError('Unknown archived newline convention')
    return (json.dumps(rows, indent=2) + '\n').replace('\n', newline).encode('utf-8')


def atomic_bytes(path, raw):
    """An interrupted write cannot truncate the previously saved checkpoint."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='wb', dir=path.parent, prefix=path.name + '.', suffix='.tmp', delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_record(row, trace, initial, index, method, method_spec, spec):
    # These fields and definitions are the original study_gate_v19.worker
    # accounting. The policy factory and run_episode are imported unchanged.
    cost = np.array(row['per_vehicle_finish']) + .05 * np.array(row['per_vehicle_variation'])
    used = sum(t.get('nominal_joint_action_admitted', method_spec['kind'] == 'raw') for t in trace)
    adopted = [t for t in trace if t.get('portfolio_joint_action_admitted', False)]
    banks_checked = [len(t['selected_bank_differences']) for t in adopted]
    failures = [int(np.any(np.array(t['selected_bank_differences']) > 1e-10, axis=(1, 2)).sum()) for t in adopted]
    row.update(controller=str(method), initial=initial.tolist(), initial_index=index,
               per_vehicle_cost=cost.tolist(), team_cost=float(cost.mean()),
               learned_action_steps=used, control_steps=len(trace),
               min_adopted_bank_checks=min(banks_checked, default=None),
               max_adopted_failed_scenarios=max(failures, default=None),
               chosen_compositions=[int(t.get('composition', 0)) for t in trace],
               forecast_transitions=sum(t.get('performance_forecast_transitions', 0) for t in trace),
               internal_transitions=sum(t.get('performance_model_transitions', 0) for t in trace))
    failure = not row['completed'] or row['collision'] or row['aborted'] or row['contract_violations']
    row['trace_saved'] = bool(spec['all_traces_saved'] or index in spec['audit_initial_indices'] or failure)
    return row, adopted, banks_checked, failures


def check_episode(trace, row, method_spec, adopted, checks, failures):
    trace_audit(trace, row)
    if adopted:
        if min(checks) != 17 or max(failures) > method_spec['max_failed_scenarios']:
            raise ValueError('Original disturbance-bank admission invariant failed')


def compare_boundary(actual, original):
    if set(actual) != set(original):
        raise ValueError('Boundary replay schema differs from the original executor')
    for key in original:
        if key not in WALL_FIELDS:
            np.testing.assert_equal(actual[key], original[key], err_msg='Boundary replay: ' + key)


def prepare():
    if (RECOVERY / 'plan.json').exists() or (STUDY / 'summary.json').exists():
        raise FileExistsError('Preserve the existing recovery/finalization record')
    _, protocol = read_protocol(STUDY)
    spec = protocol['spec']
    if spec.get('data_role') != 'risk_validation':
        raise ValueError('This recovery is only for the interrupted risk split')
    snapshot = json.loads((RECOVERY / 'snapshot.json').read_text())
    if digest(STUDY / 'protocol.json') != snapshot['protocol_sha256'] or digest(RECOVERY / 'interrupted_prefix.zip') != snapshot['prefix_archive_sha256']:
        raise ValueError('Interrupted snapshot identity changed')
    shards = {}
    with zipfile.ZipFile(RECOVERY / 'interrupted_prefix.zip') as archive:
        for shard in range(spec['shards']):
            out = STUDY / f'shard_{shard}'
            if (out / 'complete.json').exists():
                raise ValueError('A completed shard requires review, not this interrupted-prefix recovery')
            raw = (out / 'episodes.json').read_bytes()
            if raw != archive.read(f'shard_{shard}/episodes.json'):
                raise ValueError('Live prefix changed after the snapshot')
            rows = json.loads(raw)
            expected = validate_prefix(rows, spec, shard)
            if not rows:
                raise ValueError('A saved boundary case is required')
            newline = '\r\n' if b'\r\n' in raw else '\n'
            if encoded_rows(rows, newline) != raw:
                raise ValueError('Prefix cannot be round-tripped byte-for-byte')
            needed = {f"trace_{r['initial_index']}_{r['disturbance']}_{r['controller']}.json" for r in rows if r['trace_saved']}
            if {p.name for p in out.glob('trace_*.json')} != needed:
                raise ValueError('Missing or orphan traces require separate recovery review')
            for key, record in snapshot['files'].items():
                if key.startswith(f'shard_{shard}/') and digest(STUDY / key) != record['sha256']:
                    raise ValueError('A saved prefix file changed: ' + key)
            shards[str(shard)] = dict(prefix_count=len(rows), remaining_count=len(expected) - len(rows),
                                      boundary_case=expected[len(rows) - 1], first_missing_case=expected[len(rows)] if len(rows) < len(expected) else None,
                                      newline=newline, prefix_sha256=digest(out / 'episodes.json'))
    sources = [Path(__file__).resolve(), ROOT / 'tests/test_resume_frozen_risk_v19.py']
    plan = dict(purpose='Infrastructure recovery of the exact frozen calibration suffix',
                study='artifacts/v19/risk_validation', protocol_sha256=digest(STUDY / 'protocol.json'),
                snapshot_sha256=digest(RECOVERY / 'snapshot.json'), prefix_archive_sha256=snapshot['prefix_archive_sha256'],
                shards=shards, retained_records=sum(r['prefix_count'] for r in shards.values()),
                planned_remaining_records=sum(r['remaining_count'] for r in shards.values()),
                required_boundary_replays=len(shards), policy_or_statistical_rules_changed=False,
                outcome_dependent_case_selection=False, partial_certification_performed=False,
                boundary_result_use='Execution identity only; no thresholds, models, candidates or sample sizes are selected from partial outcomes',
                timing_scope='Original prefix timing records are retained. Resumed records are measured after restart; collection spans two sessions.',
                interruption_scope='Uncommitted in-flight episode work before interruption is unknown and is not claimed as a new independent sample.',
                source_sha256={p.relative_to(ROOT).as_posix(): digest(p) for p in sources})
    save(RECOVERY / 'plan.json', plan)
    for source in sources:
        destination = RECOVERY / 'source' / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    save(RECOVERY / 'sha256.json', {p.relative_to(RECOVERY).as_posix(): digest(p) for p in RECOVERY.rglob('*')
                                   if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps({key: plan[key] for key in ('retained_records', 'planned_remaining_records', 'required_boundary_replays')}), flush=True)


def read_recovery():
    verify_manifest(RECOVERY)
    plan = json.loads((RECOVERY / 'plan.json').read_text())
    require_committed([RECOVERY / 'plan.json', RECOVERY / 'snapshot.json', RECOVERY / 'interrupted_prefix.zip', RECOVERY / 'sha256.json'])
    for path, expected in plan['source_sha256'].items():
        if digest(ROOT / path) != digest(RECOVERY / 'source' / path) or digest(ROOT / path) != expected:
            raise ValueError('Recovery source changed after preparation: ' + path)
    _, protocol = read_protocol(STUDY)
    if digest(STUDY / 'protocol.json') != plan['protocol_sha256'] or digest(RECOVERY / 'snapshot.json') != plan['snapshot_sha256']:
        raise ValueError('Frozen recovery input changed')
    return plan, protocol


def worker(shard):
    torch.set_num_threads(1)
    plan, protocol = read_recovery()
    spec, entry = protocol['spec'], plan['shards'][str(shard)]
    out, audit = STUDY / f'shard_{shard}', RECOVERY / 'workers' / f'shard_{shard}'
    if (out / 'complete.json').exists() or audit.exists():
        raise FileExistsError('Do not rerun or overwrite an existing recovery worker')
    if digest(out / 'episodes.json') != entry['prefix_sha256']:
        raise ValueError('Prefix changed before recovery started')
    rows = json.loads((out / 'episodes.json').read_text())
    expected = validate_prefix(rows, spec, shard)
    if len(rows) != entry['prefix_count']:
        raise ValueError('Recovery prefix length changed')
    audit.mkdir(parents=True)
    progress = dict(shard=shard, pid=os.getpid(), prefix_records=len(rows), new_sumo_runs=0, boundary_replays=0)
    save(audit / 'status.json', progress | dict(phase='starting'))
    print(json.dumps(progress | dict(phase='starting')), flush=True)
    fs = factories(protocol)
    warm = np.array([[-80., 12.], [-85., 13.], [-120., 11.]])
    for factory in fs.values():
        factory(warm).select(warm, np.zeros(3))

    def execute(env, case):
        index, mode, method = case
        row, trace, initial = run_episode(env, spec['seed_base'] + index, regime_for(spec, index), mode, fs[method], True)
        row, adopted, checks, failures = build_record(row, trace, initial, index, method, spec['methods'][method], spec)
        return row, trace, adopted, checks, failures

    with DisturbedMergeEnv() as env:
        boundary = expected[entry['prefix_count'] - 1]
        row, trace, adopted, checks, failures = execute(env, boundary)
        # Preserve this replay even if its subsequent identity check fails.
        save(audit / 'boundary_row.json', row)
        save(audit / 'boundary_trace.json', trace)
        compare_boundary(row, rows[-1])
        check_episode(trace, row, spec['methods'][boundary[2]], adopted, checks, failures)
        old_trace = out / f'trace_{boundary[0]}_{boundary[1]}_{boundary[2]}.json'
        if old_trace.exists():
            original_trace = json.loads(old_trace.read_text())
            if len(trace) != len(original_trace):
                raise ValueError('Boundary trace length changed')
            for new, old in zip(trace, original_trace):
                for field in ('command', 'x'):
                    np.testing.assert_array_equal(new[field], old[field])
        progress['boundary_replays'] = 1
        save(audit / 'boundary_audit.json', dict(all_nontiming_fields_equal=True, new_trace_physically_reconstructed=True,
                                               archived_command_state_trace_compared=old_trace.exists(), case=boundary,
                                               independent_risk_samples_added=0))
        save(audit / 'status.json', progress | dict(phase='continuing_missing_suffix', total_records=len(rows)))
        for case in expected[entry['prefix_count']:]:
            index, mode, method = case
            row, trace, adopted, checks, failures = execute(env, case)
            if row['trace_saved']:
                trace_path = out / f'trace_{index}_{mode}_{method}.json'
                if trace_path.exists():
                    raise FileExistsError('Never replace an archived trajectory')
                save(trace_path, trace)
            rows.append(row)
            atomic_bytes(out / 'episodes.json', encoded_rows(rows, entry['newline']))
            # Same validation and append order as the original worker, with an
            # atomic checkpoint replacing its non-atomic whole-file write.
            check_episode(trace, row, spec['methods'][method], adopted, checks, failures)
            progress['new_sumo_runs'] += 1
            if progress['new_sumo_runs'] % 50 == 0:
                save(audit / 'status.json', progress | dict(phase='continuing_missing_suffix', total_records=len(rows)))
                print(json.dumps(progress | dict(total_records=len(rows))), flush=True)
    if len(rows) != len(expected) or progress['new_sumo_runs'] != entry['remaining_count']:
        raise ValueError('The entire original missing suffix has not completed')
    validate_prefix(rows, spec, shard)
    with zipfile.ZipFile(RECOVERY / 'interrupted_prefix.zip') as archive:
        if encoded_rows(rows[:entry['prefix_count']], entry['newline']) != archive.read(f'shard_{shard}/episodes.json'):
            raise ValueError('Original records changed during recovery')
    snapshot = json.loads((RECOVERY / 'snapshot.json').read_text())
    for path, record in snapshot['files'].items():
        if path.startswith(f'shard_{shard}/') and not path.endswith('/episodes.json') and digest(STUDY / path) != record['sha256']:
            raise ValueError('Original trajectory changed during recovery: ' + path)
    completion = dict(initial_indices=list(range(shard, spec['n_initials'], spec['shards'])),
                      actual_sumo_runs=len(rows), episodes_sha256=digest(out / 'episodes.json'), protocol_sha256=plan['protocol_sha256'])
    save(audit / 'complete.json', progress | dict(original_prefix_bytes_preserved=True, original_trace_files_preserved=True,
                                                final_episodes_sha256=completion['episodes_sha256'], plan_sha256=digest(RECOVERY / 'plan.json')))
    save(out / 'complete.json', completion)
    save(audit / 'status.json', progress | dict(phase='completed', total_records=len(rows)))
    print(json.dumps(progress | dict(phase='completed', total_records=len(rows))), flush=True)


def verify_completed():
    plan, protocol = read_recovery()
    snapshot = json.loads((RECOVERY / 'snapshot.json').read_text())
    results = []
    with zipfile.ZipFile(RECOVERY / 'interrupted_prefix.zip') as archive:
        for shard in range(protocol['spec']['shards']):
            entry = plan['shards'][str(shard)]
            audit, out = RECOVERY / 'workers' / f'shard_{shard}', STUDY / f'shard_{shard}'
            done = json.loads((audit / 'complete.json').read_text())
            boundary = json.loads((audit / 'boundary_audit.json').read_text())
            original_completion = json.loads((out / 'complete.json').read_text())
            rows = json.loads((out / 'episodes.json').read_text())
            validate_prefix(rows, protocol['spec'], shard)
            if len(rows) != len(schedule(protocol['spec'], shard)) or encoded_rows(rows[:entry['prefix_count']], entry['newline']) != archive.read(f'shard_{shard}/episodes.json'):
                raise ValueError('Final case set or retained prefix differs')
            if digest(out / 'episodes.json') != done['final_episodes_sha256'] or done['new_sumo_runs'] != entry['remaining_count'] or done['boundary_replays'] != 1:
                raise ValueError('Recovery execution accounting differs')
            if (done['plan_sha256'] != digest(RECOVERY / 'plan.json') or
                    done['prefix_records'] != entry['prefix_count'] or
                    original_completion['actual_sumo_runs'] != len(rows) or
                    original_completion['episodes_sha256'] != done['final_episodes_sha256'] or
                    original_completion['protocol_sha256'] != plan['protocol_sha256'] or
                    original_completion['initial_indices'] != list(range(shard, protocol['spec']['n_initials'], protocol['spec']['shards']))):
                raise ValueError('Original and recovery completion identities differ')
            if not boundary['all_nontiming_fields_equal'] or not boundary['new_trace_physically_reconstructed']:
                raise ValueError('Boundary execution identity is unverified')
            if boundary['case'] != entry['boundary_case'] or boundary['independent_risk_samples_added'] != 0:
                raise ValueError('Boundary replay must remain the declared non-independent identity check')
            compare_boundary(json.loads((audit / 'boundary_row.json').read_text()), rows[entry['prefix_count'] - 1])
            for path, record in snapshot['files'].items():
                if path.startswith(f'shard_{shard}/') and not path.endswith('/episodes.json') and digest(STUDY / path) != record['sha256']:
                    raise ValueError('An original trace changed after recovery completion: ' + path)
            results.append(done)
    destination = ROOT / 'artifacts/v19/risk_recovery_audit_20261019'
    destination.mkdir(exist_ok=False)
    report = dict(purpose='Completed interrupted-calibration recovery accounting', retained_records=plan['retained_records'],
                  resumed_unique_sumo_runs=sum(r['new_sumo_runs'] for r in results),
                  additional_boundary_replay_sumo_runs=sum(r['boundary_replays'] for r in results),
                  original_prefixes_preserved=True, original_traces_preserved=True, policy_or_statistical_rules_changed=False,
                  partial_certification_performed=False, risk_certified_by_this_audit=False,
                  interruption_scope=plan['interruption_scope'], timing_scope=plan['timing_scope'],
                  plan_sha256=digest(RECOVERY / 'plan.json'), shards=results,
                  recovery_evidence_sha256={p.relative_to(RECOVERY).as_posix(): digest(p) for p in sorted((RECOVERY / 'workers').rglob('*')) if p.is_file()})
    save(destination / 'audit.json', report)
    save(destination / 'sha256.json', {'audit.json': digest(destination / 'audit.json')})
    print(json.dumps({k: report[k] for k in ('retained_records', 'resumed_unique_sumo_runs', 'additional_boundary_replay_sumo_runs', 'original_prefixes_preserved')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    p = sub.add_parser('worker')
    p.add_argument('shard', type=int)
    sub.add_parser('verify')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'worker':
        worker(args.shard)
    else:
        verify_completed()
