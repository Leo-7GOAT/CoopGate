"""Audit both infrastructure recoveries after all native workers have exited."""
import argparse
import json
from pathlib import Path
import shutil
import zipfile

import resume_frozen_risk_v19 as first
import recover_checkpoint_v19 as second
from probe_disturbed_gate_v19 import digest, save
from start_final_v19 import verify_manifest
from study_gate_v19 import read_protocol
from verify_development_v14 import trace_audit

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'artifacts/v19/risk_validation'
FIRST = ROOT / 'artifacts/v19/risk_recovery_20261019'
SECOND = ROOT / 'artifacts/v19/checkpoint_recovery_20261019'
LOGS = Path('C:/Users/Leo/AppData/Local/Temp/coopgate-v19-checkpoint-recovery-20261019')
OUT = ROOT / 'artifacts/v19/risk_recovery_audit_20261019'


def validate_exit(receipt, completion, shard):
    if receipt['shard'] != shard or receipt['exit_code'] != 0 or not receipt['complete_record_exists']:
        raise ValueError('A native worker did not exit normally with its completion record')
    if receipt['complete_record_sha256'] != digest(completion):
        raise ValueError('Native exit evidence binds a different completion record')


def run():
    if OUT.exists():
        raise FileExistsError('Preserve an existing recovery-chain audit')
    exits = []
    # Complete native exit receipts are required before opening any large study
    # record, avoiding the read/rename conflict during live Windows writes.
    for shard in range(4):
        path = LOGS / ('replacement_worker_2_exit.json' if shard == 2 else f'retained_worker_{shard}_exit.json')
        receipt = json.loads(path.read_text())
        validate_exit(receipt, STUDY / f'shard_{shard}/complete.json', shard)
        exits.append((path, receipt))
    first_plan, protocol = first.read_recovery()
    second_plan = second.inputs()
    read_protocol(STUDY)
    for directory in (FIRST, SECOND):
        verify_manifest(directory)
    initial_snapshot = json.loads((FIRST / 'snapshot.json').read_text())
    second_snapshot = json.loads((SECOND / 'snapshot.json').read_text())
    reports, boundaries = [], []
    with zipfile.ZipFile(FIRST / 'interrupted_prefix.zip') as original, zipfile.ZipFile(SECOND / 'interrupted_prefix.zip') as pending:
        for shard in range(4):
            out = STUDY / f'shard_{shard}'
            if list(out.glob('*.tmp')):
                raise ValueError('A pending checkpoint remains; review before certification')
            rows = json.loads((out / 'episodes.json').read_bytes())
            expected = first.validate_prefix(rows, protocol['spec'], shard)
            if len(rows) != len(expected):
                raise ValueError('Original case schedule is incomplete')
            entry = first_plan['shards'][str(shard)]
            if first.encoded_rows(rows[:entry['prefix_count']], entry['newline']) != original.read(f'shard_{shard}/episodes.json'):
                raise ValueError('An initial interrupted prefix changed')
            completion = json.loads((out / 'complete.json').read_text())
            if (completion['episodes_sha256'] != digest(out / 'episodes.json') or
                    completion['protocol_sha256'] != digest(STUDY / 'protocol.json') or
                    completion['actual_sumo_runs'] != len(rows) or
                    completion['initial_indices'] != list(range(shard, protocol['spec']['n_initials'], 4))):
                raise ValueError('Original study completion identity differs')
            runtime = SECOND if shard == 2 else FIRST
            finished = json.loads((runtime / f'workers/shard_{shard}/complete.json').read_text())
            plan = second_plan if shard == 2 else first_plan
            current = plan['shards'][str(shard)]
            if (finished['new_sumo_runs'] != current['remaining_count'] or
                    finished['prefix_records'] != current['prefix_count'] or
                    finished['boundary_replays'] != 1 or
                    finished['final_episodes_sha256'] != completion['episodes_sha256'] or
                    finished['plan_sha256'] != digest(runtime / 'plan.json')):
                raise ValueError('Recovery worker accounting differs')
            boundary_checks = [(FIRST, entry)]
            if shard == 2:
                if first.encoded_rows(rows[:current['prefix_count']], current['newline']) != pending.read('shard_2/episodes.json'):
                    raise ValueError('The pending checkpoint was changed or discarded')
                if first.encoded_rows(rows[:current['prefix_count']-1], current['newline']) != pending.read('before_promotion/episodes.json'):
                    raise ValueError('The destination before promotion changed')
                if second_plan['previously_completed_suffix_runs'] != current['prefix_count'] - entry['prefix_count']:
                    raise ValueError('Pending-row accounting duplicates or drops a sample')
                boundary_checks.append((SECOND, current))
            for directory, declared in boundary_checks:
                audit = directory / f'workers/shard_{shard}'
                boundary = json.loads((audit / 'boundary_audit.json').read_text())
                row = json.loads((audit / 'boundary_row.json').read_text())
                if (boundary['case'] != declared['boundary_case'] or
                        boundary['independent_risk_samples_added'] != 0 or
                        not boundary['all_nontiming_fields_equal'] or not boundary['new_trace_physically_reconstructed']):
                    raise ValueError('A boundary replay was not verified as an identity check')
                first.compare_boundary(row, rows[declared['prefix_count']-1])
                trace_audit(json.loads((audit / 'boundary_trace.json').read_text()), row)
                boundaries.append(dict(shard=shard, directory=directory.relative_to(ROOT).as_posix(), case=boundary['case']))
            reports.append(dict(shard=shard, retained_before_first_recovery=entry['prefix_count'],
                                resumed_unique_runs=len(rows)-entry['prefix_count'], final_unique_runs=len(rows),
                                final_episodes_sha256=completion['episodes_sha256'], normal_native_exit=True))
    for snapshot in (initial_snapshot, second_snapshot):
        for path, record in snapshot['files'].items():
            if not path.endswith('/episodes.json') and digest(STUDY / path) != record['sha256']:
                raise ValueError('A retained trajectory changed: ' + path)
    total = sum(r['final_unique_runs'] for r in reports)
    resumed = sum(r['resumed_unique_runs'] for r in reports)
    if total != protocol['spec']['expected_actual_sumo_runs'] or resumed != first_plan['planned_remaining_records'] or len(boundaries) != 5:
        raise ValueError('Original study or replay accounting differs')
    report = dict(purpose='Lossless calibration recovery-chain verification', total_unique_study_runs=total,
                  retained_original_records=first_plan['retained_records'], resumed_unique_sumo_runs=resumed,
                  extra_boundary_replay_sumo_runs=len(boundaries), pending_completed_rows_preserved=1,
                  all_original_prefixes_and_traces_preserved=True, all_four_native_exits_normal=True,
                  failed_first_shard_2_exit_retained=True, policy_or_statistical_rules_changed=False,
                  partial_risk_statistics_computed=False, risk_certified_by_this_audit=False,
                  timing_scope='Original acquisition plus first recovery, with shard 2 additionally continued after checkpoint promotion; all original timing values retained.',
                  interruption_scope=first_plan['interruption_scope'], shards=reports, boundaries=boundaries,
                  protocol_sha256=digest(STUDY / 'protocol.json'),
                  first_plan_sha256=digest(FIRST / 'plan.json'), second_plan_sha256=digest(SECOND / 'plan.json'),
                  recovery_runtime_evidence_sha256={p.relative_to(ROOT).as_posix():digest(p)
                      for directory in (FIRST,SECOND) for p in sorted((directory / 'workers').rglob('*')) if p.is_file()},
                  promotion_sha256=digest(SECOND / 'promotion.json'),
                  source_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in [Path(__file__).resolve(), ROOT / 'tests/test_recovery_chain_v19.py']})
    OUT.mkdir()
    for path, _ in exits:
        shutil.copy2(path, OUT / path.name)
    for path in report['source_sha256']:
        destination = OUT / 'source' / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, destination)
    save(OUT / 'audit.json', report)
    save(OUT / 'sha256.json', {p.relative_to(OUT).as_posix():digest(p) for p in OUT.rglob('*') if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps({key:report[key] for key in ('total_unique_study_runs','retained_original_records','resumed_unique_sumo_runs','extra_boundary_replay_sumo_runs','all_original_prefixes_and_traces_preserved')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run()
