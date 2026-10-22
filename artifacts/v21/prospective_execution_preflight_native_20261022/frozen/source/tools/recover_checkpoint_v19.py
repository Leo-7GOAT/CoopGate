"""Preserve a complete pending checkpoint after a Windows sharing violation.

Only checkpoint I/O is replaced. The frozen continuation worker, execution
order, boundary replay, simulator, policies and statistical rules are reused.
"""
import argparse
import errno
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import zipfile

import resume_frozen_risk_v19 as previous
from probe_disturbed_gate_v19 import digest, save
from start_final_v19 import require_committed, verify_manifest
from study_gate_v19 import read_protocol

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'artifacts/v19/risk_validation'
FIRST = ROOT / 'artifacts/v19/risk_recovery_20261019'
RECOVERY = ROOT / 'artifacts/v19/checkpoint_recovery_20261019'
LOGS = Path('C:/Users/Leo/AppData/Local/Temp/coopgate-v19-risk-recovery-20261019')
PENDING_NAME = 'episodes.json.xcnx2b6p.tmp'


def retry_replace(source, destination, attempts=41, delay=.25):
    """Retry only access/sharing failures, at most ten seconds by default."""
    if attempts < 1 or delay < 0:
        raise ValueError('Invalid retry budget')
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except OSError as error:
            transient = (getattr(error, 'winerror', None) in (5, 32, 33) or
                         isinstance(error, PermissionError) and error.errno in (errno.EACCES, errno.EPERM))
            if not transient or attempt + 1 == attempts:
                raise
            time.sleep(delay)


def atomic_bytes(path, raw):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='wb', dir=path.parent, prefix=path.name + '.', suffix='.tmp', delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    retry_replace(temporary, path)


def inspect_pending(old_raw, pending_raw, spec, shard):
    old, pending = json.loads(old_raw), json.loads(pending_raw)
    expected = previous.validate_prefix(old, spec, shard)
    previous.validate_prefix(pending, spec, shard)
    newline = '\r\n' if b'\r\n' in old_raw else '\n'
    if len(pending) != len(old) + 1 or previous.encoded_rows(pending[:-1], newline) != old_raw:
        raise ValueError('Pending checkpoint must preserve the exact old prefix and add one original scheduled case')
    if previous.encoded_rows(pending, newline) != pending_raw:
        raise ValueError('Pending checkpoint cannot be preserved byte-for-byte')
    return pending, expected, newline


def prepare():
    if RECOVERY.exists():
        raise FileExistsError('Preserve existing checkpoint recovery evidence')
    previous.read_recovery()
    _, protocol = read_protocol(STUDY)
    out = STUDY / 'shard_2'
    if (out / 'complete.json').exists():
        raise ValueError('Shard is already complete')
    old_raw, pending_raw = (out / 'episodes.json').read_bytes(), (out / PENDING_NAME).read_bytes()
    pending, expected, newline = inspect_pending(old_raw, pending_raw, protocol['spec'], 2)
    required = {f"trace_{r['initial_index']}_{r['disturbance']}_{r['controller']}.json" for r in pending if r['trace_saved']}
    if {p.name for p in out.glob('trace_*.json')} != required:
        raise ValueError('Missing or orphan trace relative to the pending checkpoint')
    first_plan = json.loads((FIRST / 'plan.json').read_text())
    first_entry = first_plan['shards']['2']
    with zipfile.ZipFile(FIRST / 'interrupted_prefix.zip') as archive:
        if previous.encoded_rows(pending[:first_entry['prefix_count']], newline) != archive.read('shard_2/episodes.json'):
            raise ValueError('Original interrupted prefix changed')
    RECOVERY.mkdir()
    saved = {}
    with zipfile.ZipFile(RECOVERY / 'interrupted_prefix.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('shard_2/episodes.json', pending_raw)
        archive.writestr('before_promotion/episodes.json', old_raw)
        saved['shard_2/episodes.json'] = dict(sha256=digest(out / PENDING_NAME), bytes=len(pending_raw))
        for path in sorted(out.glob('trace_*.json')):
            relative = path.relative_to(STUDY).as_posix()
            raw = path.read_bytes()
            archive.writestr(relative, raw)
            saved[relative] = dict(sha256=digest(path), bytes=len(raw))
    with zipfile.ZipFile(RECOVERY / 'interrupted_prefix.zip') as archive:
        if archive.read('shard_2/episodes.json') != pending_raw or archive.read('before_promotion/episodes.json') != old_raw:
            raise ValueError('Snapshot verification failed')
        import hashlib
        for relative, record in saved.items():
            if hashlib.sha256(archive.read(relative)).hexdigest() != record['sha256']:
                raise ValueError('Snapshot trace differs: ' + relative)
    evidence = RECOVERY / 'interruption'
    evidence.mkdir()
    for name in ('worker_2.stdout.log', 'worker_2.stderr.log', 'supervisor.stderr.log'):
        shutil.copy2(LOGS / name, evidence / name)
    shutil.copy2(FIRST / 'workers/shard_2/status.json', evidence / 'old_worker_status.json')
    shutil.copy2(Path('C:/Users/Leo/AppData/Local/Temp/coopgate-v19-validation-pipeline-20261019/state.json'), evidence / 'supervisor_state.json')
    snapshot = dict(files=saved, protocol_sha256=digest(STUDY / 'protocol.json'),
                    prefix_archive_sha256=digest(RECOVERY / 'interrupted_prefix.zip'),
                    original_destination_sha256=digest(out / 'episodes.json'), pending_sha256=digest(out / PENDING_NAME),
                    partial_risk_statistics_computed=False, policy_or_threshold_changes=False)
    save(RECOVERY / 'snapshot.json', snapshot)
    sources = [Path(__file__).resolve(), ROOT / 'tools/resume_frozen_risk_v19.py', ROOT / 'tests/test_recover_checkpoint_v19.py']
    plan = dict(purpose='Lossless checkpoint promotion and bounded retry after Windows access denial',
                study='artifacts/v19/risk_validation', protocol_sha256=snapshot['protocol_sha256'],
                snapshot_sha256=digest(RECOVERY / 'snapshot.json'), prefix_archive_sha256=snapshot['prefix_archive_sha256'],
                shards={'2': dict(prefix_count=len(pending), remaining_count=len(expected)-len(pending),
                                  boundary_case=expected[len(pending)-1], first_missing_case=expected[len(pending)],
                                  newline=newline, prefix_sha256=snapshot['pending_sha256'])},
                failed_shard=2, failed_process_id=25620, failed_process_exit_code=1,
                destination_before_promotion_sha256=snapshot['original_destination_sha256'], pending_filename=PENDING_NAME,
                retained_records=len(pending), planned_remaining_records=len(expected)-len(pending), required_boundary_replays=1,
                previously_completed_suffix_runs=len(pending)-first_entry['prefix_count'],
                first_recovery_plan_sha256=digest(FIRST / 'plan.json'),
                policy_or_statistical_rules_changed=False, outcome_dependent_case_selection=False, partial_certification_performed=False,
                promotion='Preserve the completed pending row including its original timings; replay it before any new suffix case',
                runtime_overrides=['Recovery evidence directory', 'Atomic checkpoint replacement with bounded access/sharing retry'],
                retry_attempts=41, retry_delay_seconds=.25,
                source_sha256={p.relative_to(ROOT).as_posix(): digest(p) for p in sources})
    save(RECOVERY / 'plan.json', plan)
    for source in sources:
        destination = RECOVERY / 'source' / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    save(RECOVERY / 'sha256.json', {p.relative_to(RECOVERY).as_posix(): digest(p) for p in RECOVERY.rglob('*') if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps({k:plan[k] for k in ('retained_records','planned_remaining_records','previously_completed_suffix_runs')}), flush=True)


def inputs():
    verify_manifest(RECOVERY)
    require_committed([RECOVERY / name for name in ('plan.json','snapshot.json','sha256.json','interrupted_prefix.zip')])
    plan = json.loads((RECOVERY / 'plan.json').read_text())
    if digest(FIRST / 'plan.json') != plan['first_recovery_plan_sha256']:
        raise ValueError('Original recovery plan changed')
    for path, expected in plan['source_sha256'].items():
        if digest(ROOT / path) != expected or digest(RECOVERY / 'source' / path) != expected:
            raise ValueError('Checkpoint recovery source changed: ' + path)
    read_protocol(STUDY)
    return plan


def worker():
    plan = inputs()
    out = STUDY / 'shard_2'
    if (RECOVERY / 'promotion.json').exists() or (RECOVERY / 'workers').exists():
        raise FileExistsError('Review an existing promotion or worker rather than repeating it')
    if digest(out / 'episodes.json') != plan['destination_before_promotion_sha256'] or digest(out / plan['pending_filename']) != plan['shards']['2']['prefix_sha256']:
        raise ValueError('Stopped destination or pending checkpoint changed')
    retry_replace(out / plan['pending_filename'], out / 'episodes.json')
    save(RECOVERY / 'promotion.json', dict(worker_pid=os.getpid(), source_sha256=plan['shards']['2']['prefix_sha256'],
                                         destination_sha256=digest(out / 'episodes.json'), pending_row_preserved=True,
                                         plan_sha256=digest(RECOVERY / 'plan.json')))
    previous.RECOVERY = RECOVERY
    previous.atomic_bytes = atomic_bytes
    previous.worker(2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare','worker'])
    args = parser.parse_args()
    prepare() if args.command == 'prepare' else worker()
