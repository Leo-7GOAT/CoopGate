"""Verify remote main identity and every delivered recovery blob without effects."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'artifacts/recovery_20261027'
OUTPUT = ROOT / PREFIX / 'transport_verification.json'
OPTIONS = ['-c', 'maintenance.auto=false', '-c', 'gc.auto=0',
           '-c', 'http.proxy=http://127.0.0.1:7890', '-c', 'http.version=HTTP/1.1']


def git(*args):
    return subprocess.check_output(['git', *OPTIONS, *args], cwd=ROOT)


def main():
    if OUTPUT.exists():
        raise FileExistsError('Preserve existing verification instead of overwriting')
    ref = git('rev-parse', 'origin/main').decode().strip()
    advertised = git('ls-remote', '--refs', 'origin', 'refs/heads/main').decode().split()
    if not advertised or advertised[0] != ref:
        raise ValueError('Fetched and advertised remote main differ')
    paths = set(git('ls-tree', '-rz', '--name-only', ref, '--', PREFIX).decode().strip('\0').split('\0'))
    plan = json.loads((ROOT/PREFIX/'calibration_recovery/recovery/plan.json').read_text(encoding='utf-8'))
    paths.update(plan['source_sha256'])
    paths.update(('tools/execute_joint_risk_recovered_20261027.py',
                  'operations/advance_mainline_20261027.ps1',
                  'operations/test_durable_mainline_20261027.ps1',
                  'operations/continue_mainline_20261027.ps1',
                  'operations/continue_mainline_20261027_v2.ps1',
                  'operations/verify_recovery_transport_20261027.py',
                  'docs/recovery_20261027.md'))
    if '' in paths or not paths:
        raise ValueError('Empty transport scope')
    result = {}
    total_bytes = 0
    with subprocess.Popen(['git', 'cat-file', '--batch'], cwd=ROOT,
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE) as process:
        for name in sorted(paths):
            path = ROOT/name
            process.stdin.write(f'{ref}:{name}\n'.encode())
            process.stdin.flush()
            fields = process.stdout.readline().decode().strip().split()
            if len(fields) != 3 or fields[1] != 'blob':
                raise ValueError(f'Delivered blob missing: {name}')
            remaining = size = int(fields[2])
            if path.stat().st_size != size:
                raise ValueError(f'Working file size differs: {name}')
            blob_hash, working_hash = hashlib.sha256(), hashlib.sha256()
            with path.open('rb') as local:
                while remaining:
                    data = process.stdout.read(min(1024*1024, remaining))
                    if not data:
                        raise ValueError('Truncated Git blob stream')
                    blob_hash.update(data)
                    working_hash.update(local.read(len(data)))
                    remaining -= len(data)
            if process.stdout.read(1) != b'\n' or blob_hash.digest() != working_hash.digest():
                raise ValueError(f'Working/Git bytes differ: {name}')
            result[name] = blob_hash.hexdigest()
            total_bytes += size
        process.stdin.close()
        if process.wait() != 0:
            raise ValueError('Git blob verification failed')
    value = dict(utc=datetime.now(timezone.utc).isoformat(), remote='origin',
                 remote_main_commit=ref, advertised_remote_ref_verified=True,
                 git_blob_bytes_equal_working_files=True, verified_files=len(result),
                 verified_bytes=total_bytes, file_sha256=result,
                 statistical_sample_size_unchanged=True, partial_effects_read=False,
                 scientific_acceptance_claim=False)
    with OUTPUT.open('x', encoding='utf-8') as output:
        json.dump(value, output, indent=2)
        output.write('\n')
    print(json.dumps({k:value[k] for k in ('remote_main_commit','verified_files',
                                         'verified_bytes','scientific_acceptance_claim')}))


if __name__ == '__main__':
    main()
