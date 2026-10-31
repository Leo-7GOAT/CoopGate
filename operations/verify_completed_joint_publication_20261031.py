"""Verify advertised joint-study delivery bytes without changing the live queue.

The expected commit is explicit because the live computation branch may advance
independently. This verifies delivery, not a new statistical acceptance decision.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'artifacts/joint_completed_publication_20261031.json'
SOURCE = ROOT / 'artifacts/joint_risk_recovery_20261031'
DELIVERY = ROOT / 'artifacts/joint_completed_delivery_20261031'
PREFIXES = ('README.md', 'operations/archive_completed_joint_20261031.py',
            'tests/test_completed_joint_delivery.py',
            'operations/verify_completed_joint_publication_20261031.py',
            'artifacts/joint_risk_recovery_20261031/analysis',
            'artifacts/joint_completed_delivery_20261031')


def git(*args):
    return subprocess.check_output(
        ['git', '-c', 'maintenance.auto=false', '-c', 'gc.auto=0',
         '-c', 'http.proxy=http://127.0.0.1:7890',
         '-c', 'http.version=HTTP/1.1', *args], cwd=ROOT).decode().strip()


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main(target):
    if OUTPUT.exists():
        raise FileExistsError('Preserve the first delivery proof')
    target = git('rev-parse', target + '^{commit}')
    if git('ls-remote', '--refs', 'origin', 'refs/heads/main').split()[0] != target:
        raise ValueError('Expected commit is not advertised remote main')
    entries = []
    for line in git('ls-tree', '-r', '-l', target, '--', *PREFIXES).splitlines():
        metadata, name = line.split('\t', 1)
        mode, kind, blob, size = metadata.split()
        path = ROOT / name
        if (mode != '100644' or kind != 'blob' or not path.is_file()
                or path.stat().st_size != int(size)
                or git('hash-object', '--no-filters', '--', name) != blob):
            raise ValueError('Advertised raw bytes differ: ' + name)
        entries.append({'path': name, 'git_blob': blob, 'bytes': int(size)})
    selected = {item['path'] for item in entries}
    for name in ('certificate.json', 'verification.json', 'replay.json',
                 'native_finalizer_start.json', 'native_finalizer_exit.json',
                 'completion_receipt_20261031.json'):
        required = (SOURCE / 'analysis' / name).relative_to(ROOT).as_posix()
        if required not in selected:
            raise ValueError('Missing advertised completion evidence: ' + name)
    inventory = read(DELIVERY / 'inventory.json')
    if (inventory['unique_runs'] != 16384 or inventory['independent_initials'] != 4096
            or inventory['finalizer_exit_code'] != 0
            or inventory['scientific_success_assumed']):
        raise ValueError('Incorrect delivery scope')
    restored = {}
    for part in inventory['archive_parts']:
        archive_path = DELIVERY / part['name']
        relative = archive_path.relative_to(ROOT).as_posix()
        if (relative not in selected or digest(archive_path) != part['sha256']
                or archive_path.stat().st_size != part['bytes']):
            raise ValueError('Archive differs from its inventory')
        with zipfile.ZipFile(archive_path) as archive:
            if archive.namelist() != part['files']:
                raise ValueError('Archive member set or order differs')
            for name in archive.namelist():
                if PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts:
                    raise ValueError('Invalid archive member path')
                if name in restored:
                    raise ValueError('Duplicate archive member')
                with archive.open(name) as stream:
                    restored[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    if restored != inventory['original_files_sha256']:
        raise ValueError('Reconstructed source bytes differ')
    for name, expected in restored.items():
        if digest(SOURCE / name) != expected:
            raise ValueError('Original source bytes changed: ' + name)
    verification = read(SOURCE / 'analysis/verification.json')
    certificate = read(SOURCE / 'analysis/certificate.json')
    native_exit = read(SOURCE / 'analysis/native_finalizer_exit.json')
    if (verification['actual_sumo_runs'] != 16384
            or verification['independent_initials'] != 4096
            or digest(SOURCE / 'analysis/certificate.json') != verification['certificate_sha256']
            or digest(SOURCE / 'analysis/replay.json') != verification['replay_sha256']
            or native_exit['exit_code'] != 0 or not native_exit['retained_native_handle']
            or native_exit['start_sha256'] != digest(SOURCE / 'analysis/native_finalizer_start.json')):
        raise ValueError('Original finalization identity changed')
    result = {'utc': datetime.now(timezone.utc).isoformat(),
              'advertised_remote_main': target, 'local_live_head_at_review': git('rev-parse', 'HEAD'),
              'files': entries, 'verified_files': len(entries),
              'verified_git_file_bytes': sum(e['bytes'] for e in entries),
              'archive_parts': len(inventory['archive_parts']),
              'archive_bytes': sum(p['bytes'] for p in inventory['archive_parts']),
              'original_files_restored_and_verified': len(restored),
              'archive_inventory_sha256': digest(DELIVERY / 'inventory.json'),
              'certificate_sha256': verification['certificate_sha256'],
              'joint_deployment_certified_under_registered_assumptions': certificate['joint_deployment_certified'],
              'mean_gain_inference_normal_iid_exact_otherwise_asymptotic': True,
              'statewise_or_out_of_distribution_risk_certified': False,
              'new_numerical_inference': False, 'new_simulation_runs': 0,
              'raw_local_bytes_match_advertised_git_blobs': True,
              'every_archive_member_and_original_file_byte_verified': True}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('advertised_remote_main', 'verified_files',
                     'archive_parts', 'archive_bytes', 'original_files_restored_and_verified')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', required=True)
    main(parser.parse_args().target)
