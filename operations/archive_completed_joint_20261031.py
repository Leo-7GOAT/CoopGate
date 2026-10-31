"""Deliver full joint-study bytes only after the unchanged finalizer exits zero.

This is a delivery check, not a second scientific acceptance test. Negative
certificates are archived unchanged. It never imports or launches the simulator.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/joint_risk_recovery_20261031'
TARGET = ROOT / 'artifacts/joint_completed_delivery_20261031'
LIMIT = 20 * 1024 * 1024
COUNTS = {'calibration': 3072, 'confirmation': 1024}
PINS = {
    'artifacts/v42/joint_risk/plan.json':
        'b1e2d54020c95fb6e15c2355040cf2d22631b5e853687be9a54afee58aa7fab6',
    'artifacts/v42/joint_risk/binding.json':
        '77a2265bcb638cd0e0d7c5f23ef8ca53db42ed83db64569a0b9b24e2dc11764d',
    'artifacts/joint_risk_recovery_20261031/recovery.json':
        '5890ea581054133a667b8910511eb4ec6118acafc3f43642b56b5cc7c01800e0',
}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def require_finalization():
    # All checks precede archive creation and any outcome-file access.
    for name, expected in PINS.items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen input changed: ' + name)
    journal = ROOT / '.git/mainline-joint-recovery-20261031.jsonl'
    expected_args = ['-X', 'utf8', 'tools/joint_recovery_20261031.py', 'finalize']
    exits = [event for line in journal.read_text(encoding='utf-8-sig').splitlines()
             if (event := json.loads(line)).get('kind') == 'command_exit'
             and event['detail'].get('arguments') == expected_args]
    if len(exits) != 1 or exits[0]['detail']['exit_code'] != 0:
        raise ValueError('Original full finalizer must have one actual zero exit')
    verification = read(SOURCE / 'analysis/verification.json')
    if (verification['actual_sumo_runs'] != 16384
            or verification['independent_initials'] != 4096
            or verification['plan_sha256'] != PINS['artifacts/v42/joint_risk/plan.json']
            or verification['binding_sha256'] != PINS['artifacts/v42/joint_risk/binding.json']):
        raise ValueError('Full fixed-study verification required')
    for name in ('certificate', 'replay'):
        if digest(SOURCE / f'analysis/{name}.json') != verification[f'{name}_sha256']:
            raise ValueError('Original finalization bytes changed: ' + name)
    for stage, count in COUNTS.items():
        execution = SOURCE / stage / 'execution'
        receipt = read(execution / 'exit.json')
        complete = read(execution / 'collection_complete.json')
        manifest = read(execution / 'blocks_manifest.json')
        if (receipt['exit_code'] != 0
                or receipt['native_pid'] != complete['native_pid']
                or verification['stage_exit_sha256'][stage] != digest(execution / 'exit.json')
                or complete['clusters'] != count
                or complete['actual_sumo_runs'] != 4 * count
                or manifest['actual_sumo_runs'] != 4 * count
                or [entry['index'] for entry in manifest['entries']] != list(range(count))
                or complete['manifest_sha256'] != digest(execution / 'blocks_manifest.json')):
            raise ValueError('Complete zero-exit stage required: ' + stage)
        for name, expected in receipt['hashes'].items():
            if digest(execution / name) != expected:
                raise ValueError('Terminal evidence changed: ' + stage + '/' + name)
        for entry in manifest['entries']:
            if set(entry['files']) != {'complete.json', 'observations.sqlite',
                                       'observations.sqlite.seal.json'}:
                raise ValueError('Incomplete block-file coverage')
            directory = SOURCE / stage / 'blocks' / f"{entry['index']:05d}"
            for name, expected in entry['files'].items():
                if digest(directory / name) != expected:
                    raise ValueError('Sealed block changed: ' + str(directory / name))
    parallel = read(SOURCE / 'analysis/parallel_replay_execution.json')
    if (parallel['audited_episodes'] != 16384 or parallel['blocks'] != 4096
            or parallel['new_sumo_runs'] != 0 or parallel['statistical_sample_size_changed']
            or not parallel['every_original_argument_matched']
            or not parallel['every_audit_consumed_before_inference']):
        raise ValueError('Complete independent replay required')
    lineage = read(SOURCE / 'analysis/lineage_accounting.json')
    if lineage['final_unique_records'] != 16384 or lineage['statistical_attempt'] != 2:
        raise ValueError('Original sample accounting required')
    return {'finalizer_exit_code': 0, 'unique_runs': 16384,
            'independent_initials': 4096, 'frozen_inputs_sha256': PINS,
            'verification_sha256': digest(SOURCE / 'analysis/verification.json'),
            'original_certificate_sha256': verification['certificate_sha256'],
            'scientific_decision_recomputed': False,
            'scientific_success_assumed': False}


def source_files():
    paths = [p for folder in ('calibration', 'confirmation', 'analysis',
                             'parallel_replay', 'boundary_proof')
             for p in (SOURCE / folder).rglob('*') if p.is_file()]
    paths += [SOURCE / name for name in ('recovery.json', 'launch_fallback.json',
                                         'system_interruption.json')]
    return sorted(paths)


def archive_bytes(source, target, paths, limit=LIMIT):
    if target.exists():
        raise FileExistsError('Preserve any existing delivery prefix; no implicit resume')
    originals = {}
    parts, current, size = [], [], 0
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError('Only regular source files may be archived')
        name = path.relative_to(source).as_posix()
        if name in originals:
            raise ValueError('Duplicate source file')
        length = path.stat().st_size
        if length > limit:
            raise ValueError('Single file exceeds delivery budget: ' + name)
        if current and size + length > limit:
            parts.append(current)
            current, size = [], 0
        originals[name] = digest(path)
        current.append(name)
        size += length
    if current:
        parts.append(current)
    if not originals:
        raise ValueError('Empty delivery is invalid')
    target.mkdir(parents=True, exist_ok=False)
    archives, restored = [], {}
    for index, members in enumerate(parts):
        path = target / f'collection_{index:03d}.zip'
        with zipfile.ZipFile(path, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name in members:
                archive.write(source / name, name)
        if path.stat().st_size >= 40 * 1024 * 1024:
            raise ValueError('Archive exceeds Git transport budget; preserve prefix')
        with zipfile.ZipFile(path) as archive:
            if archive.namelist() != members:
                raise ValueError('Archive membership differs')
            for name in members:
                with archive.open(name) as stream:
                    restored[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
        archives.append({'name': path.name, 'sha256': digest(path),
                         'bytes': path.stat().st_size, 'files': members})
    if restored != originals or any(digest(source / name) != expected
                                    for name, expected in originals.items()):
        raise ValueError('Archive roundtrip or original source bytes changed')
    return {'original_files_sha256': originals, 'archive_parts': archives,
            'every_archived_file_byte_verified': True}


def main():
    if TARGET.exists():
        raise FileExistsError('Preserve first delivery; no overwrite')
    completion = require_finalization()
    paths = source_files()
    archive = archive_bytes(SOURCE, TARGET, paths)
    if paths != source_files():
        raise ValueError('Source membership changed during delivery')
    inventory = {'utc': datetime.now(timezone.utc).isoformat(),
                 'source': SOURCE.relative_to(ROOT).as_posix(),
                 **completion, **archive,
                 'existing_interruption_archives_duplicated': False,
                 'prior_registered_dependencies_remain_at_existing_paths': True,
                 'restore': 'Extract all parts under source and compare every relative SHA256.'}
    with (TARGET / 'inventory.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(inventory, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'files': len(paths), 'parts': len(archive['archive_parts']),
                      'bytes': sum(p['bytes'] for p in archive['archive_parts']),
                      'scientific_success_assumed': False}), flush=True)


if __name__ == '__main__':
    main()
