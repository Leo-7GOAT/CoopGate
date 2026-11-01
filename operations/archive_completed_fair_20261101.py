"""Archive the complete, failure-preserving fair study without new inference."""
from datetime import datetime, timezone
import json
from pathlib import Path

from archive_completed_joint_20261031 import archive_bytes, digest, read

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/v44/fair_failure_preserving_execution'
RECOVERY = ROOT / 'artifacts/fair_replay_recovery_20261101'
TARGET = ROOT / 'artifacts/fair_completed_delivery_20261101'
PINS = {
    'plan.json': '9cc7d4839432ec5cc567d59c00069a792e075eea219a28a8fec8eefc89ab47d5',
    'binding.json': '8afa12e881b695be49072623d1de5195e3cb02b1540225ee0d57a73beed76c88',
}


def require_completion():
    for name, expected in PINS.items():
        if digest(SOURCE / name) != expected:
            raise ValueError('Frozen input changed: ' + name)
    journal = ROOT / '.git/mainline-fair-replay-20261101.jsonl'
    args = ['-X', 'utf8', 'tools/execute_fair_failure_v44.py', 'summarize']
    exits = [event for line in journal.read_text(encoding='utf-8-sig').splitlines()
             if (event := json.loads(line)).get('kind') == 'command_exit'
             and event['detail'].get('arguments') == args]
    native = read(RECOVERY / 'native_exit.json')
    if (len(exits) != 1 or exits[0]['detail']['exit_code'] != 0
            or native['exit_code'] != 0 or not native['retained_native_handle']
            or native['start_sha256'] != digest(RECOVERY / 'native_start.json')
            or native['native_pid'] != read(RECOVERY / 'native_start.json')['native_pid']):
        raise ValueError('Complete actual-zero finalizer required')
    execution = SOURCE / 'execution'
    receipt = read(execution / 'exit.json')
    complete = read(execution / 'collection_complete.json')
    manifest = read(execution / 'blocks_manifest.json')
    verification = read(SOURCE / 'analysis/verification.json')
    if (receipt['exit_code'] != 0 or receipt['native_pid'] != complete['native_pid']
            or complete['actual_sumo_runs'] != 5120 or complete['blocks'] != 160
            or manifest['actual_sumo_runs'] != 5120
            or verification['actual_sumo_runs'] != 5120
            or verification['exit_sha256'] != digest(execution / 'exit.json')
            or complete['blocks_manifest_sha256'] != digest(execution / 'blocks_manifest.json')
            or [entry['block_index'] for entry in manifest['entries']] != list(range(160))):
        raise ValueError('Complete collection and unchanged full review required')
    for record in (receipt, complete, manifest, verification):
        if any(record[name.replace('.json', '_sha256')] != sha for name, sha in PINS.items()):
            raise ValueError('Frozen study identity differs')
    for name, expected in receipt['hashes'].items():
        if digest(execution / name) != expected:
            raise ValueError('Collection evidence changed: ' + name)
    for entry in manifest['entries']:
        if set(entry) != {'block_index', 'complete.json', 'observations.sqlite',
                          'observations.sqlite.seal.json'}:
            raise ValueError('Incomplete block coverage')
        for name, expected in entry.items():
            if name != 'block_index':
                path = SOURCE / 'blocks' / f"{entry['block_index']:05d}" / name
                if digest(path) != expected:
                    raise ValueError('Sealed block changed: ' + str(path))
    for name in ('report', 'replay'):
        if digest(SOURCE / f'analysis/{name}.json') != verification[f'{name}_sha256']:
            raise ValueError('Finalized result changed: ' + name)
    reviews = read(SOURCE / 'analysis/replay.json')['reviews']
    if len(reviews) != 5120 or not all(r['all_actual_failures_retained'] for r in reviews):
        raise ValueError('Full failure-preserving review required')
    return {'actual_sumo_runs': 5120, 'independent_initials': 640,
            'methods': 8, 'finalizer_exit_code': 0,
            'original_interrupted_review_exit': 'unknown',
            'verification_sha256': digest(SOURCE / 'analysis/verification.json'),
            'native_exit_sha256': digest(RECOVERY / 'native_exit.json'),
            'all_recorded_failures_retained': True,
            'scientific_success_assumed': False, 'new_inference': False}


def source_files():
    return sorted(path for path in SOURCE.rglob('*') if path.is_file())


def main():
    if TARGET.exists():
        raise FileExistsError('Preserve original delivery; no implicit overwrite or resume')
    completion = require_completion()
    paths = source_files()
    archive = archive_bytes(SOURCE, TARGET, paths)
    if paths != source_files():
        raise ValueError('Completed source membership changed during delivery')
    inventory = {'utc': datetime.now(timezone.utc).isoformat(),
                 'source': SOURCE.relative_to(ROOT).as_posix(),
                 **completion, **archive,
                 'restore': 'Extract all parts under source; compare every relative SHA256.'}
    with (TARGET / 'inventory.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(inventory, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'files': len(paths), 'parts': len(archive['archive_parts']),
                      'bytes': sum(p['bytes'] for p in archive['archive_parts']),
                      'all_recorded_failures_retained': True}), flush=True)


if __name__ == '__main__':
    main()
