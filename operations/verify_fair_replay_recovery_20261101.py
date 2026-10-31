"""Preserve an interrupted review; reuse sealed data without new fair simulations."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/fair_replay_recovery_20261101'
FAIR = ROOT / 'artifacts/v44/fair_failure_preserving_execution'
SOURCES = ('operations/verify_fair_replay_recovery_20261101.py',
           'operations/continue_fair_replay_20261101.ps1',
           'operations/launch_fair_replay_20261101.ps1',
           'operations/continue_mainline_20261027_v2.ps1',
           'operations/continue_joint_recovery_20261031.ps1')


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def digest(p):
    with p.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def complete_collection():
    execution = FAIR / 'execution'
    receipt = read(execution / 'exit.json')
    done = read(execution / 'collection_complete.json')
    manifest = read(execution / 'blocks_manifest.json')
    if (receipt['exit_code'] != 0 or receipt['native_pid'] != done['native_pid']
            or done['actual_sumo_runs'] != 5120 or done['blocks'] != 160
            or manifest['actual_sumo_runs'] != 5120
            or [e['block_index'] for e in manifest['entries']] != list(range(160))
            or digest(execution / 'blocks_manifest.json') != done['blocks_manifest_sha256']):
        raise ValueError('Complete original zero-exit fair collection required')
    for name, expected in receipt['hashes'].items():
        if digest(execution / name) != expected:
            raise ValueError('Terminal evidence changed: ' + name)
    directories = {int(p.name): p for p in (FAIR / 'blocks').iterdir() if p.is_dir()}
    if set(directories) != set(range(160)):
        raise ValueError('Original complete block schedule required')
    for entry in manifest['entries']:
        for name in ('complete.json', 'observations.sqlite', 'observations.sqlite.seal.json'):
            if digest(directories[entry['block_index']] / name) != entry[name]:
                raise ValueError('Original sealed block changed')
    return digest(execution / 'exit.json')


def require_unstarted_outputs():
    for path in (FAIR / 'analysis',
                 ROOT / 'artifacts/candidate_factorial/complete_fair_cases/execution',
                 ROOT / 'artifacts/literature_reproduction/cspi_factorial_prospective/analysis'):
        if path.exists():
            raise FileExistsError('Preserve any output prefix; inspect before recovery: ' + str(path))


def freeze():
    if (OUT / 'recovery.json').exists() or (OUT / 'interruption.zip').exists():
        raise FileExistsError('Preserve first recovery registration')
    require_unstarted_outputs()
    terminal = complete_collection()
    interruption = read(OUT / 'interruption.json')
    if interruption['prior_queue_or_finalizer_live']:
        raise ValueError('Never duplicate live work')
    names = ('.git/mainline-joint-recovery-20261031.jsonl',
             '.git/mainline-joint-recovery-20261031.stdout.log',
             '.git/fair-finalizer-native-start-20261031.json')
    hashes = {n: digest(ROOT / n) for n in names}
    with zipfile.ZipFile(OUT / 'interruption.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(ROOT / name, name)
    with zipfile.ZipFile(OUT / 'interruption.zip') as archive:
        if {n: hashlib.sha256(archive.read(n)).hexdigest() for n in names} != hashes:
            raise ValueError('Interruption bytes differ')
    value = {'utc': datetime.now(timezone.utc).isoformat(),
             'original_summary_native_exit': 'unknown', 'original_queue_native_exit': 'unknown',
             'collection_exit_sha256': terminal, 'fair_runs_reused': 5120,
             'fair_independent_initials': 640, 'new_fair_sumo_runs': 0,
             'full_original_replay_repeated': True, 'partial_effects_read': False,
             'statistical_sample_size_changed': False, 'scientific_functions_changed': False,
             'thresholds_or_error_allocations_changed': False,
             'remaining_original_factorial_runs': 3200, 'new_training_recipes': 0,
             'source_sha256': {n: digest(ROOT / n) for n in SOURCES},
             'preserved_original_sha256': hashes,
             'interruption_sha256': digest(OUT / 'interruption.json'),
             'archive_sha256': digest(OUT / 'interruption.zip')}
    with (OUT / 'recovery.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'registered': True, 'fair_runs_reused': 5120, 'new_fair_runs': 0}))


def verify():
    value = read(OUT / 'recovery.json')
    require_unstarted_outputs()
    if complete_collection() != value['collection_exit_sha256']:
        raise ValueError('Collection identity changed')
    for name, expected in {**value['source_sha256'], **value['preserved_original_sha256']}.items():
        if digest(ROOT / name) != expected:
            raise ValueError('Recovery input changed: ' + name)
    if (digest(OUT / 'interruption.json') != value['interruption_sha256']
            or digest(OUT / 'interruption.zip') != value['archive_sha256']):
        raise ValueError('Interruption evidence changed')
    print('Preserved full fair collection verified; original full replay required.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'verify'))
    globals()[parser.parse_args().command]()
