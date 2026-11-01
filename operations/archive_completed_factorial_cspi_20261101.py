"""Deliver every finalized factorial and original-author CSPI byte unchanged.

Delivery only: no simulations, model selection, outcome filtering or inference.
The original complete queue must have exited successfully before outcomes open.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

from archive_completed_joint_20261031 import archive_bytes, digest, read

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts'
FACTORIAL = SOURCE / 'candidate_factorial/complete_fair_cases'
CSPI = SOURCE / 'literature_reproduction/cspi_factorial_prospective'
TARGET = SOURCE / 'factorial_cspi_completed_delivery_20261101'
PLAN_SHA = '437c5386f0c6962e214c3f21d2b997ca40e6f6b6ba336068b7d519823e1830b1'
CSPI_SHA = '01f02985ab61ba5af2e71f513cfd429acc879f8446963c8b12d48edd7badddb1'


def require_completion():
    events = [json.loads(line) for line in
              (ROOT / '.git/mainline-fair-replay-20261101.jsonl').read_text(encoding='utf-8-sig').splitlines()]
    if not any(event['kind'] == 'queue_completed' for event in events):
        raise ValueError('Original factorial/CSPI queue incomplete; no outcomes read')
    for command in ('factorial', 'cspi-export', 'cspi-analysis'):
        exits = [event for event in events if event['kind'] == 'command_exit'
                 and event['detail'].get('arguments', [])[1:4]
                 == ['utf8', 'tools/finalize_episode_keys.py', command]]
        if len(exits) != 1 or exits[0]['detail']['exit_code'] != 0:
            raise ValueError('Full original actual-zero stage required: ' + command)
    if digest(FACTORIAL / 'plan.json') != PLAN_SHA or digest(CSPI / 'plan.json') != CSPI_SHA:
        raise ValueError('Frozen factorial or author-code supplement plan changed')
    execution = FACTORIAL / 'execution'
    receipt = read(execution / 'exit.json')
    complete = read(execution / 'collection_complete.json')
    manifest = read(execution / 'blocks_manifest.json')
    verification = read(FACTORIAL / 'analysis/verification.json')
    if (receipt['exit_code'] != 0 or receipt['native_pid'] != complete['native_pid']
            or complete['blocks'] != 640 or complete['actual_sumo_runs'] != 3200
            or manifest['actual_sumo_runs'] != 3200 or verification['actual_sumo_runs'] != 3200
            or complete['blocks_manifest_sha256'] != digest(execution / 'blocks_manifest.json')
            or [item['block_index'] for item in manifest['entries']] != list(range(640))
            or any(value['plan_sha256'] != PLAN_SHA for value in (receipt, complete, manifest, verification))):
        raise ValueError('Complete immutable factorial collection and review required')
    for name, expected in verification['terminal_hashes'].items():
        if digest(execution / name) != expected:
            raise ValueError('Finalized terminal evidence changed: ' + name)
    for item in manifest['entries']:
        if (set(item) != {'block_index', 'files'}
                or set(item['files']) != {'complete.json', 'observations.sqlite',
                                         'observations.sqlite.seal.json'}):
            raise ValueError('Incomplete block-file coverage')
        for name, expected in item['files'].items():
            path = FACTORIAL / 'blocks' / f"{item['block_index']:05d}" / name
            if digest(path) != expected:
                raise ValueError('Sealed factorial block changed: ' + str(path))
    for name in ('report', 'replay'):
        if digest(FACTORIAL / f'analysis/{name}.json') != verification[f'{name}_sha256']:
            raise ValueError('Full original analysis changed: ' + name)
    reviews = read(FACTORIAL / 'analysis/replay.json')['reviews']
    plan = read(FACTORIAL / 'plan.json')
    expected_keys = {(case['cluster_id'], variant) for case in plan['cases']
                     for variant in case['variants']}
    keys = [(review['cluster_id'], review['variant']) for review in reviews]
    if len(keys) != 3200 or len(set(keys)) != 3200 or set(keys) != expected_keys:
        raise ValueError('Exactly every original factorial review required')
    author = read(CSPI / 'analysis/complete.json')
    if (author['new_sumo_runs'] != 0 or author['additional_independent_clusters'] != 0
            or author['evaluated_independent_parent_clusters'] != 128
            or author['finite_sample_risk_certificate']):
        raise ValueError('Original author-code supplement scope changed')
    for name, expected in author['files_sha256'].items():
        if digest(CSPI / 'analysis' / name) != expected:
            raise ValueError('Finalized author analysis changed: ' + name)
    exported = read(CSPI / 'data/complete.json')
    for name, expected in exported['files_sha256'].items():
        if digest(CSPI / 'data' / name) != expected:
            raise ValueError('Original randomized export changed: ' + name)
    return {'factorial_sumo_runs': 3200, 'independent_initials': 640,
            'additional_independent_initials_beyond_fair': 0,
            'full_factorial_review_keys_verified': True,
            'original_author_evaluation_initials': 128, 'new_cspi_sumo_runs': 0,
            'finalized_verification_sha256': digest(FACTORIAL / 'analysis/verification.json'),
            'original_author_completion_sha256': digest(CSPI / 'analysis/complete.json'),
            'scientific_success_assumed': False, 'new_numerical_inference': False}


def source_files():
    return sorted(path for folder in (FACTORIAL, CSPI)
                  for path in folder.rglob('*') if path.is_file())


def main():
    if TARGET.exists():
        raise FileExistsError('Preserve original delivery; no implicit resume or overwrite')
    completion = require_completion()
    paths = source_files()
    # Smaller transport objects; full original records and statistical scope stay intact.
    archive = archive_bytes(SOURCE, TARGET, paths, limit=12 * 1024 * 1024)
    if paths != source_files():
        raise ValueError('Finalized source membership changed during delivery')
    inventory = {'utc': datetime.now(timezone.utc).isoformat(),
                 'source': SOURCE.relative_to(ROOT).as_posix(),
                 'source_subdirectories': [path.relative_to(SOURCE).as_posix()
                                           for path in (FACTORIAL, CSPI)],
                 **completion, **archive,
                 'restore': 'Extract all parts under artifacts; compare every relative SHA256.'}
    with (TARGET / 'inventory.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(inventory, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'files': len(paths), 'parts': len(archive['archive_parts']),
                      'bytes': sum(part['bytes'] for part in archive['archive_parts']),
                      'scientific_success_assumed': False}), flush=True)


if __name__ == '__main__':
    main()
