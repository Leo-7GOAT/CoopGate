"""Second explicit continuation; reuse verified recovery mechanics and the exact scientific plan."""
import argparse
import json
from pathlib import Path
import shutil

import execute_prospective_v23 as original
import recover_prospective_v24 as mechanics
import snapshot_interruption_v29 as snapshot

ROOT = original.ROOT
PRIOR = snapshot.CAL
OUTPUT = ROOT/'artifacts/v29/prospective_calibration_recovery_20261023'
RECOVERY = OUTPUT/'recovery'
SCRIPT = 'tools/recover_prospective_v29.py'
SOURCES = (SCRIPT, 'tests/test_repeat_recovery_v29.py', 'tools/snapshot_interruption_v29.py',
           'tools/recover_prospective_v24.py', 'tests/test_recover_prospective_v24.py')


def freeze():
    snapshot.assert_stopped()
    if OUTPUT.exists():
        raise FileExistsError('Preserve this explicit recovery after any further interruption')
    evidence = snapshot.verify_snapshot('calibration')
    _, plan = original.load('prospective')
    original.sources_committed([ROOT/name for name in SOURCES])
    if original.digest(PRIOR/'frozen/plan.json') != evidence['source_plan_sha256']:
        raise ValueError('Original scientific identity changed')
    _, _, checked = snapshot.checked_blocks('calibration')
    if checked != evidence['blocks']:
        raise ValueError('Preserved calibration boundary changed')
    entries = [{k:v for k,v in e.items() if k != 'inventory'} for e in checked]
    spec = plan['stages']['calibration']
    RECOVERY.mkdir(parents=True)
    (OUTPUT/'frozen').mkdir()
    shutil.copy2(PRIOR/'frozen/plan.json', OUTPUT/'frozen/plan.json')
    recovery = dict(utc=original.utc(), original_plan_sha256=evidence['source_plan_sha256'],
        source_sha256={name:original.digest(ROOT/name) for name in SOURCES},
        inventory_sha256=original.digest(snapshot.OUT/'calibration/inventory.json'),
        parent_recovery_plan_sha256=original.digest(PRIOR/'recovery/plan.json'),
        original_exit_status='unknown', blocks=entries, retained_roles=evidence['retained_roles'],
        missing_roles=spec['total_sumo_runs']-evidence['retained_roles'],
        boundary_replays=evidence['partial_blocks'], initial_count=spec['n_initials'],
        confirmation_initial_count=plan['stages']['confirmation']['n_initials'],
        original_attempt=plan['attempt'], allocated_error=plan['allocated_error'],
        schedule_models_thresholds_unchanged=True, statistical_sample_size_unchanged=True,
        mechanics='Unmodified V24 boundary replay, exact blob continuation, independent review and actual-exit checks; only artifact routing and immutable snapshot checks are configured here.',
        earlier_boundary_replays=original.read(PRIOR/'recovery/plan.json')['boundary_replays'],
        earlier_exits_remain_unknown=True, partial_effectiveness_outcomes_read=False,
        repeated_interruption_rule='Never automatically resume or replace this recovery; preserve prefixes for another explicit audit.')
    original.exclusive_json(RECOVERY/'plan.json', recovery)
    for name in SOURCES:
        target = RECOVERY/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
    original.exclusive_json(RECOVERY/'sha256.json', original.manifest(RECOVERY))
    print(json.dumps({k:recovery[k] for k in ('retained_roles','missing_roles','boundary_replays','initial_count')}), flush=True)


def inputs():
    _, plan = original.load('prospective')
    original.verify_manifest(RECOVERY)
    original.require_committed([RECOVERY/'plan.json', RECOVERY/'sha256.json', OUTPUT/'frozen/plan.json'])
    recovery = original.read(RECOVERY/'plan.json')
    evidence = snapshot.verify_snapshot('calibration')
    if (original.digest(OUTPUT/'frozen/plan.json') != evidence['source_plan_sha256']
            or recovery['original_plan_sha256'] != evidence['source_plan_sha256']
            or recovery['inventory_sha256'] != original.digest(snapshot.OUT/'calibration/inventory.json')
            or recovery['parent_recovery_plan_sha256'] != original.digest(PRIOR/'recovery/plan.json')
            or recovery['original_attempt'] != plan['attempt'] or recovery['allocated_error'] != plan['allocated_error']
            or recovery['retained_roles'] != evidence['retained_roles']
            or recovery['boundary_replays'] != evidence['partial_blocks']):
        raise ValueError('Repeated recovery changed the original statistical or preservation identity')
    for name, expected in {**evidence['source_sha256'], **recovery['source_sha256']}.items():
        if original.digest(ROOT/name) != expected:
            raise ValueError('Frozen source changed: '+name)
    original.sources_committed([ROOT/name for name in recovery['source_sha256']])
    return recovery, plan


def configure():
    # No simulation, gate, candidate, scoring or inference function is replaced.
    mechanics.PRIOR, mechanics.OUTPUT, mechanics.RECOVERY = PRIOR, OUTPUT, RECOVERY
    mechanics.SCRIPT, mechanics.inputs = SCRIPT, inputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest='command', required=True)
    subs.add_parser('freeze')
    for command in ('watch','supervise','worker','analyze'):
        item = subs.add_parser(command)
        item.add_argument('stage', choices=('calibration','confirmation'))
        if command == 'worker':
            item.add_argument('phase', choices=('boundary','collect','review'))
            item.add_argument('shard', type=int)
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze()
    else:
        configure()
        if args.command == 'worker':
            mechanics.worker(args.stage, args.phase, args.shard)
        else:
            getattr(mechanics, args.command)(args.stage)
