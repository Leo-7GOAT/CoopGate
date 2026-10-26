"""Fourth explicit recovery of the unchanged prospective calibration.

The V24 continuation mechanics are reused unchanged. This module only binds
them to the immutable V38 snapshot and a fresh output directory.
"""
import argparse
import json
from pathlib import Path
import shutil

import execute_prospective_v23 as original
import recover_prospective_v24 as mechanics
import snapshot_interruption_20261026 as snapshot


ROOT = original.ROOT
PRIOR = snapshot.CAL
OUTPUT = ROOT/'artifacts/recovery_20261026/calibration_recovery'
RECOVERY = OUTPUT/'recovery'
SCRIPT = 'tools/recover_prospective_20261026.py'
SOURCES = (SCRIPT,'tools/snapshot_interruption_20261026.py','tests/test_recovery_20261026.py',
           'tools/recover_prospective_v29.py','tests/test_repeat_recovery_v29.py',
           'tools/recover_prospective_v24.py','tests/test_recover_prospective_v24.py')


def validate_evidence(evidence,plan):
    spec = plan['stages']['calibration']
    if (evidence['interruption_generation'] != 4
            or evidence['source_plan_sha256'] != original.digest(PRIOR/'frozen/plan.json')
            or evidence['parent_recovery_plan_sha256'] != original.digest(PRIOR/'recovery/plan.json')
            or evidence['original_exit_status'] != 'unknown'
            or evidence['source_exit_receipt_present']
            or evidence['partial_effectiveness_outcomes_read']
            or evidence['initial_count'] != spec['n_initials']
            or evidence['confirmation_initial_count'] != plan['stages']['confirmation']['n_initials']
            or evidence['original_attempt'] != plan['attempt']
            or evidence['allocated_error'] != plan['allocated_error']
            or not evidence['schedule_models_thresholds_unchanged']
            or not evidence['statistical_sample_size_unchanged']
            or evidence['retained_roles'] != sum(item['records'] for item in evidence['blocks'])
            or evidence['fully_completed_blocks'] != sum(item['complete'] for item in evidence['blocks'])
            or evidence['partial_blocks'] != sum(not item['complete'] for item in evidence['blocks'])
            or not 0 < evidence['retained_roles'] < spec['total_sumo_runs']):
        raise ValueError('Fourth-interruption evidence changed the frozen statistical identity')
    return evidence


def freeze():
    snapshot.assert_stopped()
    if OUTPUT.exists():
        raise FileExistsError('Preserve this explicit recovery after any further interruption')
    evidence = snapshot.verify_snapshot()
    _, plan = original.load('prospective')
    validate_evidence(evidence,plan)
    if plan != original.read(PRIOR/'frozen/plan.json'):
        raise ValueError('Current registered plan differs from the interrupted frozen plan')
    original.sources_committed([ROOT/name for name in SOURCES])
    _, checked_plan, checked = snapshot.checked_blocks()
    if checked_plan != original.read(PRIOR/'frozen/plan.json') or checked != evidence['blocks']:
        raise ValueError('Preserved fourth-interruption boundary changed')
    entries = [{key:value for key,value in item.items() if key != 'inventory'} for item in checked]
    sources = snapshot.merge_source_hashes(evidence['source_sha256'],
        {name:original.digest(ROOT/name) for name in SOURCES})
    for name, expected in sources.items():
        if original.digest(ROOT/name) != expected:
            raise ValueError('Frozen transitive dependency changed: '+name)
    original.sources_committed([ROOT/name for name in sources])
    RECOVERY.mkdir(parents=True)
    (OUTPUT/'frozen').mkdir()
    shutil.copy2(PRIOR/'frozen/plan.json',OUTPUT/'frozen/plan.json')
    parent = original.read(PRIOR/'recovery/plan.json')
    recovery = dict(utc=original.utc(),interruption_generation=4,
        original_plan_sha256=evidence['source_plan_sha256'],source_sha256=sources,
        inventory_sha256=original.digest(snapshot.OUT/'calibration/inventory.json'),
        parent_recovery_plan_sha256=evidence['parent_recovery_plan_sha256'],
        original_exit_status='unknown',blocks=entries,retained_roles=evidence['retained_roles'],
        missing_roles=plan['stages']['calibration']['total_sumo_runs']-evidence['retained_roles'],
        boundary_replays=evidence['partial_blocks'],initial_count=evidence['initial_count'],
        confirmation_initial_count=evidence['confirmation_initial_count'],
        original_attempt=evidence['original_attempt'],allocated_error=evidence['allocated_error'],
        schedule_models_thresholds_unchanged=True,statistical_sample_size_unchanged=True,
        mechanics='Unmodified V24 boundary replay, exact SQLite-blob continuation, suffix-only collection, independent review and actual-exit checks; only immutable artifact routing is new.',
        earlier_boundary_replays=parent['boundary_replays']+parent.get('earlier_boundary_replays',0),
        all_earlier_exits_remain_unknown=True,partial_effectiveness_outcomes_read=False,
        boundary_rule='All partial last-committed roles replay exactly before any missing original role is collected.',
        suffix_rule='Complete blocks copy byte-for-byte; partial databases copy exact row/trace blobs into new stores; only the frozen missing suffix executes.',
        inference_rule='No calibration analysis or candidate selection before all 4096 initials, independent review, manifests and actual native zero exits are complete.',
        repeated_interruption_rule='Never automatically resume or replace this recovery; preserve every prefix for a new explicit audit.')
    original.exclusive_json(RECOVERY/'plan.json',recovery)
    for name in SOURCES:
        target = RECOVERY/'source'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
    original.exclusive_json(RECOVERY/'sha256.json',original.manifest(RECOVERY))
    print(json.dumps({key:recovery[key] for key in
        ('retained_roles','missing_roles','boundary_replays','initial_count')}),flush=True)


def inputs():
    _, plan = original.load('prospective')
    if plan != original.read(PRIOR/'frozen/plan.json'):
        raise ValueError('Current registered plan differs from the interrupted frozen plan')
    original.verify_manifest(RECOVERY)
    original.require_committed([RECOVERY/'plan.json',RECOVERY/'sha256.json',OUTPUT/'frozen/plan.json'])
    recovery = original.read(RECOVERY/'plan.json')
    evidence = validate_evidence(snapshot.verify_snapshot(),plan)
    if (original.digest(OUTPUT/'frozen/plan.json') != evidence['source_plan_sha256']
            or recovery['interruption_generation'] != 4
            or recovery['original_plan_sha256'] != evidence['source_plan_sha256']
            or recovery['inventory_sha256'] != original.digest(snapshot.OUT/'calibration/inventory.json')
            or recovery['parent_recovery_plan_sha256'] != evidence['parent_recovery_plan_sha256']
            or recovery['original_attempt'] != plan['attempt']
            or recovery['allocated_error'] != plan['allocated_error']
            or recovery['retained_roles'] != evidence['retained_roles']
            or recovery['boundary_replays'] != evidence['partial_blocks']
            or recovery['blocks'] != [{key:value for key,value in item.items() if key != 'inventory'}
                                       for item in evidence['blocks']]):
        raise ValueError('Fourth recovery changed preservation or statistical identity')
    sources = snapshot.merge_source_hashes(evidence['source_sha256'],recovery['source_sha256'])
    if sources != recovery['source_sha256']:
        raise ValueError('Fourth recovery source set differs from the immutable snapshot')
    for name, expected in sources.items():
        if original.digest(ROOT/name) != expected:
            raise ValueError('Frozen source changed: '+name)
    original.sources_committed([ROOT/name for name in recovery['source_sha256']])
    return recovery,plan


def configure():
    # Rebind storage only. Every simulation, boundary, review and inference
    # function remains the already tested V24 implementation.
    mechanics.PRIOR,mechanics.OUTPUT,mechanics.RECOVERY = PRIOR,OUTPUT,RECOVERY
    mechanics.SCRIPT,mechanics.inputs = SCRIPT,inputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    commands.add_parser('freeze')
    for command in ('watch','supervise','worker','analyze'):
        item = commands.add_parser(command)
        item.add_argument('stage',choices=('calibration','confirmation'))
        if command == 'worker':
            item.add_argument('phase',choices=('boundary','collect','review'))
            item.add_argument('shard',type=int)
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze()
    else:
        configure()
        if args.command == 'worker':
            mechanics.worker(args.stage,args.phase,args.shard)
        else:
            getattr(mechanics,args.command)(args.stage)
