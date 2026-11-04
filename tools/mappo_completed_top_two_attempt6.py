"""Require complete native and recovery evidence before reading primary effects.

This does not assume a positive scientific result. Source/study files are read
only; the registered runner, inference and any adverse observations stay fixed.
"""
import argparse
import json
from pathlib import Path
import recover_mappo_top_two_attempt6 as recovery
from mappo_reviewed_storage_attempt6 import verify_reviewed_blocks

e = recovery.original


def complete_evidence(study=None):
    study = Path(study) if study is not None else recovery.OUT
    # Root's direct native wait must precede every access to treatment effects.
    owner = e.read(study / 'orchestrator_native_receipt.json')
    if owner['actual_native_exit_code'] != 0 or not owner['observed_by_direct_native_wait']:
        raise ValueError('No actual successful native recovery owner')
    if owner['original_owner_exit_code'] is not None or not owner['original_interrupted_attempt_retained']:
        raise ValueError('Do not invent a terminal exit for the interrupted owner')
    if owner['completion_receipt_sha256'] != e.sha(study / 'completion_receipt.json'):
        raise ValueError('Owner does not bind this exact completion receipt')
    launch = e.read(study / 'owner_launch.json')
    if owner['native_pid'] != launch['native_pid']:
        raise ValueError('Recovery owner PID differs from actual launch')
    receipt = e.read(study / 'completion_receipt.json')
    if (receipt['actual_native_finalizer']['actual_native_exit_code'] != 0
            or receipt['total_actual_sumo_runs'] != 59392
            or receipt['main_actual_sumo_runs'] != 28672
            or receipt['independent_initials'] != 7168):
        raise ValueError('Full original registered scope is incomplete')
    expected = {(stage, action, shard) for stage in ('calibration', 'confirmation')
                for action in ('collect', 'review') for shard in range(4)}
    stages = receipt['stages']
    if (len(stages) != 16 or {(v['stage'], v['action'], v['shard']) for v in stages} != expected
            or any(v['actual_native_exit_code'] != 0 for v in stages)):
        raise ValueError('Every exact collector and independent reviewer must finish')
    binding, audit = recovery.guard()
    plan = e.bound(study, False)
    counts = []
    for stage in ('calibration', 'confirmation'):
        for shard in range(4):
            directory = study / stage / f'shard{shard}'
            actual = e.read(directory / 'actual_recovery_collection_counts.json')
            collected = e.read(directory / 'collection_complete.json')
            proof = e.read(directory / 'verification.json')
            event = next(v for v in stages if v['stage'] == stage and v['shard'] == shard and v['action'] == 'review')
            collect_event = next(v for v in stages if v['stage'] == stage and v['shard'] == shard and v['action'] == 'collect')
            expected_n = 10752 if stage == 'calibration' else 4096
            retained_n = sum(v['retained_records'] for v in audit['blocks'] if v['shard'] == shard) if stage == 'calibration' else 0
            if (actual['retained'] != retained_n or actual['new'] != expected_n - retained_n
                    or actual['retained'] + actual['new'] != expected_n
                    or collected['executions'] != expected_n or collected['expected'] != expected_n
                    or proof['executions'] != expected_n or not proof['all_recorded_outcomes_replay']
                    or proof['collection_complete_sha256'] != e.sha(directory / 'collection_complete.json')
                    or event['verification_sha256'] != e.sha(directory / 'verification.json')
                    or collect_event['native_pid'] != collected['native_pid']
                    or collect_event['collection_complete_sha256'] != e.sha(directory / 'collection_complete.json')):
                raise ValueError('Per-shard exact replay or physical execution accounting differs')
            # Validate every scheduled sealed block identity/hash without using
            # effects to decide which records should remain in the dataset.
            executions = verify_reviewed_blocks(e,study,stage,shard,plan)['executions']
            if executions != expected_n:
                raise ValueError('Sealed full schedule differs from native receipts')
            counts.append(actual)
    accounting = e.read(study / 'recovery_delivery_accounting.json')
    if (not accounting['complete'] or accounting['retained_actual_records'] != 642
            or accounting['new_actual_sumo_executions'] != 58750
            or accounting['full_effective_schedule'] != 59392
            or accounting['original_root_exit_code'] is not None
            or not accounting['original_unknown_lost_inflight_executions']
            or accounting['new_error_allocated']
            or sum(q['retained'] for q in counts) != 642 or sum(q['new'] for q in counts) != 58750
            or accounting['old_allocation_sha256'] != e.sha(e.FORMAL / 'allocation.json')):
        raise ValueError('Interrupted attempts and retained/new accounting must remain explicit')
    snapshot = e.read(study / 'source_snapshot.json')
    hashes = dict(plan['source_sha256'])
    hashes['artifacts/mappo_ctde_20261101/top_two_fixed_candidate_20261104.json'] = plan['candidate_sha256']
    hashes['artifacts/mappo_ctde_20261101/development/compact_coupled_bank_20261102.json'] = e.design.validate_candidate()['coupled_bank_sha256']
    if (snapshot['source_commit'] != plan['source_commit'] or snapshot['plan_sha256'] != e.sha(study / 'plan.json')
            or len(snapshot['files']) != len(hashes)):
        raise ValueError('Exact source snapshot metadata differs')
    seen = set()
    for item in snapshot['files']:
        name = item['path']; path = (study / 'source_snapshot' / name).resolve()
        if (name in seen or hashes.get(name) != item['sha256']
                or not path.is_relative_to((study / 'source_snapshot').resolve())
                or e.sha(path) != item['sha256']):
            raise ValueError('Snapshot bytes or membership changed')
        seen.add(name)
    if seen != set(hashes):
        raise ValueError('Source snapshot is incomplete')
    for name in ('certificate', 'comparisons'):
        if e.sha(study / f'analysis/{name}.json') != receipt[name + '_sha256']:
            raise ValueError('Final inference bytes changed')
    cert = e.read(study / 'analysis/certificate.json')
    comp = e.read(study / 'analysis/comparisons.json')
    if (cert['candidate_sha256'] != plan['candidate_sha256'] or cert['attempt'] != 6
            or cert['allocated_error'] != .05/42 or cert['fixed_models'] != [4703,4704,4705]
            or cert['common_gain_floor'] != .0025 or cert['calibration_initials'] != 6144
            or cert['confirmation_initials'] != 1024 or cert['primary_risk']['risk_limit'] != .05
            or cert['adoption']['threshold'] != .2 or cert['paired_gain']['threshold_percent'] != .5
            or not cert['intersection_union_test'] or cert['controller_family_selection_permitted']
            or cert['confirmation_outcomes_used_for_selection']):
        raise ValueError('Original scientific family or inferential rules changed')
    if set(comp['confirmation_methods']) != set(e.runtime.CONFIRMATION_METHODS):
        raise ValueError('Required same-pool and same-adoption comparison family missing')
    return cert, comp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check and not (recovery.OUT / 'orchestrator_native_receipt.json').exists():
        print(json.dumps(dict(complete=False, effects_read=False, reason='Waiting for actual native recovery owner')))
    else:
        cert, _ = complete_evidence()
        print(json.dumps(dict(execution_complete=True, joint_deployment_certified=cert['joint_deployment_certified'],
                              requested_combined_evidence_complete=cert['requested_combined_evidence_complete'])))
