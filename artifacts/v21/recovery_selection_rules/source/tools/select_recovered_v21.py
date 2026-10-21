"""Preserve prospective selection rules, with explicit recovered-study evidence.

This supplements, and does not forge or weaken, the original exit requirement:
unknown original exits are disclosed; immutable prefix snapshots, exact parity,
new handle-based exits and full independent reviews are additionally required.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from coopgate.prospective_validation_v21 import profile_names, select_development_family, validate_attempt_history
from prepare_prospective_validation_v21 import ATTEMPTS, RULES, read_rules
from probe_disturbed_gate_v19 import digest, save
from start_final_v19 import require_committed, verify_manifest
import recover_interrupted_v21 as recovery
from study_phase_repair_v21 import validate_worker_exits

ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / 'artifacts/v21/recovery_selection_rules'
SOURCES = ('tools/select_recovered_v21.py', 'tests/test_recovered_selection_v21.py')


def completion_receipts(directory, plan_sha):
    directory = Path(directory)
    supervisor = recovery.read(directory / 'supervisor_exit.json')
    if supervisor.get('exit_code') != 0 or supervisor.get('recovery_plan_sha256') != plan_sha:
        raise ValueError('Recovery supervisor has not successfully exited with this plan')
    completed = recovery.read(directory / 'completed.json')
    if completed.get('original_exit_status') != 'unknown':
        raise ValueError('Original exits must remain explicitly unknown')
    supervisor_ids = set()
    evidence = {}
    for stage, count in (('preflight', 3), ('suffix', 3), ('review', 2)):
        launch_path, exits_path = directory / f'{stage}_launch.json', directory / f'{stage}_exits.json'
        launch, exits = recovery.read(launch_path), recovery.read(exits_path)
        if launch.get('stage') != stage or launch.get('recovery_plan_sha256') != plan_sha:
            raise ValueError('Stage launch identity differs')
        validate_worker_exits(exits['workers'], launch, count)
        expected_suffixes = ([['worker', stage, str(i)] for i in range(3)] if stage != 'review'
                             else [['finalize', 'phase'], ['finalize', 'transfer']])
        actual = [command[4:] for command in launch['commands']]
        if actual != expected_suffixes or any(command[1:4] != ['-X', 'utf8', 'tools/recover_interrupted_v21.py'] for command in launch['commands']):
            raise ValueError('Stage commands differ from the frozen recovery executor')
        supervisor_ids.add(launch['supervisor_pid'])
        evidence[stage] = dict(launch_sha256=digest(launch_path), exits_sha256=digest(exits_path))
    if len(supervisor_ids) != 1:
        raise ValueError('Recovery stages do not share their original supervisor')
    return dict(**evidence, supervisor_exit_sha256=digest(directory / 'supervisor_exit.json'),
        completed_sha256=digest(directory / 'completed.json'), original_exit_status='unknown')


def freeze():
    if GUARD.exists():
        raise FileExistsError('Preserve the original recovery admission rules')
    read_rules()
    recovery.inputs(check_original=True)
    require_committed([ROOT / name for name in SOURCES])
    GUARD.mkdir(parents=True)
    plan = dict(purpose='Accept fully reviewed recovery without altering prospective statistical selection',
        recovery_plan_sha256=digest(recovery.FROZEN / 'plan.json'),
        recovery_frozen_manifest_sha256=digest(recovery.FROZEN / 'sha256.json'),
        original_rules_sha256=digest(RULES / 'plan.json'), original_rules_manifest_sha256=digest(RULES / 'sha256.json'),
        original_exit_status='unknown', partial_outcome_statistics_read=False,
        prerequisites=['immutable prefix archives and unchanged original files', 'all scheduled actual SUMO parity replays',
            'new original process-handle exits for preflight, suffix and review', 'recovery supervisor exit receipt',
            'both complete full-grid independent reviews and exact retained/resumed provenance'],
        changed_candidate_families=False, changed_selection_thresholds=False, changed_risk_budget=False,
        changed_calibration_or_confirmation_sample_sizes=False, calibration_execution_protocol_still_required=True,
        source_sha256={name: digest(ROOT / name) for name in SOURCES})
    save(GUARD / 'plan.json', plan)
    for name in SOURCES:
        destination = GUARD / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    save(GUARD / 'sha256.json', recovery.manifest(GUARD))
    print(json.dumps(dict(guard_sha256=digest(GUARD / 'plan.json'), risk_certified=False)), flush=True)


def inputs():
    verify_manifest(GUARD)
    guard = recovery.read(GUARD / 'plan.json')
    require_committed([GUARD / 'plan.json', GUARD / 'sha256.json', *[ROOT / name for name in SOURCES]])
    for name, expected in guard['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Recovery admission source changed')
    for path, key in ((recovery.FROZEN / 'plan.json', 'recovery_plan_sha256'),
                      (recovery.FROZEN / 'sha256.json', 'recovery_frozen_manifest_sha256'),
                      (RULES / 'plan.json', 'original_rules_sha256'),
                      (RULES / 'sha256.json', 'original_rules_manifest_sha256')):
        if digest(path) != guard[key]:
            raise ValueError('Recovery or statistical rules changed')
    return guard


def select(attempt):
    guard = inputs()
    # No outcome summary is read before all actual process exits are proved.
    evidence = completion_receipts(recovery.EXECUTION, guard['recovery_plan_sha256'])
    recovery.assert_no_existing_workers(recovery.native_snapshot())
    rules = read_rules()
    plan, protocols = recovery.inputs(check_original=True)
    completed = recovery.read(recovery.EXECUTION / 'completed.json')
    summaries = {}
    for study, protocol in protocols.items():
        result = recovery.RECOVERY / 'results' / study
        verify_manifest(result)
        if digest(result / 'summary.json') != completed['recovered_summary_sha256'][study]:
            raise ValueError('Supervisor reviewed a different summary')
        summary = recovery.read(result / 'summary.json')
        if (summary['actual_sumo_runs'] != protocol['spec']['expected_actual_sumo_runs']
                or summary['independent_review_counts']['physical_traces'] != summary['actual_sumo_runs']
                or summary['protocol_sha256'] != plan['studies'][study]['protocol_sha256']
                or summary['recovery_plan_sha256'] != guard['recovery_plan_sha256']
                or summary['original_exit_status'] != 'unknown'):
            raise ValueError('Complete recovered study identity differs')
        rows, provenance = recovery.assembled_rows(study, plan, protocol)
        rows.sort(key=lambda row: recovery.identity(study, row))
        np.testing.assert_equal(rows, recovery.read(result / 'episodes.json'))
        recorded = {tuple(r['case']): {k: v for k, v in r.items() if k != 'case'}
                    for r in recovery.read(result / 'provenance.json')}
        np.testing.assert_equal(provenance, recorded)
        summaries[study] = summary
    if (protocols['phase']['models'] != rules['models'] or
            summaries['phase']['protocol_sha256'] != rules['development_protocol_sha256']):
        raise ValueError('Prospective development model identity differs')
    history = []
    for index in range(1, attempt):
        directory = ATTEMPTS / f'attempt_{index}'
        verify_manifest(directory)
        history.append(recovery.read(directory / 'terminal.json'))
    allocation = validate_attempt_history(history, attempt)
    output = ATTEMPTS / f'attempt_{attempt}'
    if output.exists():
        raise FileExistsError('Preserve previous selection and attempt data')
    selected = select_development_family(summaries['phase']['results'], summaries['phase']['independent_initials'])
    choice = selected['selected_family']
    output.mkdir(parents=True)
    save(output / 'selection.json', dict(**selected, **allocation, recovered_completion=evidence,
        common_profiles=list(profile_names(choice)) if choice else [],
        rules_sha256=digest(RULES / 'plan.json'), recovery_admission_rules_sha256=digest(GUARD / 'plan.json'),
        development_summary_sha256=digest(recovery.RECOVERY / 'results/phase/summary.json'),
        calibration_execution_protocol_still_required=True, new_calibration_data_generated=False))
    print(json.dumps(dict(selected_family=choice, attempt=attempt, risk_certified=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('freeze')
    selector = sub.add_parser('select')
    selector.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze()
    else:
        select(args.attempt)
