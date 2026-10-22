"""Select the unchanged V21 family after complete, project-scoped execution checks.

The old selector conservatively rejected every Python process on the computer.
Unrelated projects may run concurrently. This entry retains all completion,
source, provenance and statistical checks while restricting liveness to this
repository, its virtual environment, known study commands and descendants.
"""
import argparse
import json
import os
import shutil

import numpy as np

import continue_v21_20261022 as completion
from prepare_prospective_validation_v21 import ATTEMPTS, RULES, read_rules
from probe_disturbed_gate_v19 import digest, save
import recover_interrupted_v21 as prior
from select_recovered_v21 import GUARD
from start_final_v19 import require_committed, verify_manifest
from coopgate.prospective_validation_v21 import profile_names, select_development_family, validate_attempt_history


ROOT = completion.ROOT
FROZEN = ROOT / 'artifacts/v21/project_scoped_selection_rules'
SOURCES = ('tools/select_completed_v21_scoped.py', 'tests/test_project_scoped_selection_v21.py')
KNOWN_COMMANDS = ('tools/continue_v21_20261022.py', 'tools/recover_interrupted_v21.py',
    'tools/advance_phase_repair_v21.py', 'tools/study_phase_repair_v21.py',
    'tools/advance_matched_transfer_v21.py', 'tools/study_matched_transfer_v21.py')


def project_processes(processes, root=ROOT):
    normalized = str(root).replace('\\', '/').rstrip('/').lower()
    parent = normalized.rsplit('/', 1)[0]
    anchors = (normalized+'/', parent+'/.venv/')
    relevant = {p['ProcessId'] for p in processes if any(anchor in (p.get('CommandLine') or '').replace('\\', '/').lower()
                for anchor in (*anchors, *KNOWN_COMMANDS))}
    while True:
        children = {p['ProcessId'] for p in processes if p['ParentProcessId'] in relevant}
        if children <= relevant:
            break
        relevant |= children
    return [p for p in processes if p['ProcessId'] in relevant]


def require_project_stopped(processes, own_pid=None, parent_pid=None):
    relevant = project_processes(processes)
    prior.assert_no_existing_workers(relevant, own_pid=own_pid, parent_pid=parent_pid)
    return dict(project_processes=relevant, unrelated_process_count=len(processes)-len(relevant))


def freeze():
    if FROZEN.exists():
        raise FileExistsError('Preserve prior scoped selection rules')
    continuation, _, _ = completion.inputs()
    read_rules()
    verify_manifest(GUARD)
    require_committed([ROOT/name for name in SOURCES])
    FROZEN.mkdir(parents=True)
    plan = dict(utc=prior.utc(), continuation_plan_sha256=digest(completion.FROZEN/'plan.json'),
        prospective_rules_sha256=digest(RULES/'plan.json'), prior_admission_rules_sha256=digest(GUARD/'plan.json'),
        change='Exclude unrelated native Python/SUMO processes from the final liveness check',
        unchanged='Actual supervisor/worker exits, complete source identity, full independent review, exact schedules, provenance, whole-family selection and all statistical thresholds',
        effectiveness_outcomes_read=False, new_calibration_data_generated=False,
        source_sha256={name:digest(ROOT/name) for name in SOURCES},
        expected_studies=['phase', 'transfer'], source_completion_plan_has_changed=False)
    save(FROZEN/'plan.json', plan)
    for name in SOURCES:
        target = FROZEN/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
    save(FROZEN/'sha256.json', prior.manifest(FROZEN))
    print(json.dumps(dict(plan_sha256=digest(FROZEN/'plan.json'), statistical_rules_changed=False)), flush=True)


def inputs():
    verify_manifest(FROZEN)
    plan = prior.read(FROZEN/'plan.json')
    require_committed([FROZEN/'plan.json', FROZEN/'sha256.json', *[ROOT/name for name in SOURCES]])
    for name, expected in plan['source_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Scoped selection implementation changed')
    continuation, old, protocols = completion.inputs()
    rules = read_rules()
    verify_manifest(GUARD)
    if (digest(completion.FROZEN/'plan.json') != plan['continuation_plan_sha256']
            or digest(RULES/'plan.json') != plan['prospective_rules_sha256']
            or digest(GUARD/'plan.json') != plan['prior_admission_rules_sha256']):
        raise ValueError('Frozen completion or selection rules changed')
    return plan, continuation, old, protocols, rules


def select():
    scope, plan, old, protocols, rules = inputs()
    receipt = prior.read(completion.EXECUTION/'supervisor_exit.json')
    launch = prior.read(completion.EXECUTION/'watcher_launch.json')
    if (receipt['exit_code'] != 0 or receipt['supervisor_pid'] != launch['supervisor_pid']
            or receipt['plan_sha256'] != digest(completion.FROZEN/'plan.json')
            or launch['plan_sha256'] != receipt['plan_sha256']):
        raise ValueError('The supervised continuation has not completed successfully')
    evidence = dict(resume=completion.require_stage('resume', sum(bool(g['missing_cases']) for g in plan['groups'])),
        review=completion.require_stage('review', 2), supervisor_exit_sha256=digest(completion.EXECUTION/'supervisor_exit.json'))
    processes = prior.native_snapshot()
    process_evidence = require_project_stopped(processes)
    finished = prior.read(completion.EXECUTION/'completed.json')
    summaries = {}
    for study, protocol in protocols.items():
        directory = completion.OUTPUT/'results'/study
        verify_manifest(directory)
        if digest(directory/'summary.json') != finished['summary_sha256'][study]:
            raise ValueError('Different completed summary')
        summary = prior.read(directory/'summary.json')
        if (summary['actual_sumo_runs'] != protocol['spec']['expected_actual_sumo_runs']
                or summary['independent_review_counts']['physical_traces'] != summary['actual_sumo_runs']
                or summary['protocol_sha256'] != old['studies'][study]['protocol_sha256']):
            raise ValueError('Complete independent review identity differs')
        rows, sources = completion.assembled(study, plan, old, protocol)
        rows.sort(key=lambda row:prior.identity(study, row))
        np.testing.assert_equal(rows, prior.read(directory/'episodes.json'))
        np.testing.assert_equal({tuple(r['case']):{k:v for k,v in r.items() if k!='case'}
                                 for r in prior.read(directory/'provenance.json')}, sources)
        summaries[study] = summary
    if (protocols['phase']['models'] != rules['models']
            or summaries['phase']['protocol_sha256'] != rules['development_protocol_sha256']):
        raise ValueError('Different development family')
    output = ATTEMPTS/'attempt_1'
    if output.exists():
        raise FileExistsError('Preserve prior selection')
    allocation = validate_attempt_history([], 1)
    choice = select_development_family(summaries['phase']['results'], summaries['phase']['independent_initials'])
    output.mkdir(parents=True)
    save(output/'selection.json', dict(**choice, **allocation, continuation_completion=evidence,
        continuation_plan_sha256=digest(completion.FROZEN/'plan.json'), rules_sha256=digest(RULES/'plan.json'),
        scoped_process_rules_sha256=digest(FROZEN/'plan.json'), process_check=process_evidence,
        common_profiles=list(profile_names(choice['selected_family'])) if choice['selected_family'] else [],
        development_summary_sha256=digest(completion.OUTPUT/'results/phase/summary.json'),
        calibration_execution_protocol_still_required=True, new_calibration_data_generated=False))
    print(json.dumps(dict(selected_family=choice['selected_family'], risk_certified=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['freeze', 'select'])
    globals()[parser.parse_args().command]()
