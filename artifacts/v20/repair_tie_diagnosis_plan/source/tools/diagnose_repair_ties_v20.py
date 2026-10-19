"""Post-hoc mechanism and exact-nominal-tie screen for every repair harm.

Recorded future noise is diagnostic only. No controller or frozen study changes.
This screen cannot establish closed-loop improvement or population risk.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from coopgate.advantage_gate_v18 import action_then_baseline, baseline_suffix
from coopgate.disturbance_v9 import disturbance_tape
from coopgate.disturbed_advantage_v19 import disturbed_action_difference
from coopgate.fast_v8 import parameters
from coopgate.merge_v4 import MergeConfig
from coopgate.temporal_portfolio_v20 import temporal_stress_bank
from diagnose_calibration_harm_v19 import harmful, identity
from diagnose_temporal_harm_v20 import hindsight
from probe_disturbed_gate_v19 import digest, save
from review_repair_closed_loop_v20 import check_forecast_record, check_selected, reconstruct_candidates
from start_final_v19 import require_committed, verify_manifest
from study_gate_v19 import read_protocol

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'artifacts/v20/temporal_repair_extension'
PARTNER = ROOT / 'artifacts/v20/temporal_development'
PLAN = ROOT / 'artifacts/v20/repair_tie_diagnosis_plan'
OUTPUT = ROOT / 'artifacts/v20/repair_tie_diagnosis'


def select_cases(rows, baseline_rows):
    lookup = {identity(row): row for row in baseline_rows}
    if len(lookup) != len(baseline_rows) or len({identity(row) for row in rows}) != len(rows):
        raise ValueError('Duplicate archived outcomes')
    return [dict(initial_index=row['initial_index'], disturbance=row['disturbance'], controller=row['controller'])
            for row in rows if harmful(row, lookup[row['initial_index'], row['disturbance'], 'baseline'])]


def exact_gain_ties(records, selected_difference):
    target = float(np.asarray(selected_difference)[:, 0].mean())
    result = []
    for record in records:
        delta = np.asarray(record['nominal_difference'])
        if delta.shape != (3, 2) or not np.isfinite(delta).all():
            raise ValueError('Invalid nominal forecast')
        if np.all(delta <= 1e-10) and float(delta[:, 0].mean()) == target and target < -1e-9:
            result.append(record)
    return result


def deviation_key(record, baseline):
    distance = np.abs(np.asarray(record['command']) - baseline)
    return float(distance.max()), float(distance.sum()), -record['weight'], record['composition']


def prepare():
    if PLAN.exists() or OUTPUT.exists():
        raise FileExistsError('Preserve the existing diagnostic plan and results')
    for directory in (STUDY, PARTNER):
        verify_manifest(directory)
    _, protocol = read_protocol(STUDY)
    rows = json.loads((STUDY / 'episodes.json').read_text())
    baseline_rows = json.loads((PARTNER / 'episodes.json').read_text())
    cases = select_cases(rows, baseline_rows)
    sources = dict(protocol['source_sha256'])
    for name in ('tools/diagnose_repair_ties_v20.py', 'tools/diagnose_temporal_harm_v20.py',
                 'tools/diagnose_calibration_harm_v19.py', 'tools/review_repair_closed_loop_v20.py',
                 'tests/test_repair_tie_diagnosis_v20.py'):
        sources[name] = digest(ROOT / name)
    plan = dict(purpose='Diagnose every completed repair harm and screen baseline-nearer exact nominal-gain ties',
        cases=cases, source_sha256=sources, study_manifest_sha256=digest(STUDY / 'sha256.json'),
        partner_manifest_sha256=digest(PARTNER / 'sha256.json'),
        selection='All harmful runs; inspect every admitted decision, not only positive hindsight residuals',
        outcome_dependent_selection=True, additional_sumo_runs=0, additional_independent_initials=0,
        online_future_noise=False, risk_certification=False, controller_changed=False,
        tie_rule='Exactly equal floating nominal mean cost; nominal all-component checks unchanged; entire 33-path bank checked for every tie; order by actual L-infinity then L1 command deviation, followed by original weight/code.',
        scope='One-action-then-baseline branches at selected archived states; no closed-loop efficacy claim')
    PLAN.mkdir(parents=True)
    save(PLAN / 'plan.json', plan)
    for name in sources:
        target = PLAN / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    save(PLAN / 'sha256.json', {p.relative_to(PLAN).as_posix(): digest(p) for p in PLAN.rglob('*')
                               if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps(dict(harmful_runs=len(cases), plan_sha256=digest(PLAN / 'plan.json'))), flush=True)


def run():
    torch.set_num_threads(1)
    if OUTPUT.exists():
        raise FileExistsError('Preserve existing diagnostic output')
    for directory in (PLAN, STUDY, PARTNER):
        verify_manifest(directory)
    plan = json.loads((PLAN / 'plan.json').read_text())
    require_committed([PLAN / 'plan.json', PLAN / 'sha256.json', *[ROOT / p for p in plan['source_sha256']]])
    for name, expected in plan['source_sha256'].items():
        if digest(ROOT / name) != expected or digest(PLAN / 'source' / name) != expected:
            raise ValueError('Diagnostic source changed: ' + name)
    for directory, key in ((STUDY, 'study_manifest_sha256'), (PARTNER, 'partner_manifest_sha256')):
        if digest(directory / 'sha256.json') != plan[key]:
            raise ValueError('Completed study changed')
    _, protocol = read_protocol(STUDY)
    rows = json.loads((STUDY / 'episodes.json').read_text())
    old = json.loads((PARTNER / 'episodes.json').read_text())
    if select_cases(rows, old) != plan['cases']:
        raise ValueError('Complete harm selection changed')
    lookup = {identity(row): row for row in rows + old}
    config = MergeConfig()
    packed = parameters(config)
    _, tapes = temporal_stress_bank('sampled')
    if len(tapes) != 33:
        raise ValueError('Unexpected path bank')
    OUTPUT.mkdir(parents=True)
    reports, counts = [], Counter()
    for number, case in enumerate(plan['cases']):
        row = lookup[case['initial_index'], case['disturbance'], case['controller']]
        base_row = lookup[case['initial_index'], case['disturbance'], 'baseline']
        trace_path = STUDY / f"shard_{row['initial_index'] % protocol['spec']['shards']}/trace_{row['initial_index']}_{row['disturbance']}_{row['controller']}.json"
        trace = json.loads(trace_path.read_text())
        mechanism = hindsight(row, trace, base_row, protocol['spec']['methods'][case['controller']])
        if not mechanism['available']:
            raise ValueError('Completed physical run lacks a valid hindsight decomposition')
        tape = disturbance_tape(row['seed'], row['disturbance'], .5, round(config.horizon / config.dt))
        save(OUTPUT / f'mechanism_{number}.json', mechanism)
        counts['positive_admission_residuals'] += mechanism['positive_admission_steps']
        counts['positive_fallback_residuals'] += mechanism['positive_fallback_steps']
        counts['unchanged_command_only_positive_admissions'] += mechanism['unchanged_command_only_positive_admissions']
        state, previous, decisions = np.asarray(row['initial']), np.zeros(3), []
        for step, item in enumerate(trace):
            if item['portfolio_joint_action_admitted']:
                base, baseline, _, _ = baseline_suffix(state, previous, step, packed)
                np.testing.assert_array_equal(baseline, item['baseline_command'])
                candidates, original_count = reconstruct_candidates(
                    state, previous, np.asarray(item['actor_proposal']), baseline, packed, True)
                check_selected(item, baseline, candidates, original_count)
                records = []
                for code, weight, command in candidates:
                    delta = action_then_baseline(state, previous, step, command, packed)[0] - base
                    records.append(dict(composition=code, weight=weight, command=command.tolist(),
                                        nominal_difference=delta.tolist()))
                ties = exact_gain_ties(records, item['selected_bank_differences'][0])
                counts['admitted_states_screened'] += 1
                counts['candidate_nominal_forecasts'] += len(records)
                selected = None
                for record in ties:
                    command = np.asarray(record['command'])
                    deltas = [np.asarray(record['nominal_difference'])] + [
                        disturbed_action_difference(state, previous, step, command, path, packed)[0]
                        for path in tapes[1:]]
                    bank = np.asarray(deltas)
                    record['bank_differences'] = bank.tolist()
                    record['all_33_paths_pass'] = bool(np.isfinite(bank).all() and np.all(bank <= 1e-10))
                    record['deviation_key'] = list(deviation_key(record, baseline))
                    realized = disturbed_action_difference(state, previous, step, command, tape, packed)[0]
                    record['realized_branch_difference'] = realized.tolist()
                    record['realized_branch_harmful'] = bool(np.any(realized > 1e-9))
                    counts['tie_candidates_full_bank_checked'] += 1
                    if record['composition'] == item['composition'] and record['weight'] == item['repair_weight']:
                        np.testing.assert_array_equal(command, item['command'])
                        check_forecast_record(bank, item['selected_bank_differences'])
                        np.testing.assert_allclose(realized, mechanism['steps'][step]['hindsight_bellman_difference'], atol=1e-9, rtol=0)
                        selected = record
                if selected is None or not selected['all_33_paths_pass']:
                    raise ValueError('Original selected action missing from exact gain ties')
                accepted = sorted((r for r in ties if r['all_33_paths_pass']), key=lambda r: deviation_key(r, baseline))
                preferred = accepted[0]
                changed = preferred['command'] != selected['command']
                improvement = selected['realized_branch_harmful'] and not preferred['realized_branch_harmful']
                worsening = not selected['realized_branch_harmful'] and preferred['realized_branch_harmful']
                counts['states_with_multiple_admissible_exact_ties'] += int(len(accepted) > 1)
                counts['preferred_command_changes'] += int(changed)
                counts['positive_residuals_removed_locally'] += int(improvement)
                counts['new_positive_residuals_locally'] += int(worsening)
                decisions.append(dict(step=step, original=selected, preferred=preferred,
                    candidate_nominal_records=records, exact_ties=ties, preferred_command_changes=changed,
                    removes_positive_residual_locally=bool(improvement), creates_positive_residual_locally=bool(worsening)))
            state, previous = np.asarray(item['x']), np.asarray(item['command'])
        report = dict(case=case, trace_sha256=digest(trace_path), decisions=decisions,
                      mechanism={k: v for k, v in mechanism.items() if k != 'steps'})
        save(OUTPUT / f'case_{number}.json', report)
        reports.append(dict(case=case, admitted_decisions=len(decisions),
                            changed_decisions=sum(d['preferred_command_changes'] for d in decisions)))
        print(json.dumps(dict(completed_cases=number + 1, expected_cases=len(plan['cases']))), flush=True)
    report = dict(counts=dict(counts), cases=reports, harmful_runs=len(reports),
        all_original_admitted_forecasts_and_hindsight_identities_match=True,
        additional_sumo_runs=0, additional_independent_initials=0, outcome_dependent_selection=True,
        online_future_noise=False, closed_loop_improvement_verified=False, risk_certified=False,
        controller_changed=False, plan_sha256=digest(PLAN / 'plan.json'),
        study_manifest_sha256=digest(STUDY / 'sha256.json'), scope=plan['scope'])
    save(OUTPUT / 'audit.json', report)
    save(OUTPUT / 'sha256.json', {p.relative_to(OUTPUT).as_posix(): digest(p) for p in OUTPUT.rglob('*')
                                 if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps({k:v for k,v in report.items() if k != 'cases'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'run'))
    (prepare if parser.parse_args().command == 'prepare' else run)()
