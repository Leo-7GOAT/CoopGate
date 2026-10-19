"""Independently reconstruct completed repair choices and selected branches."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import time

import numpy as np

from coopgate.disturbance_v9 import disturbance_tape
from coopgate.disturbed_advantage_v19 import disturbed_action_difference
from coopgate.fast_v8 import parameters
from coopgate.marl_v12 import project_joint
from coopgate.merge_v4 import MergeConfig
from coopgate.robust_v9 import robust_certificate
from coopgate.temporal_portfolio_v20 import temporal_stress_bank
from probe_disturbed_gate_v19 import digest, save
from start_final_v19 import require_committed, verify_manifest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/v20/action_repair_screen'
SOURCE_PLAN = ROOT / 'artifacts/v20/action_repair_screen_plan'
EVENTS = ROOT / 'artifacts/v19/calibration_harm_analysis/admission_events.json'
PLAN = ROOT / 'artifacts/v20/action_repair_review_plan'
OUTPUT = ROOT / 'artifacts/v20/action_repair_review'


def prepare():
    if PLAN.exists() or OUTPUT.exists():
        raise FileExistsError('Preserve prior audit plans')
    verify_manifest(SOURCE_PLAN)
    PLAN.mkdir(parents=True)
    record = dict(source_plan_sha256=digest(SOURCE_PLAN / 'plan.json'), events_sha256=digest(EVENTS),
                  reviewer_sha256=digest(__file__), source_path=Path(__file__).resolve().relative_to(ROOT).as_posix(),
                  scope='Reconstruct all candidate commands, choices, fallback counts and selected realized branches; no new SUMO executions or risk samples.')
    save(PLAN / 'plan.json', record)
    shutil.copy2(__file__, PLAN / Path(__file__).name)
    print(json.dumps(record), flush=True)


def run():
    if OUTPUT.exists():
        raise FileExistsError('Preserve prior review')
    plan = json.loads((PLAN / 'plan.json').read_text())
    require_committed([PLAN / 'plan.json', PLAN / Path(__file__).name, Path(__file__).resolve()])
    if digest(__file__) != plan['reviewer_sha256'] or digest(PLAN / Path(__file__).name) != plan['reviewer_sha256']:
        raise ValueError('Frozen reviewer changed')
    verify_manifest(SOURCE_PLAN)
    verify_manifest(SOURCE)  # Exists only after the entire diagnostic finishes.
    if digest(SOURCE_PLAN / 'plan.json') != plan['source_plan_sha256'] or digest(EVENTS) != plan['events_sha256']:
        raise ValueError('Diagnostic identity changed')
    original_plan = json.loads((SOURCE_PLAN / 'plan.json').read_text())
    for path, expected in original_plan['source_sha256'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('A frozen forecast dependency changed: ' + path)
    events = json.loads(EVENTS.read_text())
    names, _ = temporal_stress_bank('combined')
    indices = {kind: [names.index(name) for name in temporal_stress_bank(kind)[0]] for kind in original_plan['banks']}
    packed = parameters(MergeConfig())
    counts = {kind: Counter() for kind in indices}
    model_counts = {kind: {} for kind in indices}
    decisions = []
    branches_checked = original_identities = candidate_count = 0
    started = time.perf_counter()
    for number, event in enumerate(events):
        prediction = json.loads((SOURCE / f'prediction_{number}.json').read_text())
        diagnosis = json.loads((SOURCE / f'diagnosis_{number}.json').read_text())
        if prediction['event_index'] != number or diagnosis['event_index'] != number or diagnosis['case_index'] != event['case_index']:
            raise ValueError('Case identity mismatch')
        state, previous, learned, baseline = (np.asarray(event[key]) for key in
            ('state_before', 'previous_command', 'chosen_command', 'baseline_command'))
        expected_candidates, seen = [], [baseline]
        for weight in original_plan['weights']:
            command = learned.copy() if weight == 1. else project_joint(
                state, previous, baseline + weight * (learned - baseline), packed, .5)[0]
            if not robust_certificate(state, previous, command, packed, .5):
                raise ValueError('An attenuated command violates the common physical contract')
            if any(np.array_equal(command, old) for old in seen):
                continue
            seen.append(command.copy())
            expected_candidates.append((weight, command))
        if len(expected_candidates) != len(prediction['predictions']):
            raise ValueError('Candidate coverage differs')
        for (weight, command), candidate in zip(expected_candidates, prediction['predictions']):
            if weight != candidate['weight']:
                raise ValueError('Weight order changed')
            np.testing.assert_array_equal(command, candidate['command'])
            if len(candidate['differences']) != len(names) or len(candidate['forecast_errors']) != len(names):
                raise ValueError('Forecast path coverage differs')
            candidate_count += 1
        selected = {}
        for kind, paths in indices.items():
            valid = []
            for candidate_index, candidate in enumerate(prediction['predictions']):
                if any(candidate['forecast_errors'][path] is not None for path in paths):
                    continue
                values = np.asarray([candidate['differences'][path] for path in paths])
                if values.shape != (len(paths), 3, 2) or not np.isfinite(values).all():
                    raise ValueError('Missing or nonfinite certified component')
                gain = -float(np.asarray(candidate['differences'][0])[:, 0].mean())
                if values.max() <= 1e-10 and gain > 1e-9:
                    valid.append((-gain, -candidate['weight'], candidate_index))
            selected[kind] = sorted(valid)[0][2] if valid else None
        if selected != prediction['selected'] or selected != diagnosis['selected']:
            raise ValueError('Independent choice differs')
        tape = disturbance_tape(event['seed'], event['disturbance'], .5, 250)
        recomputed = {}
        for candidate_index in sorted({i for i in selected.values() if i is not None}):
            candidate = prediction['predictions'][candidate_index]
            old_branch = diagnosis['post_selection_realized_branches'][str(candidate_index)]
            try:
                delta, _, _ = disturbed_action_difference(state, previous, event['step'],
                    np.asarray(candidate['command']), tape, packed)
            except RuntimeError as error:
                if old_branch.get('available') or old_branch.get('error') != str(error):
                    raise ValueError('Realized branch exception differs') from error
                recomputed[candidate_index] = None
                continue
            if not old_branch['available']:
                raise ValueError('Previously unavailable branch now succeeds')
            np.testing.assert_allclose(delta, old_branch['difference'], atol=1e-9, rtol=0)
            if bool(np.any(delta > 1e-9)) != old_branch['positive']:
                raise ValueError('Realized branch harm classification differs')
            if candidate['weight'] == 1.:
                np.testing.assert_allclose(delta, event['hindsight_difference'], atol=1e-9, rtol=0)
                original_identities += 1
            recomputed[candidate_index] = delta
            branches_checked += 1
        for kind, choice in selected.items():
            model = model_counts[kind].setdefault(event['controller'], Counter())
            restored = bool(diagnosis['original_rejected'][kind] and choice is not None)
            if choice is None:
                label = 'fallback'
            elif recomputed[choice] is None:
                label = 'selected_branch_unavailable'
            else:
                delta = recomputed[choice]
                prefix = 'repair' if prediction['predictions'][choice]['weight'] < 1. else 'original'
                label = prefix + ('_positive_branch' if np.any(delta > 1e-9) else '_nonpositive_branch')
                if restored and not np.any(delta > 1e-9):
                    qualifier = 'restored_with_strict_realized_team_gain' if delta[:, 0].mean() < -1e-9 else 'restored_with_zero_realized_team_gain'
                    counts[kind][qualifier] += 1
                    model[qualifier] += 1
            counts[kind][label] += 1
            model[label] += 1
            if restored:
                counts[kind]['original_rejection_replaced_by_repair'] += 1
                model['original_rejection_replaced_by_repair'] += 1
        decisions.append(dict(event_index=number, selected=selected, controller=event['controller']))
        if number % 8 == 0:
            print(json.dumps(dict(verified_events=number + 1, elapsed_s=time.perf_counter() - started)), flush=True)
    source_report = json.loads((SOURCE / 'summary.json').read_text())
    for kind in indices:
        for key, count in source_report['results'][kind].items():
            if counts[kind][key] != count:
                raise ValueError('Reconstructed aggregate differs from the diagnostic')
    report = dict(events=len(events), candidate_commands_checked=candidate_count,
                  selected_realized_branches_recomputed=branches_checked, original_hindsight_identities_checked=original_identities,
                  all_choices_match=True, all_recomputed_branches_match=True, counts=counts, per_model=model_counts,
                  decisions=decisions, source_manifest_sha256=digest(SOURCE / 'sha256.json'),
                  plan_sha256=digest(PLAN / 'plan.json'), actual_sumo_runs=0, independent_risk_samples=0,
                  risk_certified=False, final_confirmation_outcomes_read=False, scope=original_plan['diagnostic_scope'])
    OUTPUT.mkdir()
    save(OUTPUT / 'audit.json', report)
    shutil.copy2(__file__, OUTPUT / Path(__file__).name)
    save(OUTPUT / 'sha256.json', {p.relative_to(OUTPUT).as_posix(): digest(p)
                                 for p in OUTPUT.rglob('*') if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'run'))
    args = parser.parse_args()
    (prepare if args.command == 'prepare' else run)()
