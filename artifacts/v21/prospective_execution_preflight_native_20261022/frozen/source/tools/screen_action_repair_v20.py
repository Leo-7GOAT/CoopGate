"""Diagnostic repair of all selected-state commands, without future-noise access.

Selection uses fixed forecast banks only. A separate post-selection branch
calculation uses the recorded realized tape to diagnose the chosen action.
Neither operation is a new closed-loop SUMO experiment or risk calibration.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import time

import numpy as np

from coopgate.action_repair_v20 import REPAIR_WEIGHTS, attenuated_candidates
from coopgate.disturbance_v9 import disturbance_tape
from coopgate.disturbed_advantage_v19 import disturbed_action_difference
from coopgate.fast_v8 import parameters
from coopgate.merge_v4 import MergeConfig
from coopgate.temporal_portfolio_v20 import BANKS, temporal_stress_bank
from probe_disturbed_gate_v19 import digest, save
from start_final_v19 import require_committed, verify_manifest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/v19/calibration_harm_analysis'
SCREEN = ROOT / 'artifacts/v20/temporal_screen'
PLAN = ROOT / 'artifacts/v20/action_repair_screen_plan'
OUTPUT = ROOT / 'artifacts/v20/action_repair_screen'


def choose(predictions, bank_indices):
    """No episode seed, realized tape, hindsight label or outcome input."""
    selected = {}
    for kind, indices in bank_indices.items():
        accepted = []
        for index, record in enumerate(predictions):
            if any(record['forecast_errors'][i] is not None for i in indices):
                continue
            differences = np.asarray([record['differences'][i] for i in indices], dtype=float)
            if not np.isfinite(differences).all():
                raise ValueError('Nonfinite forecast without an explicit exception')
            nominal_gain = float(-np.asarray(record['differences'][0])[:, 0].mean())
            if np.all(differences <= 1e-10) and nominal_gain > 1e-9:
                accepted.append((-nominal_gain, -record['weight'], index))
        selected[kind] = min(accepted)[2] if accepted else None
    return selected


def prepare():
    if PLAN.exists() or OUTPUT.exists():
        raise FileExistsError('Keep prior repair evidence')
    verify_manifest(SCREEN)
    verify_manifest(SOURCE)
    old = json.loads((ROOT / 'artifacts/v20/temporal_screen_plan/plan.json').read_text())
    sources = dict(old['source_sha256'])
    for name in ('src/coopgate/action_repair_v20.py', 'tools/screen_action_repair_v20.py',
                 'tests/test_action_repair_v20.py', 'tests/test_action_repair_screen_v20.py'):
        sources[name] = digest(ROOT / name)
    events = json.loads((SOURCE / 'admission_events.json').read_text())
    plan = dict(purpose='Fixed-state learning-command attenuation development diagnosis',
                cases=len(events), weights=REPAIR_WEIGHTS, banks=BANKS, source_sha256=sources,
                event_sha256=digest(SOURCE / 'admission_events.json'), screen_sha256=digest(SCREEN / 'sha256.json'),
                selection='Highest nominal mean gain among original and physically reprojected attenuated commands that pass every path and every vehicle cost/time component; ties prefer the original/larger weight; otherwise baseline.',
                no_realized_future_in_selection=True, actual_sumo_runs=0, independent_risk_samples=0,
                final_confirmation_outcomes_read=False, outcome_selected_states=True,
                diagnostic_scope='The actual future tape is used only after candidate selection for a one-action-then-baseline counterfactual value difference at an old observed state. This is not a repaired-policy trajectory, adoption-rate estimate, or risk certificate.')
    PLAN.mkdir(parents=True)
    save(PLAN / 'plan.json', plan)
    for source in sources:
        target = PLAN / 'source' / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source, target)
    save(PLAN / 'sha256.json', {p.relative_to(PLAN).as_posix(): digest(p)
                               for p in PLAN.rglob('*') if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps(dict(cases=len(events), plan_sha256=digest(PLAN / 'plan.json'))), flush=True)


def run():
    if OUTPUT.exists():
        raise FileExistsError('Do not overwrite repair screening')
    verify_manifest(PLAN)
    verify_manifest(SCREEN)
    plan = json.loads((PLAN / 'plan.json').read_text())
    require_committed([PLAN / 'plan.json', PLAN / 'sha256.json', *[ROOT / p for p in plan['source_sha256']]])
    for path, expected in plan['source_sha256'].items():
        if digest(ROOT / path) != expected or digest(PLAN / 'source' / path) != expected:
            raise ValueError('Frozen repair source changed: ' + path)
    if digest(SOURCE / 'admission_events.json') != plan['event_sha256'] or digest(SCREEN / 'sha256.json') != plan['screen_sha256']:
        raise ValueError('Selected states or original forecasts changed')
    names, tapes = temporal_stress_bank('combined')
    indices = {kind: [names.index(name) for name in temporal_stress_bank(kind)[0]] for kind in BANKS}
    events = json.loads((SOURCE / 'admission_events.json').read_text())
    packed, started = parameters(MergeConfig()), time.perf_counter()
    reports = []
    OUTPUT.mkdir()
    for event_index, event in enumerate(events):
        state, previous, learned, baseline = (np.asarray(event[key]) for key in
            ('state_before', 'previous_command', 'chosen_command', 'baseline_command'))
        original = json.loads((SCREEN / f'event_{event_index}.json').read_text())
        if original['case_index'] != event['case_index'] or original['step'] != event['step']:
            raise ValueError('Original event identity mismatch')
        predictions = []
        for weight, command in attenuated_candidates(state, previous, learned, baseline, packed):
            if weight == 1.:
                differences, errors = original['differences'], original['forecast_errors']
            else:
                differences, errors = [], []
                for tape in tapes:
                    try:
                        delta, _, _ = disturbed_action_difference(state, previous, event['step'], command, tape, packed)
                        differences.append(delta.tolist())
                        errors.append(None)
                    except RuntimeError as error:
                        differences.append(None)
                        errors.append(str(error))
            predictions.append(dict(weight=weight, command=command.tolist(), differences=differences, forecast_errors=errors))
        selected = choose(predictions, indices)
        # Archive online-available decisions BEFORE creating a realized tape.
        save(OUTPUT / f'prediction_{event_index}.json', dict(event_index=event_index, predictions=predictions, selected=selected))
        realized = disturbance_tape(event['seed'], event['disturbance'], .5, 250)
        branches = {}
        for candidate in sorted({index for index in selected.values() if index is not None}):
            command = np.asarray(predictions[candidate]['command'])
            try:
                delta, _, _ = disturbed_action_difference(state, previous, event['step'], command, realized, packed)
                branches[str(candidate)] = dict(available=True, difference=delta.tolist(), positive=bool(np.any(delta > 1e-9)))
            except RuntimeError as error:
                branches[str(candidate)] = dict(available=False, error=str(error))
        result = dict(event_index=event_index, case_index=event['case_index'], controller=event['controller'],
                      old_hindsight_positive=event['positive_hindsight_residual'],
                      original_rejected=original['rejected'], selected=selected,
                      selected_weights={kind: predictions[index]['weight'] if index is not None else None for kind, index in selected.items()},
                      post_selection_realized_branches=branches)
        save(OUTPUT / f'diagnosis_{event_index}.json', result)
        reports.append(result)
        print(json.dumps(dict(completed_events=event_index + 1, total_events=len(events), elapsed_s=time.perf_counter() - started)), flush=True)
    counts = {kind: Counter() for kind in BANKS}
    by_model = {kind: {} for kind in BANKS}
    for row in reports:
        for kind in BANKS:
            counter = by_model[kind].setdefault(row['controller'], Counter())
            index = row['selected'][kind]
            if index is None:
                key = 'fallback'
            elif not row['post_selection_realized_branches'][str(index)]['available']:
                key = 'selected_branch_unavailable'
            else:
                positive = row['post_selection_realized_branches'][str(index)]['positive']
                key = ('repair' if row['selected_weights'][kind] < 1. else 'original') + ('_positive_branch' if positive else '_nonpositive_branch')
            counts[kind][key] += 1
            counter[key] += 1
            if row['original_rejected'][kind] and index is not None:
                counts[kind]['original_rejection_replaced_by_repair'] += 1
    report = dict(plan_sha256=digest(PLAN / 'plan.json'), events=len(reports), results=counts, per_model=by_model,
                  actual_sumo_runs=0, independent_risk_samples=0, risk_certified=False,
                  scope=plan['diagnostic_scope'], final_confirmation_outcomes_read=False)
    save(OUTPUT / 'summary.json', report)
    save(OUTPUT / 'sha256.json', {p.relative_to(OUTPUT).as_posix(): digest(p)
                                 for p in OUTPUT.rglob('*') if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'run'))
    args = parser.parse_args()
    (prepare if args.command == 'prepare' else run)()
