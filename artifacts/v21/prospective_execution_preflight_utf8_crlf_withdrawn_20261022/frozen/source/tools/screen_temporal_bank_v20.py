"""Screen fixed V20 banks on every archived V19 harmful-run admission state.

This is outcome-selected development diagnosis, not a closed-loop improvement
experiment or risk certificate. All candidates and all cases are retained.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from coopgate.disturbed_advantage_v19 import disturbed_action_difference
from coopgate.fast_v8 import parameters
from coopgate.merge_v4 import MergeConfig
from coopgate.temporal_portfolio_v20 import BANKS, BANK_STREAM, temporal_stress_bank
from probe_disturbed_gate_v19 import digest, save
from start_final_v19 import require_committed, verify_manifest
from study_gate_v19 import read_protocol

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/v19/calibration_harm_analysis'
PLAN = ROOT / 'artifacts/v20/temporal_screen_plan'
OUTPUT = ROOT / 'artifacts/v20/temporal_screen'


def bank_record(kind):
    names, tapes = temporal_stress_bank(kind)
    return dict(names=names, paths=len(names), float64_tapes_sha256=hashlib.sha256(tapes.tobytes()).hexdigest())


def prepare():
    if PLAN.exists() or OUTPUT.exists():
        raise FileExistsError('Preserve prior screen plans/results')
    verify_manifest(SOURCE)
    _, old = read_protocol(ROOT / 'artifacts/v19/risk_validation')
    events = json.loads((SOURCE / 'admission_events.json').read_text())
    sources = dict(old['source_sha256'])
    for name in ('src/coopgate/temporal_portfolio_v20.py', 'tools/screen_temporal_bank_v20.py',
                 'tests/test_temporal_portfolio_v20.py', 'tools/start_final_v19.py'):
        sources[name] = digest(ROOT / name)
    plan = dict(purpose='Post-calibration fixed-state temporal-bank development screen',
                outcome_selected=True, cases=len(events), actual_sumo_runs=0,
                independent_risk_samples=0, final_confirmation_outcomes_read=False,
                source_events_sha256=digest(SOURCE / 'admission_events.json'),
                source_manifest_sha256=digest(SOURCE / 'sha256.json'),
                bank_stream=BANK_STREAM, banks={kind: bank_record(kind) for kind in BANKS},
                admission_tolerance=1e-10, source_sha256=sources,
                scope='Recheck all 55 previously admitted actions from all 46 harmful calibration runs. Report rejection of positive and nonpositive hindsight steps separately. These selected nonpositive steps do not estimate population benefit retention. New closed-loop development data and independent risk calibration remain necessary.')
    PLAN.mkdir(parents=True)
    save(PLAN / 'plan.json', plan)
    for source in sources:
        target = PLAN / 'source' / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source, target)
    save(PLAN / 'sha256.json', {p.relative_to(PLAN).as_posix(): digest(p)
                               for p in PLAN.rglob('*') if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps(dict(cases=len(events), banks={k: v['paths'] for k, v in plan['banks'].items()},
                          plan_sha256=digest(PLAN / 'plan.json'))), flush=True)


def run():
    if OUTPUT.exists():
        raise FileExistsError('Never overwrite a screen')
    verify_manifest(PLAN)
    plan = json.loads((PLAN / 'plan.json').read_text())
    require_committed([PLAN / 'plan.json', PLAN / 'sha256.json', *[ROOT / p for p in plan['source_sha256']]])
    for path, expected in plan['source_sha256'].items():
        if digest(ROOT / path) != expected or digest(PLAN / 'source' / path) != expected:
            raise ValueError('Frozen source changed: ' + path)
    if digest(SOURCE / 'admission_events.json') != plan['source_events_sha256']:
        raise ValueError('Selected states changed')
    names, tapes = temporal_stress_bank('combined')
    bank_indices = {}
    for kind in BANKS:
        current = json.loads(json.dumps(bank_record(kind)))
        if current != plan['banks'][kind]:
            raise ValueError('Frozen bank changed')
        bank_indices[kind] = [names.index(name) for name in current['names']]
    events = json.loads((SOURCE / 'admission_events.json').read_text())
    OUTPUT.mkdir()
    packed, started = parameters(MergeConfig()), time.perf_counter()
    reports = []
    for index, event in enumerate(events):
        state, previous, command = (np.asarray(event[key]) for key in ('state_before', 'previous_command', 'chosen_command'))
        differences, errors = [], []
        for tape_index, tape in enumerate(tapes):
            try:
                difference, _, _ = disturbed_action_difference(state, previous, event['step'], command, tape, packed)
                differences.append(difference.tolist())
                errors.append(None)
            except RuntimeError as error:
                differences.append(None)
                errors.append(str(error))
        if any(errors[:17]):
            raise ValueError('Previously certified forecast can no longer be reconstructed')
        np.testing.assert_allclose(np.asarray(differences[:17]).max(axis=0), event['bank_componentwise_max'], atol=1e-9, rtol=0)
        rejected = {kind: any(errors[j] is not None or np.any(np.asarray(differences[j]) > 1e-10)
                              for j in ids) for kind, ids in bank_indices.items()}
        result = dict(event_index=index, case_index=event['case_index'], step=event['step'],
                      controller=event['controller'], hindsight_positive=event['positive_hindsight_residual'],
                      indirect_only=event['harm_only_on_currently_unchanged_vehicle_commands'],
                      rejected=rejected, differences=differences, forecast_errors=errors)
        save(OUTPUT / f'event_{index}.json', result)
        reports.append(result)
        print(json.dumps(dict(completed_events=index + 1, total_events=len(events), elapsed_s=time.perf_counter() - started)), flush=True)
    summary = {}
    for kind in BANKS:
        counts = Counter()
        by_model = {}
        for row in reports:
            model = row['controller']
            count = by_model.setdefault(model, Counter())
            key = ('positive' if row['hindsight_positive'] else 'nonpositive') + ('_rejected' if row['rejected'][kind] else '_retained')
            counts[key] += 1
            count[key] += 1
            if row['indirect_only']:
                counts['indirect_rejected' if row['rejected'][kind] else 'indirect_retained'] += 1
        summary[kind] = dict(counts=counts, per_model=by_model)
    report = dict(plan_sha256=digest(PLAN / 'plan.json'), events=len(reports), results=summary,
                  actual_sumo_runs=0, independent_risk_samples=0, risk_certified=False,
                  final_confirmation_outcomes_read=False, scope=plan['scope'])
    save(OUTPUT / 'summary.json', report)
    save(OUTPUT / 'sha256.json', {p.relative_to(OUTPUT).as_posix(): digest(p)
                                 for p in OUTPUT.rglob('*') if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'run'))
    args = parser.parse_args()
    (prepare if args.command == 'prepare' else run)()
