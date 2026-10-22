"""Aggregate completed hindsight replay evidence without new risk samples."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil

import numpy as np

from probe_disturbed_gate_v19 import digest, save
from start_final_v19 import verify_manifest

ROOT = Path(__file__).resolve().parents[1]
REPLAY = ROOT / 'artifacts/v20/temporal_harm_replay'
OUTPUT = ROOT / 'artifacts/v20/temporal_harm_analysis'


def run():
    if OUTPUT.exists():
        raise FileExistsError('Preserve the completed mechanism analysis')
    verify_manifest(REPLAY)
    audit = json.loads((REPLAY / 'audit.json').read_text())
    if not all(audit[k] for k in ('all_non_timing_metrics_match', 'all_physical_traces_checked', 'all_hindsight_identities_match')):
        raise ValueError('Complete replay verification is required')
    counts, by_bank, events = Counter(), defaultdict(Counter), []
    for case in audit['cases']:
        if case['controller'] == 'baseline':
            continue
        bank_name = case['controller'].rsplit('_', 1)[0]
        record = case['mechanism']
        counts['harmful_runs'] += 1
        counts['exactly_one_positive_admission'] += int(record['positive_admission_steps'] == 1)
        counts['positive_fallback_steps'] += record['positive_fallback_steps']
        by_bank[bank_name]['harmful_runs'] += 1
        by_bank[bank_name]['positive_only_on_unchanged_commands'] += record['unchanged_command_only_positive_admissions']
        detail_path = REPLAY / f'mechanism_{case["case_index"]}.json'
        detail = json.loads(detail_path.read_text())
        for step in detail['steps']:
            if not step['admitted'] or not step['positive_hindsight_residual']:
                continue
            delta, margin = np.asarray(step['hindsight_bellman_difference']), np.asarray(step['bank_componentwise_max'])
            positive = delta > 1e-9
            zero = positive & (np.abs(margin) <= 1e-10)
            negative = positive & (margin < -1e-10)
            if not np.all((zero | negative)[positive]):
                raise ValueError('A positive realized component lacks a valid admitted forecast margin')
            counts['positive_cost_time_components'] += int(positive.sum())
            counts['positive_with_zero_bank_margin'] += int(zero.sum())
            counts['positive_with_strict_negative_bank_margin'] += int(negative.sum())
            events.append(dict(initial_index=case['initial_index'], disturbance=case['disturbance'],
                controller=case['controller'], mechanism_sha256=digest(detail_path), **step))
    OUTPUT.mkdir(parents=True)
    save(OUTPUT / 'analysis.json', dict(counts=dict(counts), by_bank={k: dict(v) for k, v in by_bank.items()},
        positive_admission_events=events, source_sha256=digest(__file__),
        replay_manifest_sha256=digest(REPLAY / 'sha256.json'), additional_sumo_runs=0,
        additional_independent_initials=0, outcome_dependent_selection=True,
        inference_scope='Selected harmful development outcomes only. Margin counts are correlated components, not independent samples.',
        risk_certified=False, policies_changed=False))
    shutil.copy2(__file__, OUTPUT / Path(__file__).name)
    save(OUTPUT / 'sha256.json', {p.name: digest(p) for p in OUTPUT.iterdir() if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps(dict(counts=dict(counts), by_bank={k: dict(v) for k, v in by_bank.items()})), flush=True)


if __name__ == '__main__':
    run()
