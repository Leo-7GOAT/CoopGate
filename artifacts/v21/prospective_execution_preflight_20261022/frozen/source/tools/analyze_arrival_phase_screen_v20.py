"""Fixed-grid development description of the completed phase screen.

This analyzes archived one-action branches. No new controller, population risk,
or closed-loop adoption claim is made, and no margin is selected automatically.
"""
from collections import Counter
import json
from pathlib import Path
import shutil

import numpy as np

from probe_disturbed_gate_v19 import digest, save
from start_final_v19 import require_committed, verify_manifest

ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / 'artifacts/v20/arrival_phase_screen'
EXECUTION = ROOT / 'artifacts/v20/arrival_phase_screen_execution'
STUDY = ROOT / 'artifacts/v20/temporal_repair_extension'
OUTPUT = ROOT / 'artifacts/v20/arrival_phase_margin_analysis'
MARGINS_SECONDS = (0., .01, .025, .05, .1, .2, .5)


def run():
    if OUTPUT.exists():
        raise FileExistsError('Preserve the existing complete margin-grid analysis')
    require_committed([Path(__file__)])
    for directory in (SCREEN, EXECUTION, STUDY):
        verify_manifest(directory)
    if json.loads((EXECUTION/'exit.json').read_text(encoding='utf-8-sig'))['exit_code'] != 0:
        raise ValueError('The original screen has not completed successfully')
    audit = json.loads((SCREEN/'audit.json').read_text())
    rows = json.loads((SCREEN/'occurrences.json').read_text())
    spec = json.loads((STUDY/'protocol.json').read_text())['spec']
    if len(rows) != audit['admitted_occurrences']:
        raise ValueError('Admission coverage changed')
    states = {i:json.loads((SCREEN/f'state_{i}.json').read_text()) for i in range(audit['distinct_admitted_forecasts'])}
    maxima = {i:(float(np.max(s['phase_difference'])) if s['all_forecasts_complete'] else float('inf')) for i,s in states.items()}
    grid = []
    for margin in MARGINS_SECONDS:
        total, models, actual_runs = Counter(), {}, {}
        for model in ('repair_1802','repair_1812','repair_1822'):
            counts, retained_runs = Counter(), set()
            for row in rows:
                if row['controller'] != model:
                    continue
                keep = maxima[row['state_index']] <= margin + 1e-10
                label = ('retained_' if keep else 'rejected_') + (
                    'positive_residual' if row['positive_realized_branch_residual'] else 'nonpositive_residual')
                counts[label] += 1
                if keep:
                    index = row['initial_index']
                    mode = spec['modes'][int(np.random.default_rng(np.random.SeedSequence([
                        spec['seed_base']+index,spec['mode_selection_stream']])).integers(len(spec['modes'])))]
                    if row['disturbance'] == mode:
                        retained_runs.add(index)
            models[model] = dict(counts)
            actual_runs[model] = len(retained_runs)
            total.update(counts)
        if margin == 0. and dict(total) != audit['counts']:
            raise ValueError('Zero-margin screen was not reconstructed')
        if margin == .5 and all(s['all_forecasts_complete'] for s in states.values()) and any(
                total[label] for label in ('rejected_positive_residual','rejected_nonpositive_residual')):
            raise ValueError('Half-period margin should retain all originally eligible complete arrivals')
        grid.append(dict(margin_seconds=margin,counts=dict(total),by_model=models,
                         archived_actual_runs_with_any_retained_admission=actual_runs))
    output = dict(grid=grid,admitted_occurrences=len(rows),distinct_admitted_forecasts=len(states),
        independent_initials=128, original_actual_episode_adoptions={'repair_1802':41,'repair_1812':33,'repair_1822':38},
        interpretation='Counts share trajectories, models and forecasts. Retained archived runs are not adoption rates or bounds for a changed closed-loop policy.',
        original_gate_and_metrics_unchanged=True, selected_margin=None,
        new_sumo_runs=0,new_independent_initials=0,risk_certified=False,
        screen_manifest_sha256=digest(SCREEN/'sha256.json'),execution_manifest_sha256=digest(EXECUTION/'sha256.json'),
        source_sha256=digest(__file__))
    OUTPUT.mkdir(parents=True)
    save(OUTPUT/'analysis.json',output)
    shutil.copy2(__file__,OUTPUT/Path(__file__).name)
    save(OUTPUT/'sha256.json',{p.name:digest(p) for p in OUTPUT.iterdir() if p.is_file() and p.name!='sha256.json'})
    print(json.dumps(dict(grid=[{k:v for k,v in row.items() if k!='by_model'} for row in grid],selected_margin=None)),flush=True)


if __name__ == '__main__':
    run()
