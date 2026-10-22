"""Describe the completed 16-initial repair study without claiming certification."""
import json
from pathlib import Path

import numpy as np

from probe_disturbed_gate_v19 import digest, save
from start_final_v19 import verify_manifest
from study_gate_v19 import read_protocol
from study_temporal_v20 import is_failure, pair_delta

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'artifacts/v20/repair_development'
REVIEW = ROOT / 'artifacts/v20/repair_closed_loop_review'
OUTPUT = ROOT / 'artifacts/v20/repair_development_results'


def run():
    if OUTPUT.exists():
        raise FileExistsError('Preserve completed analysis')
    for directory in (STUDY, REVIEW):
        verify_manifest(directory)
    _, protocol = read_protocol(STUDY)
    audit = json.loads((REVIEW / 'audit.json').read_text())
    if not audit['all_reconstructed_commands_and_admitted_forecasts_match']:
        raise ValueError('Independent review did not pass')
    if audit['study_manifest_sha256'] != digest(STUDY / 'sha256.json'):
        raise ValueError('Review targets different study data')
    spec = protocol['spec']
    rows = json.loads((STUDY / 'episodes.json').read_text())
    summary = json.loads((STUDY / 'summary.json').read_text())
    lookup = {(r['initial_index'], r['disturbance'], r['controller']): r for r in rows}
    n, modes = spec['n_initials'], spec['modes']
    chosen = [int(np.random.default_rng(np.random.SeedSequence([spec['seed_base'] + i,
              spec['mode_selection_stream']])).integers(len(modes))) for i in range(n)]
    resamples = np.random.default_rng(20519).integers(n, size=(5000, n))
    comparisons, harms = {}, []
    for seed in (1802, 1812, 1822):
        gains, actual, added, retained = [], [], 0, 0
        for i in range(n):
            group = []
            for mi, mode in enumerate(modes):
                old, new, base = [lookup[i, mode, name] for name in
                                  (f'sampled_{seed}', f'repair_{seed}', 'baseline')]
                difference = old['team_cost'] - new['team_cost']
                group.append(difference)
                if mi == chosen[i]:
                    actual.append(difference)
                    retained += int(old['learned_action_steps'] > 0)
                    added += int(new['learned_action_steps'] > 0 and old['learned_action_steps'] == 0)
                delta = pair_delta(new, base)
                if is_failure(new) or np.any(delta > 1e-9):
                    path = STUDY / f'shard_{i % spec["shards"]}/trace_{i}_{mode}_repair_{seed}.json'
                    trace = json.loads(path.read_text())
                    events = [dict(step=j, t=item['t'], weight=item['repair_weight'],
                                   composition=item['composition'], command=item['command'],
                                   baseline_command=item['baseline_command'],
                                   selected_bank_differences=item['selected_bank_differences'])
                              for j, item in enumerate(trace) if item.get('portfolio_joint_action_admitted')]
                    harms.append(dict(initial_index=i, seed=new['seed'], model=seed, mode=mode,
                                      cost_time_differences=delta.tolist(), failure=is_failure(new),
                                      original_used=old['learned_action_steps'] > 0,
                                      trace_path=path.relative_to(ROOT).as_posix(), trace_sha256=digest(path),
                                      adopted_events=events))
            gains.append(float(np.mean(group)))
        def interval(values):
            array = np.asarray(values)
            return dict(mean=float(array.mean()), pointwise_bootstrap95=np.quantile(
                array[resamples].mean(1), [.025, .975]).tolist())
        comparisons[str(seed)] = dict(actual_incremental_gain_seconds=interval(actual),
                                     four_mode_incremental_gain_seconds=interval(gains),
                                     actual_retained_adoptions=retained, actual_added_adoptions=added)
    OUTPUT.mkdir(parents=True)
    save(OUTPUT / 'analysis.json', dict(
        data_role='post-hoc development analysis', independent_initials=n, actual_sumo_runs=len(rows),
        results=summary['results'], comparisons=comparisons, harmful_repair_runs=harms,
        independent_review_counts=audit['counts'], bootstrap_stream=20519, bootstrap_samples=5000,
        interval_scope='Paired initial resampling, descriptive pointwise development intervals; no simultaneous or post-selection guarantee.',
        risk_certified=False, hard_real_time_certified=False,
        study_manifest_sha256=digest(STUDY / 'sha256.json'), review_manifest_sha256=digest(REVIEW / 'sha256.json'),
        source_sha256=digest(__file__)))
    save(OUTPUT / 'sha256.json', {'analysis.json': digest(OUTPUT / 'analysis.json')})
    print(json.dumps(dict(comparisons=comparisons, harmful_repair_runs=len(harms),
                          analysis_sha256=digest(OUTPUT / 'analysis.json'))))


if __name__ == '__main__':
    run()
