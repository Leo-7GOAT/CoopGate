"""Recount the completed temporal-bank study and retain every harmful outcome."""
import json
from pathlib import Path
import shutil

import numpy as np

from diagnose_calibration_harm_v19 import harmful, identity
from probe_disturbed_gate_v19 import digest, save
from review_repair_closed_loop_v20 import outcome_counts
from start_final_v19 import verify_manifest
from study_gate_v19 import read_protocol

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'artifacts/v20/temporal_development'
EXECUTION = ROOT / 'artifacts/v20/temporal_development_execution'
OUTPUT = ROOT / 'artifacts/v20/temporal_development_results'


def run():
    if OUTPUT.exists():
        raise FileExistsError('Preserve completed temporal analysis')
    for directory in (STUDY, EXECUTION):
        verify_manifest(directory)
    state = json.loads((EXECUTION / 'state.json').read_text())
    native = json.loads((EXECUTION / 'supervisor_native_exit.json').read_text())
    if state['phase'] != 'completed' or native['exit_code'] != 0 or state['summary_sha256'] != digest(STUDY / 'summary.json'):
        raise ValueError('The native collection has not successfully completed')
    _, protocol = read_protocol(STUDY)
    spec = protocol['spec']
    rows = json.loads((STUDY / 'episodes.json').read_text())
    summary = json.loads((STUDY / 'summary.json').read_text())
    lookup = {identity(r): r for r in rows}
    expected = {(i, m, name) for i in range(spec['n_initials']) for m in spec['modes'] for name in spec['methods']}
    if len(lookup) != len(rows) or set(lookup) != expected:
        raise ValueError('Incomplete or duplicate paired grid')
    recount = {}
    for name in spec['methods']:
        recount[name] = outcome_counts(rows, spec, name)
        for key, value in recount[name].items():
            np.testing.assert_allclose(value, summary['results'][name][key], atol=1e-12, rtol=1e-12)
    harms, comparisons = [], {}
    for row in rows:
        base = lookup[row['initial_index'], row['disturbance'], 'baseline']
        if harmful(row, base):
            harms.append(dict(initial_index=row['initial_index'], seed=row['seed'],
                disturbance=row['disturbance'], controller=row['controller'], scenario=row['scenario'],
                learned_action_steps=row['learned_action_steps'],
                cost_time_difference=(np.column_stack((row['per_vehicle_cost'], row['per_vehicle_finish'])) -
                                      np.column_stack((base['per_vehicle_cost'], base['per_vehicle_finish']))).tolist()))
    n = spec['n_initials']
    chosen = [spec['modes'][int(np.random.default_rng(np.random.SeedSequence(
        [spec['seed_base'] + i, spec['mode_selection_stream']])).integers(len(spec['modes'])))] for i in range(n)]
    for seed in (1802, 1812, 1822):
        for bank in ('alternating', 'sampled', 'combined'):
            cases = []
            for i in range(n):
                for mode in spec['modes']:
                    old, new, base = [lookup[i, mode, name] for name in (f'portfolio0_{seed}', f'{bank}_{seed}', 'baseline')]
                    cases.append(dict(index=i, mode=mode, actual=mode == chosen[i],
                        old_adopted=old['learned_action_steps'] > 0, new_adopted=new['learned_action_steps'] > 0,
                        old_harm=harmful(old, base), new_harm=harmful(new, base)))
            def count(selected):
                return dict(paired_runs=len(selected),
                    old_adoptions_retained=sum(c['old_adopted'] and c['new_adopted'] for c in selected),
                    old_adoptions_lost=sum(c['old_adopted'] and not c['new_adopted'] for c in selected),
                    newly_adopted=sum(not c['old_adopted'] and c['new_adopted'] for c in selected),
                    old_harms_retained=sum(c['old_harm'] and c['new_harm'] for c in selected),
                    old_harms_removed=sum(c['old_harm'] and not c['new_harm'] for c in selected),
                    new_harms=sum(not c['old_harm'] and c['new_harm'] for c in selected))
            comparisons[f'{bank}_{seed}'] = dict(actual=count([c for c in cases if c['actual']]), all_modes=count(cases))
    OUTPUT.mkdir(parents=True)
    save(OUTPUT / 'analysis.json', dict(data_role='post-hoc complete development recount', independent_initials=n,
        actual_sumo_runs=len(rows), additional_independent_initials=0, results=summary['results'],
        independently_recounted_outcomes=recount, paired_changes=comparisons, harmful_runs=harms,
        independent_full_trajectory_replays_pending=True, saved_original_traces=sum(r['trace_saved'] for r in rows),
        risk_certified=False, hard_real_time_certified=False, study_manifest_sha256=digest(STUDY / 'sha256.json'),
        execution_manifest_sha256=digest(EXECUTION / 'sha256.json'), source_sha256=digest(__file__)))
    shutil.copy2(__file__, OUTPUT / Path(__file__).name)
    save(OUTPUT / 'sha256.json', {p.name: digest(p) for p in OUTPUT.iterdir() if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps(dict(recounted_methods=len(recount), harmful_runs=len(harms), paired_changes=comparisons)), flush=True)


if __name__ == '__main__':
    run()
