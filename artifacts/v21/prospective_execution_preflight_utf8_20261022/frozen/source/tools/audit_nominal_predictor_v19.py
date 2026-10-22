"""Verify observable-state nominal forecasts against standard SUMO construction."""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from coopgate.disturbance_v9 import DisturbedMergeEnv
from coopgate.observable_initial_v19 import ObservableInitialMergeEnv
from experiment_ego_v13 import run_episode
from probe_disturbed_gate_v19 import digest, save
from study_gate_v19 import ROOT, factories
from verify_development_v14 import trace_audit


def freeze(directory):
    family = json.loads((ROOT / 'artifacts/v19/risk_validation/protocol.json').read_text())
    plan = dict(purpose='Mechanism audit on new development initials: nominal predictions use observed initial state only, not actual seed or disturbance',
                cases=[dict(seed=61930001 + i, scenario='nominal' if i < 3 else 'dense') for i in range(6)],
                spec=dict(methods=family['spec']['methods']), models=family['models'],
                actual_sumo_runs_planned=120, future_disturbance_input=False,
                timing_scope='Full shadow-family preparation includes simulator startup, resets and all ten nominal rollouts. Other studies are running; this is not a real-time claim.')
    sources = list((ROOT / 'src/coopgate').glob('*.py')) + [Path(__file__),
               ROOT / 'tools/experiment_ego_v13.py', ROOT / 'tools/probe_disturbed_gate_v19.py',
               ROOT / 'tools/study_gate_v19.py', ROOT / 'tools/verify_development_v14.py']
    plan['source_sha256'] = {p.resolve().relative_to(ROOT).as_posix(): digest(p) for p in sources}
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'plan.json', plan)
    for source in plan['source_sha256']:
        target = out / 'source' / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source, target)
    print(json.dumps(dict(actual_sumo_runs_planned=120, plan_sha256=digest(out / 'plan.json'))), flush=True)


def run(directory):
    torch.set_num_threads(1)
    out = Path(directory)
    if (out / 'episodes.json').exists():
        raise FileExistsError('Preserve existing audit runs')
    plan = json.loads((out / 'plan.json').read_text())
    for path, expected in plan['source_sha256'].items():
        assert digest(out / 'source' / path) == digest(ROOT / path) == expected, path
    fs = factories(plan)
    warm = np.array([[-80., 12.], [-85., 13.], [-120., 11.]])
    for factory in fs.values():
        factory(warm).select(warm, np.zeros(3))
    rows, pairs, preparations = [], [], []
    with DisturbedMergeEnv() as original:
        for case_index, case in enumerate(plan['cases']):
            references = {}
            for method, factory in fs.items():
                row, trace, initial = run_episode(original, case['seed'], case['scenario'], 'clean', factory, True)
                row.update(initial=initial.tolist(), controller=method, initialization='original', case_index=case_index)
                trace_audit(trace, row)
                references[method] = row, trace
                rows.append(row)
                save(out / f'trace_{case_index}_original_{method}.json', trace)
                save(out / 'episodes.json', rows)
            expected_initial = references['baseline'][0]['initial']
            started = time.perf_counter()
            with ObservableInitialMergeEnv(expected_initial) as shadow:
                for method, factory in fs.items():
                    # The labels intentionally differ from both the true seed
                    # and the engine's fixed seed; the predictor ignores them.
                    row, trace, initial = run_episode(shadow, 90000000 + case_index, 'unused_label', 'clean', factory, True)
                    row.update(initial=initial.tolist(), controller=method, initialization='observed_state', case_index=case_index)
                    trace_audit(trace, row)
                    ref_row, ref_trace = references[method]
                    np.testing.assert_allclose(row['initial'], expected_initial, atol=1e-8, rtol=0)
                    for field in ('per_vehicle_finish', 'per_vehicle_variation', 'completed', 'collision', 'aborted', 'contract_violations'):
                        np.testing.assert_array_equal(row[field], ref_row[field], err_msg=f'{case_index}/{method}/{field}')
                    assert len(trace) == len(ref_trace)
                    for left, right in zip(ref_trace, trace):
                        np.testing.assert_array_equal(left['command'], right['command'])
                        np.testing.assert_allclose(left['x'], right['x'], atol=1e-8, rtol=0)
                        assert not np.any(np.array(right['disturbance']))
                    rows.append(row)
                    save(out / f'trace_{case_index}_observed_state_{method}.json', trace)
                    save(out / 'episodes.json', rows)
                    pairs.append(dict(case_index=case_index, controller=method, decision_steps=len(trace),
                                      initial_error=float(np.max(np.abs(initial - np.array(expected_initial))))))
            preparations.append(dict(case_index=case_index, full_shadow_family_wall_s=time.perf_counter() - started))
            print(json.dumps(dict(case=case_index, actual_sumo_runs=len(rows), paired_profiles=len(pairs))), flush=True)
    assert len(rows) == plan['actual_sumo_runs_planned']
    result = dict(actual_sumo_runs=len(rows), paired_profiles=len(pairs),
                  paired_decision_steps=sum(p['decision_steps'] for p in pairs), pairs=pairs,
                  preparations=preparations, uses_actual_episode_seed=False, uses_future_disturbance=False,
                  risk_certified=False, timing_scope=plan['timing_scope'],
                  episodes_sha256=digest(out / 'episodes.json'), plan_sha256=digest(out / 'plan.json'))
    save(out / 'audit.json', result)
    save(out / 'sha256.json', {p.relative_to(out).as_posix(): digest(p)
         for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps({k: v for k, v in result.items() if k != 'pairs'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('freeze', 'run'))
    parser.add_argument('directory')
    args = parser.parse_args()
    (freeze if args.command == 'freeze' else run)(args.directory)
