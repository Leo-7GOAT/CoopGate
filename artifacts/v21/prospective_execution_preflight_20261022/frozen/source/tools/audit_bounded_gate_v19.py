"""Actual SUMO equivalence audit for optimistic rejection-only pruning."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from coopgate.advantage_gate_v18 import action_then_baseline, baseline_suffix
from coopgate.bounded_portfolio_v19 import BoundedJointPortfolioGate
from coopgate.cached_portfolio_v19 import CachedJointPortfolioGate
from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.disturbance_v9 import DisturbedMergeEnv
from coopgate.fast_v8 import parameters
from coopgate.merge_v4 import MergeConfig
from experiment_ego_v13 import run_episode
from probe_disturbed_gate_v19 import digest, save
from verify_development_v14 import trace_audit

ROOT = Path(__file__).resolve().parents[1]


def verify_step(reference, bounded, state, previous, step, packed):
    identical = ('t', 'x', 'command', 'disturbance', 'in_set', 'certified', 'intervention',
                 'nominal_joint_action_admitted', 'portfolio_joint_action_admitted', 'composition',
                 'candidate_count', 'nominal_candidate_count', 'selected_bank_differences', 'max_failed_scenarios')
    for key in identical:
        assert reference[key] == bounded[key], (step, key)
    assert bounded['performance_model_transitions'] <= reference['performance_model_transitions']
    assert len(reference['candidate_evidence']) == len(bounded['candidate_evidence'])
    bank_prunes = 0
    for left, right in zip(reference['candidate_evidence'], bounded['candidate_evidence']):
        for key in ('composition', 'command', 'checked_scenarios', 'failed_scenarios'):
            assert left[key] == right[key], (step, key)
        for index, (exact, bound) in enumerate(zip(left['differences'], right['differences'])):
            exact, bound = np.array(exact), np.array(bound)
            if index in right['lower_bound_scenarios']:
                assert np.all(bound <= exact + 1e-9), (step, index, bound, exact)
                assert np.any(bound > 1e-10) and np.any(exact > 1e-10)
                bank_prunes += 1
            else:
                np.testing.assert_array_equal(exact, bound)
    if bounded['portfolio_joint_action_admitted']:
        assert len(bounded['selected_bank_differences']) == 17
        assert np.all(np.array(bounded['selected_bank_differences']) <= 1e-10)
    # Reference nominal rejections are not stored in the original trace. Use
    # the unoptimized, independently preserved full potential as the oracle.
    nominal_prunes = bounded['nominal_pruned_bounds']
    if nominal_prunes:
        base = baseline_suffix(state, previous, step, packed)[0]
        for item in nominal_prunes:
            exact = action_then_baseline(state, previous, step, np.array(item['command']), packed)[0] - base
            bound = np.array(item['lower_difference'])
            assert np.all(bound <= exact + 1e-9), (step, item['composition'], bound, exact)
            assert np.any(exact > 1e-10) and np.any(bound > 1e-10)
    return len(nominal_prunes), bank_prunes


def freeze(directory):
    previous_path = ROOT / 'artifacts/v19/cached_gate_audit/plan.json'
    previous = json.loads(previous_path.read_text())
    sources = list((ROOT / 'src/coopgate').glob('*.py')) + [Path(__file__),
               ROOT / 'tools/experiment_ego_v13.py', ROOT / 'tools/probe_disturbed_gate_v19.py',
               ROOT / 'tools/verify_development_v14.py']
    plan = dict(purpose='Paired SUMO audit of rejection-only physical lower bounds against the cached exact controller',
                cases=previous['cases'], actual_sumo_runs_planned=2 * len(previous['cases']),
                models=previous['models'], prior_case_plan_sha256=digest(previous_path),
                study_protocol_sha256=previous['study_protocol_sha256'],
                study_episodes_sha256=previous['study_episodes_sha256'],
                oracle='Every pruned branch is checked against an original full evaluation, outside timing measurement',
                timing_scope='Warmed functions, alternating implementation order; other frozen experiments run concurrently. No independent efficacy or hard deadline claim.',
                source_sha256={p.resolve().relative_to(ROOT).as_posix(): digest(p) for p in sources})
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'plan.json', plan)
    for source in plan['source_sha256']:
        target = out / 'source' / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source, target)
    print(json.dumps(dict(cases=len(plan['cases']), actual_sumo_runs_planned=plan['actual_sumo_runs_planned'],
                          plan_sha256=digest(out / 'plan.json'))), flush=True)


def run(directory):
    torch.set_num_threads(1)
    out = Path(directory)
    if (out / 'episodes.json').exists():
        raise FileExistsError('Preserve earlier audit runs')
    plan = json.loads((out / 'plan.json').read_text())
    for path, expected in plan['source_sha256'].items():
        assert digest(out / 'source' / path) == digest(ROOT / path) == expected, path
    study = ROOT / 'artifacts/v19/portfolio_probe'
    assert digest(study / 'protocol.json') == plan['study_protocol_sha256']
    assert digest(study / 'episodes.json') == plan['study_episodes_sha256']
    historical = {(r['initial_index'], r['disturbance'], r['controller']): r
                  for r in json.loads((study / 'episodes.json').read_text())}
    models = {}
    for seed, record in plan['models'].items():
        assert digest(ROOT / record['path']) == record['sha256']
        models[seed], _ = load_coupled(ROOT / record['path'])
    warm = np.array([[-80., 12.], [-85., 13.], [-120., 11.]])
    for model in models.values():
        for gate in (CachedJointPortfolioGate, BoundedJointPortfolioGate):
            gate(model).select(warm, np.zeros(3))
    packed = parameters(MergeConfig())
    rows, pairs = [], []
    with DisturbedMergeEnv() as env:
        for ci, (index, mode, method) in enumerate(plan['cases']):
            archived = historical[index, mode, method]
            model = models[method.rsplit('_', 1)[1]]
            results = {}
            order = ('cached', 'bounded') if ci % 2 == 0 else ('bounded', 'cached')
            for implementation in order:
                gate = CachedJointPortfolioGate if implementation == 'cached' else BoundedJointPortfolioGate
                row, trace, initial = run_episode(env, archived['seed'], archived['scenario'], mode,
                                                 lambda x, g=gate, m=model: g(m), True)
                row.update(controller=method, initial=initial.tolist(), initial_index=index, implementation=implementation,
                           internal_transitions=sum(t['performance_model_transitions'] for t in trace),
                           optimistic_transitions=sum(t.get('optimistic_forecast_transitions', 0) for t in trace),
                           learned_action_steps=sum(t['portfolio_joint_action_admitted'] for t in trace))
                trace_audit(trace, row)
                for field in ('initial', 'per_vehicle_finish', 'per_vehicle_variation', 'completed', 'collision', 'aborted', 'contract_violations'):
                    np.testing.assert_array_equal(row[field], archived[field], err_msg=f'{ci}/{implementation}/{field}')
                results[implementation] = row, trace
                rows.append(row)
                save(out / f'trace_{ci}_{implementation}.json', trace)
                save(out / 'episodes.json', rows)
            reference, bounded = results['cached'][1], results['bounded'][1]
            assert len(reference) == len(bounded)
            state, previous = np.array(archived['initial']), np.zeros(3)
            nominal_prunes, bank_prunes = 0, 0
            for step, (left, right) in enumerate(zip(reference, bounded)):
                np_count, bp_count = verify_step(left, right, state, previous, step, packed)
                nominal_prunes += np_count
                bank_prunes += bp_count
                state, previous = np.array(left['x']), np.array(left['command'])
            pairs.append(dict(index=index, mode=mode, method=method, exact_decision_steps=len(reference),
                              nominal_prunes_verified=nominal_prunes, bank_prunes_verified=bank_prunes))
            print(json.dumps(dict(case=ci, actual_sumo_runs=len(rows), verified_nominal_prunes=nominal_prunes,
                                  verified_bank_prunes=bank_prunes)), flush=True)
    assert len(rows) == plan['actual_sumo_runs_planned']
    timings = {}
    for implementation in ('cached', 'bounded'):
        data = [r for r in rows if r['implementation'] == implementation]
        values = [v for r in data for v in r['controller_ms']]
        timings[implementation] = dict(
            p50_ms=float(np.median(values)), p95_ms=float(np.percentile(values, 95)),
            p99_ms=float(np.percentile(values, 99)), max_ms=float(max(values)), total_controller_ms=float(sum(values)),
            steps_over_500ms=sum(v > 500 for v in values), decision_steps=len(values),
            internal_transitions=sum(r['internal_transitions'] for r in data),
            optimistic_transitions=sum(r['optimistic_transitions'] for r in data))
    result = dict(actual_sumo_replays=len(rows), exact_paired_cases=len(pairs),
                  exact_paired_decision_steps=sum(p['exact_decision_steps'] for p in pairs),
                  verified_nominal_prunes=sum(p['nominal_prunes_verified'] for p in pairs),
                  verified_bank_prunes=sum(p['bank_prunes_verified'] for p in pairs),
                  timings=timings, pairs=pairs, risk_certified=False, hard_real_time_certified=False,
                  controller_time_reduction=1 - timings['bounded']['total_controller_ms'] / timings['cached']['total_controller_ms'],
                  timing_scope=plan['timing_scope'], episodes_sha256=digest(out / 'episodes.json'), plan_sha256=digest(out / 'plan.json'))
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
