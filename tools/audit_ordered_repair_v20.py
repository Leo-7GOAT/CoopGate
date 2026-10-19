"""Verify path-order invariance and measure complete calls in actual SUMO."""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from coopgate.advantage_gate_v18 import action_then_baseline, baseline_suffix
from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.disturbance_v9 import DisturbedMergeEnv
from coopgate.disturbed_advantage_v19 import disturbed_action_difference
from coopgate.fast_repair_v20 import FastRepairPortfolioGate
from coopgate.fast_v8 import parameters
from coopgate.merge_v4 import MergeConfig
from coopgate.ordered_repair_v20 import TemporalFirstRepairGate
from coopgate.temporal_portfolio_v20 import temporal_stress_bank
from audit_fast_repair_v20 import IDENTICAL
from experiment_ego_v13 import run_episode
from probe_disturbed_gate_v19 import digest, save
from recover_checkpoint_v19 import atomic_bytes
from start_final_v19 import require_committed, verify_manifest
from verify_development_v14 import trace_audit

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'artifacts/v20/repair_development'
PARENT = ROOT / 'artifacts/v20/fast_repair_audit'
OUTPUT = ROOT / 'artifacts/v20/ordered_repair_audit'
IMPLEMENTATIONS = ('cached', 'temporal_first', 'temporal_first_bounded')


def development_profile(rows):
    groups = {}
    for row in rows:
        if not row['controller'].startswith('repair_'):
            continue
        index, mode, name = row['initial_index'], row['disturbance'], row['controller']
        trace = json.loads((STUDY / f'shard_{index % 2}/trace_{index}_{mode}_{name}.json').read_text())
        if len(trace) != len(row['controller_ms']):
            raise ValueError('Timing and trajectory lengths differ')
        for item, elapsed in zip(trace, row['controller_ms']):
            group = ('admitted' if item['portfolio_joint_action_admitted'] else 'no_candidates' if item['candidate_count'] == 0
                     else 'nominal_rejection' if item['nominal_candidate_count'] == 0 else 'disturbance_rejection')
            result = groups.setdefault(group, dict(steps=0, controller_ms=0., internal_transitions=0))
            result['steps'] += 1
            result['controller_ms'] += elapsed
            result['internal_transitions'] += item['performance_model_transitions']
    total = sum(group['controller_ms'] for group in groups.values())
    for group in groups.values():
        group['observed_time_fraction'] = group['controller_ms'] / total
    return groups


def freeze():
    if OUTPUT.exists():
        raise FileExistsError('Preserve the existing path-order audit')
    # Read the previously frozen case plan, never its live outcome files.
    parent = json.loads((PARENT / 'plan.json').read_text())
    verify_manifest(STUDY)
    sources = dict(parent['source_sha256'])
    for path in ('src/coopgate/ordered_repair_v20.py', 'tools/audit_ordered_repair_v20.py',
                 'tests/test_ordered_repair_v20.py', 'tests/test_ordered_repair_audit_v20.py'):
        sources[path] = digest(ROOT / path)
    profile = development_profile(json.loads((STUDY / 'episodes.json').read_text()))
    plan = dict(cases=parent['cases'], implementations=list(IMPLEMENTATIONS), models=parent['models'],
                expected_actual_sumo_runs=3 * len(parent['cases']), source_sha256=sources,
                parent_case_plan_sha256=digest(PARENT / 'plan.json'),
                study_protocol_sha256=parent['study_protocol_sha256'], study_episodes_sha256=parent['study_episodes_sha256'],
                path_order=[0, *range(17, 33), *range(1, 17)], additional_independent_initials=0,
                completed_development_runtime_profile=profile,
                selection_rule='Reuse all 19 previously frozen engineering cases without inspecting that running audit outcomes.',
                scope='Development equivalence and complete-call timing; preserve all harmful outcomes and all models.',
                verification='Every actual action and accepted canonical certificate must match the original archived trajectory. Each rejected candidate has an independently recomputed violating path; every lower bound is checked against its full value.',
                timing_scope='Warmed functions and cyclic order with concurrent desktop experiments; no isolated deployment or hard deadline claim.')
    OUTPUT.mkdir(parents=True)
    save(OUTPUT / 'plan.json', plan)
    for path in sources:
        target = OUTPUT / 'source' / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, target)
    print(json.dumps(dict(cases=len(plan['cases']), actual_sumo_runs=plan['expected_actual_sumo_runs'],
                          plan_sha256=digest(OUTPUT / 'plan.json'))), flush=True)


def verify_witness(reported, full, bounded):
    reported, full = np.asarray(reported), np.asarray(full)
    if not np.isfinite(reported).all() or not np.isfinite(full).all():
        raise ValueError('A rejection witness must be finite')
    if bounded:
        if not np.all(reported <= full + 1e-9):
            raise ValueError('Rejection bound exceeds the full forecast')
    else:
        np.testing.assert_array_equal(reported, full)
    if not np.any(reported > 1e-10) or not np.any(full > 1e-10):
        raise ValueError('Candidate rejection has no true violating component')


def verify_step(original, item, state, previous, step, packed, tapes, cache):
    for key in IDENTICAL:
        np.testing.assert_equal(original[key], item[key], err_msg=f'{step}/{key}')
    if len(original['candidate_evidence']) != len(item['candidate_evidence']):
        raise ValueError('Nominal candidate ordering or selection changed')
    order = item.get('forecast_path_order', list(range(33)))
    if order[0] != 0 or sorted(order) != list(range(33)):
        raise ValueError('The reordered bank changed its paths')
    witnesses, bounds = 0, 0
    for left, right in zip(original['candidate_evidence'], item['candidate_evidence']):
        for key in ('composition', 'weight', 'command', 'failed_scenarios', 'forecast_error'):
            np.testing.assert_equal(left[key], right[key])
        if right['forecast_error'] is not None:
            raise ValueError('Unexpected forecast failure in equivalence audit')
        indices = right.get('evaluated_original_path_indices', list(range(right['checked_scenarios'])))
        if indices != order[:right['checked_scenarios']] or len(right['differences']) != len(indices):
            raise ValueError('Partial forecast path identities changed')
        if right['failed_scenarios']:
            local_index = len(indices) - 1
            if set(right['lower_bound_scenarios']) - {local_index}:
                raise ValueError('A rejected bound was followed by extra forecast work')
            if not np.isfinite(right['differences']).all() or np.any(np.asarray(right['differences'][:-1]) > 1e-10):
                raise ValueError('The first rejection witness was omitted')
            canonical = indices[-1]
            if canonical == 0:
                raise ValueError('A nominally admitted candidate cannot fail nominally')
            command = np.asarray(right['command'])
            key = (step, state.tobytes(), previous.tobytes(), command.tobytes(), canonical)
            if key not in cache:
                cache[key] = disturbed_action_difference(state, previous, step, command, tapes[canonical], packed)[0]
            bounded = local_index in right['lower_bound_scenarios']
            verify_witness(right['differences'][-1], cache[key], bounded)
            witnesses += 1
            bounds += int(bounded)
        else:
            if len(indices) != 33 or right['lower_bound_scenarios']:
                raise ValueError('Only complete exact forecasts may authorize an action')
            canonical_values = np.asarray(right['differences'])[np.argsort(indices)]
            if not np.isfinite(canonical_values).all() or np.any(canonical_values > 1e-10):
                raise ValueError('Admitted complete bank contains a violation')
            np.testing.assert_array_equal(canonical_values, original['selected_bank_differences'])
    nominal = item['nominal_pruned_bounds']
    if nominal:
        base = baseline_suffix(state, previous, step, packed)[0]
        for rejection in nominal:
            full = action_then_baseline(state, previous, step, np.asarray(rejection['command']), packed)[0] - base
            verify_witness(rejection['lower_difference'], full, True)
    return witnesses, bounds, len(nominal)


def run():
    torch.set_num_threads(1)
    if (OUTPUT / 'episodes.json').exists():
        raise FileExistsError('Preserve all previous ordered audit runs')
    plan = json.loads((OUTPUT / 'plan.json').read_text())
    for source, expected in plan['source_sha256'].items():
        if digest(ROOT / source) != expected or digest(OUTPUT / 'source' / source) != expected:
            raise ValueError('Frozen order audit source changed: ' + source)
    require_committed([OUTPUT / 'plan.json', *[ROOT / path for path in plan['source_sha256']]])
    verify_manifest(STUDY)
    if digest(PARENT / 'plan.json') != plan['parent_case_plan_sha256']:
        raise ValueError('Parent case selection changed')
    if digest(STUDY / 'episodes.json') != plan['study_episodes_sha256'] or digest(STUDY / 'protocol.json') != plan['study_protocol_sha256']:
        raise ValueError('Original study changed')
    historical = {(r['initial_index'], r['disturbance'], r['controller']): r for r in json.loads((STUDY / 'episodes.json').read_text())}
    models = {}
    for seed, record in plan['models'].items():
        if digest(ROOT / record['path']) != record['sha256']:
            raise ValueError('Frozen actor changed')
        models[seed] = load_coupled(ROOT / record['path'])[0]
    def gate(model, implementation):
        if implementation == 'cached':
            return FastRepairPortfolioGate(model, False)
        return TemporalFirstRepairGate(model, implementation == 'temporal_first_bounded')
    for model in models.values():
        for implementation in IMPLEMENTATIONS:
            gate(model, implementation).select(np.asarray([[-80., 12.], [-85., 13.], [-120., 11.]]), np.zeros(3))
    packed, tapes = parameters(MergeConfig()), temporal_stress_bank('sampled')[1]
    rows, pairs = [], []
    with DisturbedMergeEnv() as env:
        for ci, case in enumerate(plan['cases']):
            index, mode, name = case['index'], case['mode'], case['method']
            archived = historical[index, mode, name]
            trace_path = STUDY / f'shard_{index % 2}/trace_{index}_{mode}_{name}.json'
            original = json.loads(trace_path.read_text())
            model = models[name.rsplit('_', 1)[1]]
            shift, checked, witness_cache = ci % 3, {}, {}
            for implementation in IMPLEMENTATIONS[shift:] + IMPLEMENTATIONS[:shift]:
                cpu = time.process_time()
                row, trace, initial = run_episode(env, archived['seed'], archived['scenario'], mode,
                                                 lambda x, m=model, impl=implementation: gate(m, impl), True)
                row.update(controller=name, initial_index=index, initial=initial.tolist(), implementation=implementation,
                           process_cpu_ms=(time.process_time() - cpu) * 1000.,
                           internal_transitions=sum(t['performance_model_transitions'] for t in trace),
                           optimistic_transitions=sum(t['optimistic_forecast_transitions'] for t in trace),
                           learned_action_steps=sum(t['portfolio_joint_action_admitted'] for t in trace),
                           repair_action_steps=sum(t['repair_action_admitted'] for t in trace))
                save(OUTPUT / f'trace_{ci}_{implementation}.json', trace)
                rows.append(row)
                atomic_bytes(OUTPUT / 'episodes.json', (json.dumps(rows, indent=2) + '\n').encode())
                trace_audit(trace, row)
                for key in ('initial', 'per_vehicle_finish', 'per_vehicle_variation', 'completed', 'collision', 'aborted',
                            'contract_violations', 'learned_action_steps', 'repair_action_steps'):
                    np.testing.assert_equal(row[key], archived[key], err_msg=f'{ci}/{implementation}/{key}')
                if len(trace) != len(original):
                    raise ValueError('Closed-loop length changed')
                state, previous = np.asarray(archived['initial']), np.zeros(3)
                counts = np.zeros(3, dtype=int)
                for step, (left, right) in enumerate(zip(original, trace)):
                    counts += verify_step(left, right, state, previous, step, packed, tapes, witness_cache)
                    state, previous = np.asarray(left['x']), np.asarray(left['command'])
                checked[implementation] = dict(rejection_witnesses=int(counts[0]), bank_bounds=int(counts[1]), nominal_bounds=int(counts[2]))
            pairs.append(dict(case=case, matched_steps=len(original), verified=checked))
            print(json.dumps(dict(completed_cases=len(pairs), expected_cases=len(plan['cases']), actual_sumo_runs=len(rows))), flush=True)
    if len(rows) != plan['expected_actual_sumo_runs']:
        raise ValueError('Incomplete ordered audit')
    timings = {}
    for implementation in IMPLEMENTATIONS:
        selected = [r for r in rows if r['implementation'] == implementation]
        values = np.asarray([v for r in selected for v in r['controller_ms']])
        timings[implementation] = dict(total_controller_ms=float(values.sum()), p50_ms=float(np.median(values)),
            p95_ms=float(np.percentile(values, 95)), p99_ms=float(np.percentile(values, 99)), max_ms=float(values.max()),
            over_500ms=int((values > 500).sum()), decision_steps=len(values), process_cpu_ms=sum(r['process_cpu_ms'] for r in selected),
            internal_transitions=sum(r['internal_transitions'] for r in selected), optimistic_transitions=sum(r['optimistic_transitions'] for r in selected))
    report = dict(actual_sumo_runs=len(rows), matched_cases=len(pairs), matched_decision_steps=sum(p['matched_steps'] for p in pairs),
                  timings=timings, paired_cases=pairs, additional_independent_initials=0, risk_certified=False,
                  hard_real_time_certified=False, timing_scope=plan['timing_scope'],
                  plan_sha256=digest(OUTPUT / 'plan.json'), episodes_sha256=digest(OUTPUT / 'episodes.json'))
    save(OUTPUT / 'audit.json', report)
    save(OUTPUT / 'sha256.json', {p.relative_to(OUTPUT).as_posix(): digest(p) for p in OUTPUT.rglob('*') if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps({k: v for k, v in report.items() if k != 'paired_cases'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('freeze', 'run'))
    (freeze if parser.parse_args().command == 'freeze' else run)()
