"""Paired actual SUMO runs for exact repair-gate acceleration and bound checks."""
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
from coopgate.fast_repair_v20 import FastRepairPortfolioGate
from coopgate.fast_v8 import parameters
from coopgate.merge_v4 import MergeConfig
from coopgate.repair_portfolio_v20 import RepairPortfolioGate
from experiment_ego_v13 import run_episode
from probe_disturbed_gate_v19 import digest, save
from recover_checkpoint_v19 import atomic_bytes
from start_final_v19 import require_committed, verify_manifest
from study_gate_v19 import read_protocol
from study_temporal_v20 import is_failure, pair_delta
from verify_development_v14 import trace_audit

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'artifacts/v20/repair_development'
OUTPUT = ROOT / 'artifacts/v20/fast_repair_audit'
IMPLEMENTATIONS = ('reference', 'cached', 'bounded')
IDENTICAL = ('t', 'x', 'command', 'disturbance', 'in_set', 'certified', 'intervention',
             'nominal_joint_action_admitted', 'portfolio_joint_action_admitted', 'repair_action_admitted',
             'composition', 'repair_weight', 'baseline_command', 'actor_proposal', 'candidate_count',
             'original_candidate_count', 'repair_candidate_count', 'nominal_candidate_count',
             'selected_bank_differences', 'max_failed_scenarios', 'temporal_bank', 'forecast_bank_size',
             'generation_errors', 'forecast_errors', 'work_counts_complete')


def select_cases(rows):
    lookup = {(r['initial_index'], r['disturbance'], r['controller']): r for r in rows}
    repairs = [r for r in rows if r['controller'].startswith('repair_')]
    chosen = {}
    def add(row, reason):
        key = (row['initial_index'], row['disturbance'], row['controller'])
        chosen.setdefault(key, []).append(reason)
    for seed in (1802, 1812, 1822):
        family = [r for r in repairs if r['controller'] == f'repair_{seed}']
        for mode in ('clean', 'uniform', 'bias', 'pulse'):
            pool = [r for r in family if r['disturbance'] == mode]
            repaired = [r for r in pool if r['repair_action_steps'] > 0]
            add(min(repaired or pool, key=lambda r: r['initial_index']), f'{seed}/{mode} attenuation coverage')
        add(max(family, key=lambda r: (max(r['controller_ms']), -r['initial_index'], r['disturbance'])),
            f'{seed} largest archived single-call time')
        zero = [r for r in family if r['learned_action_steps'] == 0]
        if zero:
            add(min(zero, key=lambda r: (r['initial_index'], r['disturbance'])), f'{seed} complete fallback')
    for row in repairs:
        base = lookup[row['initial_index'], row['disturbance'], 'baseline']
        if is_failure(row) or np.any(pair_delta(row, base) > 1e-9):
            add(row, 'all archived harmful repair outcomes')
    return [dict(index=k[0], mode=k[1], method=k[2], reasons=chosen[k]) for k in sorted(chosen)]


def freeze():
    if OUTPUT.exists():
        raise FileExistsError('Preserve existing runtime audit')
    verify_manifest(STUDY)
    _, protocol = read_protocol(STUDY)
    cases = select_cases(json.loads((STUDY / 'episodes.json').read_text()))
    sources = dict(protocol['source_sha256'])
    for path in ('src/coopgate/fast_repair_v20.py', 'src/coopgate/cached_portfolio_v19.py',
                 'src/coopgate/bounded_portfolio_v19.py', 'src/coopgate/deduplicated_planner_v19.py',
                 'tools/audit_fast_repair_v20.py',
                 'tests/test_fast_repair_v20.py', 'tests/test_fast_repair_audit_v20.py'):
        sources[path] = digest(ROOT / path)
    plan = dict(cases=cases, implementations=list(IMPLEMENTATIONS), expected_actual_sumo_runs=3 * len(cases),
                models=protocol['models'], study_protocol_sha256=digest(STUDY / 'protocol.json'),
                study_episodes_sha256=digest(STUDY / 'episodes.json'), source_sha256=sources,
                selection_rule='For every model/mode, first attenuated episode or first available; plus every harmful outcome, each model largest single-call time and first complete fallback.',
                scope='Paired development replay and behavior equivalence, not population efficacy or risk calibration.',
                timing_scope='Warmed functions; cyclic execution order across three implementations; concurrent desktop experiments. Complete controller wall time and process CPU time reported; no hard deadline claim.',
                pruning_oracle='Every lower-bound rejection is independently compared with the original complete forecast outside measured execution.',
                additional_independent_initials=0)
    OUTPUT.mkdir(parents=True)
    save(OUTPUT / 'plan.json', plan)
    for path in sources:
        target = OUTPUT / 'source' / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, target)
    print(json.dumps(dict(cases=len(cases), actual_sumo_runs=plan['expected_actual_sumo_runs'],
                          plan_sha256=digest(OUTPUT / 'plan.json'))), flush=True)


def verify_step(reference, optimized, state, previous, step, packed):
    for key in IDENTICAL:
        np.testing.assert_equal(reference[key], optimized[key], err_msg=f'{step}/{key}')
    if optimized['performance_model_transitions'] > reference['performance_model_transitions']:
        raise ValueError('Reported internal planning work increased')
    if len(reference['candidate_evidence']) != len(optimized['candidate_evidence']):
        raise ValueError('Candidate evaluation order changed')
    bank_prunes = 0
    for original, fast in zip(reference['candidate_evidence'], optimized['candidate_evidence']):
        for key in ('composition', 'weight', 'command', 'checked_scenarios', 'failed_scenarios', 'forecast_error'):
            np.testing.assert_equal(original[key], fast[key])
        if len(original['differences']) != len(fast['differences']):
            raise ValueError('Forecast count differs')
        for path, (full, value) in enumerate(zip(original['differences'], fast['differences'])):
            full, value = np.asarray(full), np.asarray(value)
            if path in fast['lower_bound_scenarios']:
                if not np.all(value <= full + 1e-9) or not np.any(value > 1e-10) or not np.any(full > 1e-10):
                    raise ValueError('Invalid physical rejection bound')
                bank_prunes += 1
            else:
                np.testing.assert_array_equal(full, value)
    if optimized['portfolio_joint_action_admitted']:
        if len(optimized['selected_bank_differences']) != 33 or not np.all(np.asarray(optimized['selected_bank_differences']) <= 1e-10):
            raise ValueError('Admitted action lacks the complete bank')
        if optimized['candidate_evidence'][-1]['lower_bound_scenarios']:
            raise ValueError('A bound authorized execution')
    pruned = optimized['nominal_pruned_bounds']
    if pruned:
        base = baseline_suffix(state, previous, step, packed)[0]
        for item in pruned:
            full = action_then_baseline(state, previous, step, np.asarray(item['command']), packed)[0] - base
            lower = np.asarray(item['lower_difference'])
            if not np.all(lower <= full + 1e-9) or not np.any(lower > 1e-10) or not np.any(full > 1e-10):
                raise ValueError('Invalid nominal rejection bound')
    return len(pruned), bank_prunes


def run():
    torch.set_num_threads(1)
    if (OUTPUT / 'episodes.json').exists():
        raise FileExistsError('Preserve all prior audit runs')
    plan = json.loads((OUTPUT / 'plan.json').read_text())
    for path, expected in plan['source_sha256'].items():
        if digest(ROOT / path) != expected or digest(OUTPUT / 'source' / path) != expected:
            raise ValueError('Frozen audit source changed: ' + path)
    require_committed([OUTPUT / 'plan.json', *[ROOT / p for p in plan['source_sha256']]])
    verify_manifest(STUDY)
    if digest(STUDY / 'protocol.json') != plan['study_protocol_sha256'] or digest(STUDY / 'episodes.json') != plan['study_episodes_sha256']:
        raise ValueError('Source study changed')
    historical = {(r['initial_index'], r['disturbance'], r['controller']): r
                  for r in json.loads((STUDY / 'episodes.json').read_text())}
    models = {}
    for seed, item in plan['models'].items():
        if digest(ROOT / item['path']) != item['sha256']:
            raise ValueError('Training model changed')
        models[seed] = load_coupled(ROOT / item['path'])[0]
    def gate(model, implementation):
        return RepairPortfolioGate(model) if implementation == 'reference' else FastRepairPortfolioGate(model, implementation == 'bounded')
    warm = np.asarray([[-80., 12.], [-85., 13.], [-120., 11.]])
    for model in models.values():
        for implementation in IMPLEMENTATIONS:
            gate(model, implementation).select(warm, np.zeros(3))
    rows, pairs, packed = [], [], parameters(MergeConfig())
    with DisturbedMergeEnv() as env:
        for ci, case in enumerate(plan['cases']):
            index, mode, method = case['index'], case['mode'], case['method']
            archived = historical[index, mode, method]
            archived_trace = json.loads((STUDY / f'shard_{index % 2}/trace_{index}_{mode}_{method}.json').read_text())
            model = models[method.rsplit('_', 1)[1]]
            results = {}
            shift = ci % 3
            for implementation in IMPLEMENTATIONS[shift:] + IMPLEMENTATIONS[:shift]:
                cpu_started = time.process_time()
                row, trace, initial = run_episode(env, archived['seed'], archived['scenario'], mode,
                                                 lambda x, m=model, impl=implementation: gate(m, impl), True)
                cpu_ms = (time.process_time() - cpu_started) * 1000.
                row.update(controller=method, initial=initial.tolist(), initial_index=index, implementation=implementation,
                           process_cpu_ms=cpu_ms, internal_transitions=sum(t['performance_model_transitions'] for t in trace),
                           optimistic_transitions=sum(t.get('optimistic_forecast_transitions', 0) for t in trace),
                           max_exact_cache_entries=max(t.get('exact_cache_entries', 0) for t in trace),
                           learned_action_steps=sum(t['portfolio_joint_action_admitted'] for t in trace),
                           repair_action_steps=sum(t['repair_action_admitted'] for t in trace))
                # Preserve the simulator output before any potentially failing check.
                save(OUTPUT / f'trace_{ci}_{implementation}.json', trace)
                rows.append(row)
                atomic_bytes(OUTPUT / 'episodes.json', (json.dumps(rows, indent=2) + '\n').encode())
                trace_audit(trace, row)
                for key in ('initial', 'per_vehicle_finish', 'per_vehicle_variation', 'completed', 'collision', 'aborted',
                            'contract_violations', 'learned_action_steps', 'repair_action_steps'):
                    np.testing.assert_equal(row[key], archived[key], err_msg=f'{ci}/{implementation}/{key}')
                if len(trace) != len(archived_trace):
                    raise ValueError('Archived trajectory length changed')
                for left, right in zip(archived_trace, trace):
                    for key in IDENTICAL:
                        np.testing.assert_equal(left[key], right[key], err_msg=f'{ci}/{implementation}/{key}')
                results[implementation] = row, trace
            reference = results['reference'][1]
            state, previous = np.asarray(archived['initial']), np.zeros(3)
            nominal_prunes, bank_prunes = 0, 0
            for step, original in enumerate(reference):
                for implementation in ('cached', 'bounded'):
                    n, b = verify_step(original, results[implementation][1][step], state, previous, step, packed)
                    nominal_prunes += n
                    bank_prunes += b
                state, previous = np.asarray(original['x']), np.asarray(original['command'])
            pairs.append(dict(case=case, decision_steps=len(reference), nominal_prunes=nominal_prunes, bank_prunes=bank_prunes))
            print(json.dumps(dict(completed_cases=len(pairs), expected_cases=len(plan['cases']), actual_sumo_runs=len(rows),
                                  verified_nominal_prunes=nominal_prunes, verified_bank_prunes=bank_prunes)), flush=True)
    if len(rows) != plan['expected_actual_sumo_runs']:
        raise ValueError('Incomplete runtime study')
    timing = {}
    for implementation in IMPLEMENTATIONS:
        selected = [r for r in rows if r['implementation'] == implementation]
        values = np.asarray([value for row in selected for value in row['controller_ms']])
        timing[implementation] = dict(total_controller_ms=float(values.sum()), p50_ms=float(np.median(values)),
            p95_ms=float(np.percentile(values, 95)), p99_ms=float(np.percentile(values, 99)), max_ms=float(values.max()),
            over_500ms=int((values > 500).sum()), decision_steps=len(values),
            process_cpu_ms=sum(r['process_cpu_ms'] for r in selected),
            internal_transitions=sum(r['internal_transitions'] for r in selected),
            optimistic_transitions=sum(r['optimistic_transitions'] for r in selected),
            max_exact_cache_entries=max(r['max_exact_cache_entries'] for r in selected))
    report = dict(actual_sumo_runs=len(rows), matched_cases=len(pairs), matched_decision_steps=sum(p['decision_steps'] for p in pairs),
                  timings=timing, paired_cases=pairs, additional_independent_initials=0,
                  risk_certified=False, hard_real_time_certified=False, timing_scope=plan['timing_scope'],
                  plan_sha256=digest(OUTPUT / 'plan.json'), episodes_sha256=digest(OUTPUT / 'episodes.json'))
    save(OUTPUT / 'audit.json', report)
    save(OUTPUT / 'sha256.json', {p.relative_to(OUTPUT).as_posix(): digest(p) for p in OUTPUT.rglob('*')
                                if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps({k: v for k, v in report.items() if k != 'paired_cases'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('freeze', 'run'))
    args = parser.parse_args()
    (freeze if args.command == 'freeze' else run)()
