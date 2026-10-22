"""Check exact decisions and measure full-call latency/CPU for two-worker gating."""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from coopgate.bounded_portfolio_v19 import BoundedJointPortfolioGate
from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.disturbance_v9 import DisturbedMergeEnv
from coopgate.parallel_portfolio_v19 import ParallelJointPortfolioGate
from experiment_ego_v13 import run_episode
from probe_disturbed_gate_v19 import digest, save
from start_final_v19 import verify_manifest
from verify_development_v14 import trace_audit

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/v19/bounded_gate_audit'
IDENTICAL = ('in_set', 'certified', 'intervention', 'nominal_joint_action_admitted',
             'portfolio_joint_action_admitted', 'composition', 'candidate_count', 'nominal_candidate_count',
             'candidate_evidence', 'selected_bank_differences', 'nominal_pruned_bounds',
             'bank_pruned_count', 'max_failed_scenarios')


class MeasuredGate:
    def __init__(self, model, parallel):
        self.controller = ParallelJointPortfolioGate(model, workers=2) if parallel else BoundedJointPortfolioGate(model)

    def select(self, state, previous, unused=None):
        started, cpu_started = time.perf_counter(), time.process_time()
        command, info = self.controller.select(state, previous, unused)
        cpu, wall = time.process_time() - cpu_started, time.perf_counter() - started
        info.update(measured_select_wall_ms=1000 * wall, measured_select_process_cpu_ms=1000 * cpu)
        return command, info


def load_models(records):
    models = {}
    for seed, record in records.items():
        assert digest(ROOT / record['path']) == record['sha256']
        models[seed], _ = load_coupled(ROOT / record['path'])
    return models


def same_decision(left, right):
    for key in IDENTICAL:
        assert left[key] == right[key], key
    assert right['unused_speculative_errors'] == 0 and right['work_counts_complete'] is True
    if right['portfolio_joint_action_admitted']:
        assert len(right['selected_bank_differences']) == 17
        assert np.all(np.array(right['selected_bank_differences']) <= 1e-10)


def preflight(output):
    """Same six already-recorded slow states; not another SUMO efficacy study."""
    torch.set_num_threads(1)
    verify_manifest(SOURCE)
    profile_dir = ROOT / 'artifacts/v19/runtime_profile'
    verify_manifest(profile_dir)
    profile = json.loads((profile_dir / 'profile.json').read_text())
    plan = json.loads((SOURCE / 'plan.json').read_text())
    archived_rows = json.loads((SOURCE / 'episodes.json').read_text())
    lookup = {(r['initial_index'], r['disturbance'], r['controller'], r['implementation']): r for r in archived_rows}
    models = load_models(plan['models'])
    warm = np.array([[-80., 12.], [-85., 13.], [-120., 11.]])
    for model in models.values():
        for parallel in (False, True):
            MeasuredGate(model, parallel).select(warm, np.zeros(3))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for index, record in enumerate(profile['records']):
        ci, step, seed = record['case_index'], record['step'], record['seed']
        initial_index, mode, name = plan['cases'][ci]
        archived = lookup[initial_index, mode, name, 'bounded']
        trace = json.loads((SOURCE / f'trace_{ci}_bounded.json').read_text())
        state = np.array(archived['initial'] if step == 0 else trace[step - 1]['x'])
        previous = np.zeros(3) if step == 0 else np.array(trace[step - 1]['command'])
        results = {}
        for implementation in (('bounded', 'parallel') if index % 2 == 0 else ('parallel', 'bounded')):
            gate = MeasuredGate(models[seed], implementation == 'parallel')
            gate.controller.step = step
            command, info = gate.select(state, previous)
            np.testing.assert_array_equal(command, trace[step]['command'])
            results[implementation] = info
        same_decision(results['bounded'], results['parallel'])
        records.append(dict(seed=seed, case_index=ci, step=step, results=results))
        save(output / 'records.json', records)
        print(json.dumps(dict(case=ci, step=step,
            wall_ms={k: v['measured_select_wall_ms'] for k, v in results.items()},
            cpu_ms={k: v['measured_select_process_cpu_ms'] for k, v in results.items()})), flush=True)
    report = dict(paired_states=len(records), all_decisions_equal=True, actual_new_sumo_runs=0,
                  scope='Post-hoc slow-state development preflight; no runtime-distribution or efficacy claim',
                  profile_sha256=digest(profile_dir / 'profile.json'),
                  source_sha256={path: digest(ROOT / path) for path in ('src/coopgate/parallel_portfolio_v19.py',
                                                                      'tools/audit_parallel_gate_v19.py')})
    save(output / 'audit.json', report)
    for path in report['source_sha256']:
        shutil.copy2(ROOT / path, output / Path(path).name)
    save(output / 'sha256.json', {p.name: digest(p) for p in output.iterdir() if p.is_file() and p.name != 'sha256.json'})


def freeze(output):
    verify_manifest(SOURCE)
    old = json.loads((SOURCE / 'plan.json').read_text())
    sources = list((ROOT / 'src/coopgate').glob('*.py')) + [Path(__file__),
               ROOT / 'tools/experiment_ego_v13.py', ROOT / 'tools/probe_disturbed_gate_v19.py',
               ROOT / 'tools/verify_development_v14.py']
    plan = dict(purpose='Actual SUMO decision equivalence and complete-call timing for two-worker disturbance evaluation',
                cases=old['cases'], models=old['models'], actual_sumo_runs_planned=2 * len(old['cases']),
                source_audit_sha256=digest(SOURCE / 'audit.json'), source_episodes_sha256=digest(SOURCE / 'episodes.json'),
                workers=2, source_sha256={p.resolve().relative_to(ROOT).as_posix(): digest(p) for p in sources},
                timing_scope='Warmed kernels, alternating implementation order, external select wall/CPU timers include pool startup, joins, teardown and metadata construction. Other frozen studies run concurrently. Process CPU excludes the separate SUMO server and may be coarsely quantized on Windows.',
                semantic_scope='Every logical candidate and disturbance check remains in the original order; accepted candidates still have all 17 full scenarios; extra speculative work is counted.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'plan.json', plan)
    for source in plan['source_sha256']:
        target = output / 'source' / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source, target)
    print(json.dumps(dict(plan_sha256=digest(output / 'plan.json'), actual_sumo_runs_planned=plan['actual_sumo_runs_planned'])), flush=True)


def run(output):
    torch.set_num_threads(1)
    output = Path(output)
    if (output / 'episodes.json').exists():
        raise FileExistsError('Preserve previous execution records')
    plan = json.loads((output / 'plan.json').read_text())
    for path, expected in plan['source_sha256'].items():
        assert digest(ROOT / path) == digest(output / 'source' / path) == expected, path
    verify_manifest(SOURCE)
    assert digest(SOURCE / 'audit.json') == plan['source_audit_sha256']
    assert digest(SOURCE / 'episodes.json') == plan['source_episodes_sha256']
    historical = {(r['initial_index'], r['disturbance'], r['controller'], r['implementation']): r
                  for r in json.loads((SOURCE / 'episodes.json').read_text())}
    models = load_models(plan['models'])
    warm = np.array([[-80., 12.], [-85., 13.], [-120., 11.]])
    for model in models.values():
        for parallel in (False, True):
            MeasuredGate(model, parallel).select(warm, np.zeros(3))
    rows, pairs = [], []
    with DisturbedMergeEnv() as env:
        for ci, (index, mode, method) in enumerate(plan['cases']):
            archived = historical[index, mode, method, 'bounded']
            model = models[method.rsplit('_', 1)[1]]
            results = {}
            for implementation in (('bounded', 'parallel') if ci % 2 == 0 else ('parallel', 'bounded')):
                factory = lambda x, m=model, p=(implementation == 'parallel'): MeasuredGate(m, p)
                row, trace, initial = run_episode(env, archived['seed'], archived['scenario'], mode, factory, True)
                row.update(controller=method, implementation=implementation, initial=initial.tolist(), initial_index=index,
                           measured_wall_ms=[t['measured_select_wall_ms'] for t in trace],
                           measured_cpu_ms=[t['measured_select_process_cpu_ms'] for t in trace],
                           internal_transitions=sum(t['performance_model_transitions'] for t in trace),
                           forecast_transitions=sum(t['performance_forecast_transitions'] for t in trace),
                           optimistic_transitions=sum(t['optimistic_forecast_transitions'] for t in trace),
                           speculative_scenarios=sum(t.get('unused_speculative_scenarios', 0) for t in trace),
                           unused_speculative_errors=sum(t.get('unused_speculative_errors', 0) for t in trace),
                           maximum_cache_entries=max(t['exact_cache_entries'] for t in trace),
                           learned_action_steps=sum(t['portfolio_joint_action_admitted'] for t in trace))
                trace_audit(trace, row)
                for field in ('initial', 'per_vehicle_finish', 'per_vehicle_variation', 'completed', 'collision', 'aborted', 'contract_violations'):
                    np.testing.assert_array_equal(row[field], archived[field], err_msg=f'{ci}/{implementation}/{field}')
                results[implementation] = trace
                rows.append(row)
                save(output / f'trace_{ci}_{implementation}.json', trace)
                save(output / 'episodes.json', rows)
            assert len(results['bounded']) == len(results['parallel'])
            for left, right in zip(results['bounded'], results['parallel']):
                for key in ('command', 'x', 't', 'disturbance'):
                    assert left[key] == right[key], (ci, key)
                same_decision(left, right)
            pairs.append(dict(case_index=ci, decision_steps=len(results['bounded'])))
            print(json.dumps(dict(completed_pairs=len(pairs), actual_sumo_runs=len(rows))), flush=True)
    assert len(rows) == plan['actual_sumo_runs_planned']
    timings = {}
    for implementation in ('bounded', 'parallel'):
        data = [r for r in rows if r['implementation'] == implementation]
        wall = [v for r in data for v in r['measured_wall_ms']]
        cpu = [v for r in data for v in r['measured_cpu_ms']]
        timings[implementation] = dict(p50_ms=float(np.median(wall)), p95_ms=float(np.percentile(wall, 95)),
            p99_ms=float(np.percentile(wall, 99)), max_ms=float(max(wall)), total_select_wall_ms=float(sum(wall)),
            total_select_process_cpu_ms=float(sum(cpu)), decision_steps=len(wall), steps_over_500ms=sum(v > 500 for v in wall),
            internal_transitions=sum(r['internal_transitions'] for r in data),
            forecast_transitions=sum(r['forecast_transitions'] for r in data),
            optimistic_transitions=sum(r['optimistic_transitions'] for r in data),
            speculative_scenarios=sum(r['speculative_scenarios'] for r in data),
            unused_speculative_errors=sum(r['unused_speculative_errors'] for r in data),
            maximum_cache_entries=max(r['maximum_cache_entries'] for r in data))
    report = dict(actual_sumo_replays=len(rows), exact_paired_cases=len(pairs),
                  exact_paired_decisions=sum(r['decision_steps'] for r in pairs), pairs=pairs, timings=timings,
                  all_commands_and_admissions_match=True, all_selected_scenarios_fully_evaluated=True,
                  risk_certified=False, hard_real_time_certified=False,
                  timing_scope=plan['timing_scope'], episodes_sha256=digest(output / 'episodes.json'),
                  plan_sha256=digest(output / 'plan.json'))
    save(output / 'audit.json', report)
    save(output / 'sha256.json', {p.relative_to(output).as_posix(): digest(p)
         for p in output.rglob('*') if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps({key: value for key, value in report.items() if key != 'pairs'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('preflight', 'freeze', 'run'))
    parser.add_argument('output')
    args = parser.parse_args()
    {'preflight': preflight, 'freeze': freeze, 'run': run}[args.command](args.output)
