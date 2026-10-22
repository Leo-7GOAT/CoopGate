"""Paired actual SUMO replays for exact-command caching and timing tradeoffs."""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from coopgate.cached_portfolio_v19 import CachedJointPortfolioGate
from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.disturbance_v9 import DisturbedMergeEnv
from coopgate.portfolio_gate_v19 import JointPortfolioGate
from experiment_ego_v13 import run_episode
from probe_disturbed_gate_v19 import digest, save
from verify_development_v14 import trace_audit

ROOT = Path(__file__).resolve().parents[1]


def freeze(directory):
    old = ROOT / 'artifacts/v19/portfolio_probe'
    protocol = json.loads((old / 'protocol.json').read_text())
    audit = json.loads((ROOT / 'artifacts/v19/portfolio_audit/plan.json').read_text())
    cases = {(i, mode, f'portfolio0_{seed}') for i in (0, 16)
             for mode in protocol['spec']['modes'] for seed in protocol['models']}
    cases.update(tuple(c) for c in audit['selected_nonvacuous_cases'])
    cases.update(tuple(c) for c in audit['all_strict_harm_cases'])
    sources = list((ROOT / 'src/coopgate').glob('*.py')) + [Path(__file__),
        ROOT / 'tools/experiment_ego_v13.py', ROOT / 'tools/probe_disturbed_gate_v19.py', ROOT / 'tools/verify_development_v14.py']
    plan = dict(purpose='Post-development exact-behavior and paired timing audit; not new independent efficacy or risk evidence',
                study_protocol_sha256=digest(old / 'protocol.json'), study_episodes_sha256=digest(old / 'episodes.json'),
                cases=sorted(cases), actual_sumo_runs_planned=2 * len(cases), models=protocol['models'],
                source_sha256={p.resolve().relative_to(ROOT).as_posix(): digest(p) for p in sources},
                timing_scope='Alternating reference/cached order per case, all functions warmed before measurement. Other frozen experiments run concurrently; no isolated-host or hard real-time claim.')
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'plan.json', plan)
    for source in plan['source_sha256']:
        target = out / 'source' / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source, target)
    print(json.dumps(dict(cases=len(cases), actual_sumo_runs_planned=2 * len(cases), plan_sha256=digest(out / 'plan.json'))), flush=True)


def run(directory):
    torch.set_num_threads(1)
    out = Path(directory)
    plan = json.loads((out / 'plan.json').read_text())
    if (out / 'episodes.json').exists():
        raise FileExistsError('Preserve earlier audit executions')
    for path, expected in plan['source_sha256'].items():
        assert digest(ROOT / path) == digest(out / 'source' / path) == expected, path
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
        for gate in (JointPortfolioGate, CachedJointPortfolioGate):
            gate(model).select(warm, np.zeros(3))
    rows, pairs = [], []
    with DisturbedMergeEnv() as env:
        for case_index, (index, mode, method) in enumerate(plan['cases']):
            archived = historical[index, mode, method]
            model = models[method.rsplit('_', 1)[1]]
            executions = {}
            order = ('reference', 'cached') if case_index % 2 == 0 else ('cached', 'reference')
            for implementation in order:
                gate = JointPortfolioGate if implementation == 'reference' else CachedJointPortfolioGate
                started = time.perf_counter()
                row, trace, initial = run_episode(env, archived['seed'], archived['scenario'], mode,
                                                 lambda x, g=gate, m=model: g(m), True)
                row.update(controller=method, implementation=implementation, initial=initial.tolist(),
                           initial_index=index, wall_execution_s=time.perf_counter() - started,
                           internal_transitions=sum(t['performance_model_transitions'] for t in trace),
                           learned_action_steps=sum(t['portfolio_joint_action_admitted'] for t in trace))
                trace_audit(trace, row)
                for key in ('initial', 'per_vehicle_finish', 'per_vehicle_variation', 'completed',
                            'collision', 'aborted', 'contract_violations'):
                    np.testing.assert_array_equal(row[key], archived[key], err_msg=f'{case_index}/{implementation}/{key}')
                rows.append(row)
                save(out / f'trace_{case_index}_{implementation}.json', trace)
                save(out / 'episodes.json', rows)
                executions[implementation] = row, trace
            original, fast = executions['reference'], executions['cached']
            assert len(original[1]) == len(fast[1])
            excluded = {'gate_ms', 'performance_model_transitions'}
            for left, right in zip(original[1], fast[1]):
                for key in left:
                    if key not in excluded:
                        assert left[key] == right[key], (case_index, key)
                assert right['performance_model_transitions'] <= left['performance_model_transitions']
            pairs.append(dict(index=index, mode=mode, method=method, exact_decision_steps=len(original[1]),
                              adopted_steps=original[0]['learned_action_steps'],
                              reference_internal_transitions=original[0]['internal_transitions'],
                              cached_internal_transitions=fast[0]['internal_transitions']))
            print(json.dumps(dict(case=case_index, actual_sumo_runs=len(rows), exact_decision_steps=len(original[1]))), flush=True)
    assert len(rows) == plan['actual_sumo_runs_planned']
    timings = {}
    for name in ('reference', 'cached'):
        data = [r for r in rows if r['implementation'] == name]
        values = [v for r in data for v in r['controller_ms']]
        timings[name] = dict(p50_ms=float(np.median(values)), p95_ms=float(np.percentile(values, 95)),
                             p99_ms=float(np.percentile(values, 99)), max_ms=float(max(values)),
                             total_controller_ms=float(sum(values)), steps_over_500ms=sum(v > 500 for v in values),
                             decision_steps=len(values), internal_transitions=sum(r['internal_transitions'] for r in data))
    result = dict(actual_sumo_replays=len(rows), exact_paired_cases=len(pairs),
                  exact_paired_decision_steps=sum(p['exact_decision_steps'] for p in pairs),
                  timings=timings, pairs=pairs, risk_certified=False, hard_real_time_certified=False,
                  controller_time_reduction=1 - timings['cached']['total_controller_ms'] / timings['reference']['total_controller_ms'],
                  internal_work_reduction=1 - timings['cached']['internal_transitions'] / timings['reference']['internal_transitions'],
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
