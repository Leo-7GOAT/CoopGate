"""Freeze, execute and assemble paired SUMO gate studies with disjoint seed sets.

Risk certification is deliberately not performed by this executor. It exports
one row per independent initial and method, including actual closed-loop use.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from coopgate.advantage_gate_v18 import JointAdvantageGate
from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.cooperative_v15 import JointPolicyGate
from coopgate.disturbance_v9 import DisturbedMergeEnv
from coopgate.disturbed_advantage_v19 import DisturbanceBankGate
from coopgate.joint_planner_v15 import JointRolloutGate
from coopgate.portfolio_gate_v19 import JointPortfolioGate
from experiment_ego_v13 import run_episode
from probe_disturbed_gate_v19 import digest, save
from verify_development_v14 import trace_audit

ROOT = Path(__file__).resolve().parents[1]


def factories(protocol):
    models = {}
    for seed, record in protocol['models'].items():
        if digest(ROOT / record['path']) != record['sha256']:
            raise ValueError('Frozen model changed')
        models[seed], _ = load_coupled(ROOT / record['path'])
    result = {}
    for name, spec in protocol['spec']['methods'].items():
        kind = spec['kind']
        if kind == 'baseline':
            result[name] = lambda x: JointRolloutGate()
        elif kind == 'raw':
            result[name] = lambda x, m=models[str(spec['seed'])]: JointPolicyGate(m)
        elif kind == 'nominal':
            result[name] = lambda x, m=models[str(spec['seed'])]: JointAdvantageGate(m)
        elif kind == 'bank':
            result[name] = lambda x, m=models[str(spec['seed'])], b=spec['bank']: DisturbanceBankGate(m, b)
        elif kind == 'portfolio':
            result[name] = lambda x, m=models[str(spec['seed'])], k=spec['max_failed_scenarios']: JointPortfolioGate(m, k)
        else:
            raise ValueError('Unknown method: ' + kind)
    return result


def regime_for(spec, index):
    if spec['sampling'] == 'stratified':
        return 'nominal' if index < spec['n_initials'] // 2 else 'dense'
    if spec['sampling'] == 'iid_mixture':
        # Independent, declared stream for regime labels; not index quotas.
        rng = np.random.default_rng(np.random.SeedSequence([spec['seed_base'] + index, 19191]))
        return ('nominal', 'dense')[rng.integers(2)]
    raise ValueError('Unknown scenario sampling')


def freeze(spec_path, directory):
    spec_path, directory = Path(spec_path), Path(directory)
    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    assert 0 < spec['shards'] <= spec['n_initials']
    assert 'baseline' in spec['methods'] and spec['methods']['baseline'] == {'kind': 'baseline'}
    expected = spec['n_initials'] * len(spec['modes']) * len(spec['methods'])
    assert expected == spec['expected_actual_sumo_runs']
    historical = json.loads((ROOT / 'artifacts/v18/independent_study/protocol.json').read_text())
    seeds = sorted({str(record['seed']) for record in spec['methods'].values() if 'seed' in record})
    models = {seed: historical['models'][seed] for seed in seeds}
    protocol = dict(spec=spec, spec_sha256=digest(spec_path), models=models)
    factories(protocol)
    for i in range(spec['n_initials']):
        regime_for(spec, i)
    sources = list((ROOT / 'src/coopgate').glob('*.py')) + [
        Path(__file__), ROOT / 'tools/experiment_ego_v13.py', ROOT / 'tools/verify_development_v14.py',
        ROOT / 'tools/probe_disturbed_gate_v19.py']
    protocol['source_sha256'] = {p.resolve().relative_to(ROOT).as_posix(): digest(p) for p in sources}
    directory.mkdir(parents=True, exist_ok=False)
    save(directory / 'protocol.json', protocol)
    for source in protocol['source_sha256']:
        destination = directory / 'source' / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source, destination)
    print(json.dumps(dict(directory=str(directory), expected_actual_sumo_runs=expected,
                          protocol_sha256=digest(directory / 'protocol.json'))), flush=True)


def read_protocol(directory):
    directory = Path(directory)
    protocol = json.loads((directory / 'protocol.json').read_text(encoding='utf-8'))
    for source, expected in protocol['source_sha256'].items():
        if digest(directory / 'source' / source) != expected or digest(ROOT / source) != expected:
            raise ValueError('Saved/runtime source changed: ' + source)
    return directory, protocol


def worker(directory, shard):
    torch.set_num_threads(1)
    directory, protocol = read_protocol(directory)
    spec = protocol['spec']
    if not 0 <= shard < spec['shards']:
        raise ValueError('Invalid shard')
    fs = factories(protocol)
    out = directory / f'shard_{shard}'
    out.mkdir(exist_ok=False)
    rows = []
    ids = list(range(shard, spec['n_initials'], spec['shards']))
    started = time.perf_counter()
    warm = np.array([[-80., 12.], [-85., 13.], [-120., 11.]])
    for factory in fs.values():
        factory(warm).select(warm, np.zeros(3))
    with DisturbedMergeEnv() as env:
        for index in ids:
            seed = spec['seed_base'] + index
            regime = regime_for(spec, index)
            for mi, mode in enumerate(spec['modes']):
                for method in np.random.default_rng(seed + mi + 1919).permutation(list(fs)):
                    row, trace, initial = run_episode(env, seed, regime, mode, fs[method], True)
                    method_spec = spec['methods'][method]
                    cost = np.array(row['per_vehicle_finish']) + .05 * np.array(row['per_vehicle_variation'])
                    used = sum(t.get('nominal_joint_action_admitted', method_spec['kind'] == 'raw') for t in trace)
                    adopted = [t for t in trace if t.get('portfolio_joint_action_admitted', False)]
                    banks_checked = [len(t['selected_bank_differences']) for t in adopted]
                    failures = [int(np.any(np.array(t['selected_bank_differences']) > 1e-10, axis=(1, 2)).sum())
                                for t in adopted]
                    row.update(controller=str(method), initial=initial.tolist(), initial_index=index,
                               per_vehicle_cost=cost.tolist(), team_cost=float(cost.mean()),
                               learned_action_steps=used, control_steps=len(trace),
                               min_adopted_bank_checks=min(banks_checked, default=None),
                               max_adopted_failed_scenarios=max(failures, default=None),
                               chosen_compositions=[int(t.get('composition', 0)) for t in trace],
                               forecast_transitions=sum(t.get('performance_forecast_transitions', 0) for t in trace),
                               internal_transitions=sum(t.get('performance_model_transitions', 0) for t in trace))
                    failure = not row['completed'] or row['collision'] or row['aborted'] or row['contract_violations']
                    save_trace = spec['all_traces_saved'] or index in spec['audit_initial_indices'] or failure
                    row['trace_saved'] = bool(save_trace)
                    if save_trace:
                        save(out / f'trace_{index}_{mode}_{method}.json', trace)
                    rows.append(row)
                    save(out / 'episodes.json', rows)
                    trace_audit(trace, row)
                    if adopted:
                        assert min(banks_checked) == 17
                        assert max(failures) <= method_spec['max_failed_scenarios']
                if spec['n_initials'] <= 32 or index % 16 < spec['shards']:
                    print(json.dumps(dict(shard=shard, index=index, mode=mode, actual_sumo_runs=len(rows),
                                          elapsed_s=time.perf_counter() - started)), flush=True)
    assert len(rows) == len(ids) * len(spec['modes']) * len(fs)
    save(out / 'complete.json', dict(initial_indices=ids, actual_sumo_runs=len(rows),
                                    episodes_sha256=digest(out / 'episodes.json'),
                                    protocol_sha256=digest(directory / 'protocol.json')))


def finalize(directory):
    directory, protocol = read_protocol(directory)
    spec = protocol['spec']
    rows = []
    for shard in range(spec['shards']):
        out = directory / f'shard_{shard}'
        done = json.loads((out / 'complete.json').read_text())
        assert done['initial_indices'] == list(range(shard, spec['n_initials'], spec['shards']))
        assert done['protocol_sha256'] == digest(directory / 'protocol.json')
        assert done['episodes_sha256'] == digest(out / 'episodes.json')
        part = json.loads((out / 'episodes.json').read_text())
        assert len(part) == done['actual_sumo_runs']
        rows.extend(part)
    lookup = {(r['initial_index'], r['disturbance'], r['controller']): r for r in rows}
    expected = {(i, mode, method) for i in range(spec['n_initials'])
                for mode in spec['modes'] for method in spec['methods']}
    assert set(lookup) == expected and len(rows) == len(lookup) == spec['expected_actual_sumo_runs']
    summaries = {}
    cluster_records = []
    for method, method_spec in spec['methods'].items():
        data = [r for r in rows if r['controller'] == method]
        differences = []
        adopted, harmed, failed = [], [], []
        for index in range(spec['n_initials']):
            cluster, has_adoption, has_failure = [], False, False
            for mode in spec['modes']:
                row, base = lookup[index, mode, method], lookup[index, mode, 'baseline']
                assert row['scenario'] == base['scenario'] == regime_for(spec, index)
                np.testing.assert_allclose(row['initial'], base['initial'], atol=1e-8, rtol=0)
                delta = np.column_stack((row['per_vehicle_cost'], row['per_vehicle_finish'])) - np.column_stack(
                    (base['per_vehicle_cost'], base['per_vehicle_finish']))
                cluster.append(delta)
                has_adoption |= row['learned_action_steps'] > 0
                has_failure |= not row['completed'] or row['collision'] or row['aborted'] or bool(row['contract_violations'])
                if row['trace_saved']:
                    trace_path = directory / f"shard_{index % spec['shards']}" / f'trace_{index}_{mode}_{method}.json'
                    trace_audit(json.loads(trace_path.read_text()), row)
                if method_spec['kind'] == 'portfolio':
                    if row['learned_action_steps']:
                        assert row['min_adopted_bank_checks'] == 17
                        assert row['max_adopted_failed_scenarios'] <= method_spec['max_failed_scenarios']
                    if mode == 'clean' or (method_spec['max_failed_scenarios'] == 0 and mode == 'bias'):
                        assert np.all(delta <= 1e-9), (index, mode, method, delta)
                if method_spec['kind'] in ('nominal', 'bank', 'portfolio') and row['learned_action_steps'] == 0:
                    np.testing.assert_allclose(delta, 0., atol=1e-9, rtol=0)
            delta = np.array(cluster)
            harm = bool(np.any(delta > 1e-9)) or has_failure
            gain = float(-delta[:, :, 0].mean())
            cluster_records.append(dict(initial_index=index, seed=spec['seed_base'] + index,
                                        scenario=regime_for(spec, index), method=method,
                                        adopted=bool(has_adoption), harmful=harm, failed=bool(has_failure),
                                        mean_team_gain=gain,
                                        worst_component_loss=float(delta.max()),
                                        differences=delta.tolist()))
            differences.append(delta)
            adopted.append(has_adoption)
            harmed.append(harm)
            failed.append(has_failure)
        delta = np.array(differences)
        adopted, harmed = np.array(adopted), np.array(harmed)
        summaries[method] = dict(
            mean_team_cost=float(np.mean([r['team_cost'] for r in data])),
            per_vehicle_cost_gain=(-delta[:, :, :, 0].mean((0, 1))).tolist(),
            per_vehicle_time_gain=(-delta[:, :, :, 1].mean((0, 1))).tolist(),
            harmful_initials=int(harmed.sum()), failed_initials=int(sum(failed)),
            adopted_initials=int(adopted.sum()), harmful_adopted_initials=int((harmed & adopted).sum()),
            harm_by_mode={mode: int(np.any(delta[:, k] > 1e-9, axis=(1, 2)).sum()) for k, mode in enumerate(spec['modes'])},
            learned_action_steps=sum(r['learned_action_steps'] for r in data),
            control_steps=sum(r['control_steps'] for r in data),
            completed=sum(r['completed'] for r in data), runs=len(data),
            collisions=sum(r['collision'] for r in data), aborted=sum(r['aborted'] for r in data),
            contract_violations=sum(r['contract_violations'] for r in data),
            controller_p95_ms=float(np.percentile([v for r in data for v in r['controller_ms']], 95)),
            controller_max_ms=float(max(v for r in data for v in r['controller_ms'])))
    rows.sort(key=lambda r: (r['initial_index'], r['disturbance'], r['controller']))
    save(directory / 'episodes.json', rows)
    save(directory / 'clusters.json', cluster_records)
    summary = dict(purpose=spec['purpose'], actual_sumo_runs=len(rows), results=summaries,
                   disturbance_risk_certified=False, protocol_sha256=digest(directory / 'protocol.json'),
                   episodes_sha256=digest(directory / 'episodes.json'), clusters_sha256=digest(directory / 'clusters.json'))
    save(directory / 'summary.json', summary)
    save(directory / 'sha256.json', {p.relative_to(directory).as_posix(): digest(p)
                                   for p in sorted(directory.rglob('*')) if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('freeze')
    p.add_argument('spec')
    p.add_argument('directory')
    p = sub.add_parser('worker')
    p.add_argument('directory')
    p.add_argument('shard', type=int)
    p = sub.add_parser('finalize')
    p.add_argument('directory')
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze(args.spec, args.directory)
    elif args.command == 'worker':
        worker(args.directory, args.shard)
    else:
        finalize(args.directory)
