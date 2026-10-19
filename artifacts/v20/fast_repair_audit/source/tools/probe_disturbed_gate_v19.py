"""Frozen, paired actual SUMO development probe for disturbance-bank gating."""
import argparse
import hashlib
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
from experiment_ego_v13 import run_episode
from verify_development_v14 import trace_audit

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def factories(records, bank):
    result = {'baseline': lambda x: JointRolloutGate()}
    for seed, record in records.items():
        path = ROOT / record['path']
        if digest(path) != record['sha256']:
            raise ValueError('Frozen model changed')
        model, meta = load_coupled(path)
        if meta['updates'] != meta['target_updates']:
            raise ValueError('Incomplete actor training')
        result['raw_' + seed] = lambda x, m=model: JointPolicyGate(m)
        result['nominal_' + seed] = lambda x, m=model: JointAdvantageGate(m)
        result['bank_' + seed] = lambda x, m=model: DisturbanceBankGate(m, bank)
    return result


def freeze(spec_path, directory, shards):
    spec_path, directory = Path(spec_path), Path(directory)
    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    if shards < 1 or shards > spec['n_initials']:
        raise ValueError('Invalid shard count')
    old = json.loads((ROOT / 'artifacts/v18/independent_study/protocol.json').read_text())
    models = {str(seed): old['models'][str(seed)] for seed in spec['models']}
    fs = factories(models, spec['bank'])
    expected = spec['n_initials'] * len(spec['modes']) * len(fs)
    if expected != spec['expected_actual_sumo_runs']:
        raise ValueError('Wrong frozen run budget')
    sources = list((ROOT / 'src/coopgate').glob('*.py')) + [
        Path(__file__), ROOT / 'tools/experiment_ego_v13.py', ROOT / 'tools/verify_development_v14.py']
    protocol = dict(spec=spec, spec_sha256=digest(spec_path), models=models,
                    methods=list(fs), shards=shards,
                    source_sha256={p.resolve().relative_to(ROOT).as_posix(): digest(p) for p in sources})
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
    if not 0 <= shard < protocol['shards']:
        raise ValueError('Invalid shard')
    fs = factories(protocol['models'], spec['bank'])
    out = directory / f'shard_{shard}'
    out.mkdir(exist_ok=False)
    rows = []
    ids = list(range(shard, spec['n_initials'], protocol['shards']))
    started = time.perf_counter()
    # Compile on a declared fixed state, without an actual SUMO evaluation.
    warm = np.array([[-80., 12.], [-85., 13.], [-120., 11.]])
    for factory in fs.values():
        factory(warm).select(warm, np.zeros(3))
    with DisturbedMergeEnv() as env:
        for index in ids:
            seed = spec['seed_base'] + index
            regime = 'nominal' if index < spec['n_initials'] // 2 else 'dense'
            for mode_index, mode in enumerate(spec['modes']):
                methods = np.random.default_rng(seed + mode_index + 1900).permutation(protocol['methods'])
                for method in methods:
                    row, trace, initial = run_episode(env, seed, regime, mode, fs[method], True)
                    cost = np.array(row['per_vehicle_finish']) + .05 * np.array(row['per_vehicle_variation'])
                    row.update(controller=str(method), initial=initial.tolist(), initial_index=index,
                               per_vehicle_cost=cost.tolist(), team_cost=float(cost.mean()),
                               learned_action_steps=sum(t.get('nominal_joint_action_admitted', False) for t in trace),
                               control_steps=len(trace),
                               forecast_transitions=sum(t.get('performance_forecast_transitions', 0) for t in trace),
                               internal_transitions=sum(t.get('performance_model_transitions', 0) for t in trace))
                    # Preserve outcomes before auditing, including a possible failing trace.
                    save(out / f'trace_{index}_{mode}_{method}.json', trace)
                    rows.append(row)
                    save(out / 'episodes.json', rows)
                    trace_audit(trace, row)
                print(json.dumps(dict(shard=shard, index=index, mode=mode, actual_sumo_runs=len(rows),
                                      elapsed_s=time.perf_counter() - started)), flush=True)
    if len(rows) != len(ids) * len(spec['modes']) * len(fs):
        raise RuntimeError('Incomplete probe')
    save(out / 'complete.json', dict(initial_indices=ids, actual_sumo_runs=len(rows),
                                    episodes_sha256=digest(out / 'episodes.json'),
                                    protocol_sha256=digest(directory / 'protocol.json')))


def finalize(directory):
    directory, protocol = read_protocol(directory)
    spec = protocol['spec']
    rows = []
    for shard in range(protocol['shards']):
        out = directory / f'shard_{shard}'
        done = json.loads((out / 'complete.json').read_text())
        assert done['initial_indices'] == list(range(shard, spec['n_initials'], protocol['shards']))
        assert done['protocol_sha256'] == digest(directory / 'protocol.json')
        assert done['episodes_sha256'] == digest(out / 'episodes.json')
        part = json.loads((out / 'episodes.json').read_text())
        assert len(part) == done['actual_sumo_runs']
        rows.extend(part)
    lookup = {(r['initial_index'], r['disturbance'], r['controller']): r for r in rows}
    expected = {(i, mode, method) for i in range(spec['n_initials'])
                for mode in spec['modes'] for method in protocol['methods']}
    assert set(lookup) == expected and len(rows) == len(lookup) == spec['expected_actual_sumo_runs']
    summaries = {}
    for method in protocol['methods']:
        data = [r for r in rows if r['controller'] == method]
        differences, admitted = [], []
        for index in range(spec['n_initials']):
            cluster = []
            has_adoption = False
            for mode in spec['modes']:
                row = lookup[index, mode, method]
                base = lookup[index, mode, 'baseline']
                np.testing.assert_allclose(row['initial'], base['initial'], atol=1e-8, rtol=0)
                delta = np.column_stack((row['per_vehicle_cost'], row['per_vehicle_finish'])) - np.column_stack(
                    (base['per_vehicle_cost'], base['per_vehicle_finish']))
                cluster.append(delta)
                has_adoption |= row['learned_action_steps'] > 0
                trace_path = directory / f"shard_{index % protocol['shards']}" / f'trace_{index}_{mode}_{method}.json'
                trace = json.loads(trace_path.read_text())
                trace_audit(trace, row)
                if method.startswith('bank_'):
                    for item in trace:
                        if item['bank_joint_action_admitted']:
                            assert len(item['bank_scenarios_checked']) == 17
                            assert np.all(np.asarray(item['bank_differences']) <= 1e-10)
                    if mode in ('clean', 'bias'):
                        assert np.all(delta <= 1e-9), (index, mode, method, delta)
                if method.startswith(('bank_', 'nominal_')) and row['learned_action_steps'] == 0:
                    np.testing.assert_allclose(delta, 0., atol=1e-9, rtol=0)
            differences.append(cluster)
            admitted.append(has_adoption)
        delta = np.array(differences)
        admitted = np.array(admitted)
        harm = np.any(delta > 1e-9, axis=(1, 2, 3))
        bad = np.array([any(not lookup[i, m, method]['completed'] or lookup[i, m, method]['collision'] or
                            lookup[i, m, method]['aborted'] or lookup[i, m, method]['contract_violations']
                            for m in spec['modes']) for i in range(spec['n_initials'])])
        summaries[method] = dict(
            mean_team_cost=float(np.mean([r['team_cost'] for r in data])),
            per_vehicle_cost_gain=(-delta[:, :, :, 0].mean((0, 1))).tolist(),
            per_vehicle_time_gain=(-delta[:, :, :, 1].mean((0, 1))).tolist(),
            harmful_initials=int(harm.sum()), failed_initials=int(bad.sum()),
            adopted_initials=int(admitted.sum()), harmful_adopted_initials=int(((harm | bad) & admitted).sum()),
            harm_by_mode={mode: int(np.any(delta[:, k] > 1e-9, axis=(1, 2)).sum())
                          for k, mode in enumerate(spec['modes'])},
            learned_action_steps=sum(r['learned_action_steps'] for r in data),
            control_steps=sum(r['control_steps'] for r in data),
            completed=sum(r['completed'] for r in data), runs=len(data),
            collisions=sum(r['collision'] for r in data), aborted=sum(r['aborted'] for r in data),
            contract_violations=sum(r['contract_violations'] for r in data),
            controller_p95_ms=float(np.percentile([v for r in data for v in r['controller_ms']], 95)),
            controller_max_ms=float(max(v for r in data for v in r['controller_ms'])))
    rows.sort(key=lambda r: (r['initial_index'], r['disturbance'], r['controller']))
    save(directory / 'episodes.json', rows)
    summary = dict(purpose=spec['purpose'], actual_sumo_runs=len(rows), results=summaries,
                   finite_bank_inequalities_verified=True, disturbance_risk_certified=False,
                   protocol_sha256=digest(directory / 'protocol.json'), episodes_sha256=digest(directory / 'episodes.json'))
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
    p.add_argument('--shards', type=int, default=4)
    p = sub.add_parser('worker')
    p.add_argument('directory')
    p.add_argument('shard', type=int)
    p = sub.add_parser('finalize')
    p.add_argument('directory')
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze(args.spec, args.directory, args.shards)
    elif args.command == 'worker':
        worker(args.directory, args.shard)
    else:
        finalize(args.directory)
