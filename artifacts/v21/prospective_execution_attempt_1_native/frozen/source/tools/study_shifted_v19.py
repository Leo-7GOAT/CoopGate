"""Frozen direct-transfer SUMO study: initial distribution x physical plant.

Weights, gates and nominal physical assumptions are held fixed. All failures,
including baseline failures, are retained. This study does not recalibrate a
gate or extend an IID risk certificate to a shifted distribution.
"""
import argparse
from contextlib import ExitStack
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from coopgate.fast_v8 import parameters
from coopgate.merge_v4 import MergeConfig
from coopgate.robust_v9 import robust_certificate
from coopgate.shifted_sumo_v19 import PLANTS, SCENARIOS, ShiftedMergeEnv, audit_shifted_trace, shifted_initial
from experiment_ego_v13 import run_episode
from probe_disturbed_gate_v19 import digest, save
from study_gate_v19 import ROOT, factories, read_protocol


def cases(spec):
    """Independent initial IDs within scenario, paired across all plant types."""
    if spec['n_initials_per_scenario'] > 1000:
        raise ValueError('Scenario seed blocks would overlap')
    for si, scenario in enumerate(spec['scenarios']):
        for i in range(spec['n_initials_per_scenario']):
            yield si * spec['n_initials_per_scenario'] + i, scenario, i, spec['seed_base'] + si * 1000 + i


def freeze(spec_path, directory):
    spec = json.loads(Path(spec_path).read_text(encoding='utf-8'))
    assert spec['data_role'] == 'direct_transfer' and spec['mode'] == 'uniform'
    assert set(spec['scenarios']) <= set(SCENARIOS) and set(spec['plants']) <= set(PLANTS)
    assert spec['shards'] >= 1 and spec['n_initials_per_scenario'] >= spec['shards']
    assert spec['methods']['baseline'] == dict(kind='baseline')
    expected = len(list(cases(spec))) * len(spec['plants']) * len(spec['methods'])
    assert spec['expected_actual_sumo_runs'] == expected
    model_records = json.loads((ROOT / 'artifacts/v18/independent_study/protocol.json').read_text())['models']
    seeds = sorted({str(m['seed']) for m in spec['methods'].values() if 'seed' in m})
    protocol = dict(spec=spec, spec_sha256=digest(spec_path), models={seed: model_records[seed] for seed in seeds})
    factories(protocol)
    packed = parameters(MergeConfig())
    # Validate the whole frozen shifted population, with no seed replacement.
    for _, scenario, _, seed in cases(spec):
        if scenario not in ('nominal', 'dense'):
            x = shifted_initial(seed, scenario)
            if not robust_certificate(x, np.zeros(3), np.full(3, -2.5), packed, .5):
                raise ValueError(f'Frozen initial outside original backup set: {scenario}/{seed}')
    sources = list((ROOT / 'src/coopgate').glob('*.py')) + [Path(__file__),
        ROOT / 'tools/study_gate_v19.py', ROOT / 'tools/experiment_ego_v13.py',
        ROOT / 'tools/verify_development_v14.py', ROOT / 'tools/probe_disturbed_gate_v19.py']
    protocol['source_sha256'] = {p.resolve().relative_to(ROOT).as_posix(): digest(p) for p in sources}
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'protocol.json', protocol)
    for source in protocol['source_sha256']:
        destination = out / 'source' / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source, destination)
    print(json.dumps(dict(protocol_sha256=digest(out / 'protocol.json'), actual_runs_planned=expected)), flush=True)


def worker(directory, shard):
    torch.set_num_threads(1)
    directory, protocol = read_protocol(directory)
    spec = protocol['spec']
    assert 0 <= shard < spec['shards']
    fs = factories(protocol)
    out = directory / f'shard_{shard}'
    out.mkdir(exist_ok=False)
    assigned = [c for c in cases(spec) if c[0] % spec['shards'] == shard]
    warm = np.array([[-80., 12.], [-85., 13.], [-120., 11.]])
    for factory in fs.values():
        factory(warm).select(warm, np.zeros(3))
    started = time.perf_counter()
    rows = []
    with ExitStack() as stack:
        environments = {p: stack.enter_context(ShiftedMergeEnv(plant=p)) for p in spec['plants']}
        for global_index, scenario, index, seed in assigned:
            for pi, plant in enumerate(spec['plants']):
                env = environments[plant]
                for method in np.random.default_rng(np.random.SeedSequence([seed, pi, 19337])).permutation(list(fs)):
                    row, trace, initial = run_episode(env, seed, scenario, spec['mode'], fs[method], True)
                    kind = spec['methods'][method]['kind']
                    cost = np.array(row['per_vehicle_finish']) + .05 * np.array(row['per_vehicle_variation'])
                    used = sum(t.get('nominal_joint_action_admitted', kind == 'raw') for t in trace)
                    row.update(controller=str(method), initial=initial.tolist(), initial_index=index,
                               global_initial_index=global_index, plant_name=plant,
                               per_vehicle_cost=cost.tolist(), team_cost=float(cost.mean()),
                               learned_action_steps=int(used), control_steps=len(trace))
                    failure = not row['completed'] or row['collision'] or row['aborted'] or row['contract_violations']
                    row['trace_saved'] = bool(index in spec['audit_initial_indices'] or failure)
                    if row['trace_saved']:
                        save(out / f'trace_{scenario}_{index}_{plant}_{method}.json', trace)
                    rows.append(row)
                    save(out / 'episodes.json', rows)
                    audit_shifted_trace(trace, row)
                print(json.dumps(dict(shard=shard, scenario=scenario, index=index, plant=plant,
                                      actual_sumo_runs=len(rows), elapsed_s=time.perf_counter() - started)), flush=True)
    assert len(rows) == len(assigned) * len(spec['plants']) * len(fs)
    save(out / 'complete.json', dict(actual_sumo_runs=len(rows),
         global_initial_indices=[c[0] for c in assigned], episodes_sha256=digest(out / 'episodes.json'),
         protocol_sha256=digest(directory / 'protocol.json')))


def finalize(directory):
    directory, protocol = read_protocol(directory)
    spec = protocol['spec']
    rows = []
    for shard in range(spec['shards']):
        out = directory / f'shard_{shard}'
        done = json.loads((out / 'complete.json').read_text())
        assert done['global_initial_indices'] == [c[0] for c in cases(spec) if c[0] % spec['shards'] == shard]
        assert done['protocol_sha256'] == digest(directory / 'protocol.json')
        assert done['episodes_sha256'] == digest(out / 'episodes.json')
        part = json.loads((out / 'episodes.json').read_text())
        assert len(part) == done['actual_sumo_runs']
        rows.extend(part)
    lookup = {(r['scenario'], r['initial_index'], r['plant_name'], r['controller']): r for r in rows}
    expected = {(s, i, p, m) for _, s, i, _ in cases(spec) for p in spec['plants'] for m in spec['methods']}
    assert set(lookup) == expected and len(lookup) == len(rows) == spec['expected_actual_sumo_runs']
    output_cells = []
    for scenario in spec['scenarios']:
        for plant in spec['plants']:
            for method in spec['methods']:
                data, differences, harmful, used = [], [], [], []
                for i in range(spec['n_initials_per_scenario']):
                    row, base = lookup[scenario, i, plant, method], lookup[scenario, i, plant, 'baseline']
                    np.testing.assert_array_equal(row['initial'], base['initial'])
                    assert row['plant'] == base['plant']
                    delta = np.column_stack((row['per_vehicle_cost'], row['per_vehicle_finish'])) - np.column_stack(
                        (base['per_vehicle_cost'], base['per_vehicle_finish']))
                    failure = not row['completed'] or row['collision'] or row['aborted'] or row['contract_violations']
                    harmful.append(bool(failure or np.any(delta > 1e-9)))
                    used.append(row['learned_action_steps'] > 0)
                    differences.append(delta)
                    data.append(row)
                    if row['trace_saved']:
                        trace_path = directory / f"shard_{row['global_initial_index'] % spec['shards']}" / f'trace_{scenario}_{i}_{plant}_{method}.json'
                        audit_shifted_trace(json.loads(trace_path.read_text()), row)
                    if spec['methods'][method]['kind'] in ('nominal', 'portfolio') and row['learned_action_steps'] == 0:
                        np.testing.assert_allclose(delta, 0., atol=1e-9, rtol=0)
                delta = np.array(differences)
                times = [v for row in data for v in row['controller_ms']]
                base_cost = np.mean([lookup[scenario, i, plant, 'baseline']['team_cost'] for i in range(len(data))])
                output_cells.append(dict(
                    scenario=scenario, plant=plant, method=method, independent_initials=len(data),
                    runs=len(data), completed=sum(r['completed'] for r in data),
                    aborted=sum(r['aborted'] for r in data), collisions=sum(r['collision'] for r in data),
                    contract_violations=sum(r['contract_violations'] for r in data),
                    adopted=int(sum(used)), harmful=int(sum(harmful)),
                    harmful_adopted=int(sum(a and h for a, h in zip(used, harmful))),
                    mean_team_gain_seconds=float(-delta[:, :, 0].mean()),
                    mean_team_gain_percent=float(-100 * delta[:, :, 0].mean() / base_cost),
                    per_vehicle_cost_gain=(-delta[:, :, 0].mean(0)).tolist(),
                    per_vehicle_time_gain=(-delta[:, :, 1].mean(0)).tolist(),
                    maximum_actual_error=max(r['max_actual_error'] for r in data),
                    error_contract_exceeded_runs=sum(r['error_contract_exceeded_substeps'] > 0 for r in data),
                    error_contract_exceeded_substeps=sum(r['error_contract_exceeded_substeps'] for r in data),
                    controller_p95_ms=float(np.percentile(times, 95)) if times else None,
                    controller_p99_ms=float(np.percentile(times, 99)) if times else None,
                    controller_max_ms=float(max(times)) if times else None,
                    steps_over_500ms=sum(t > 500 for t in times), total_control_steps=len(times)))
    rows.sort(key=lambda r: (r['scenario'], r['initial_index'], r['plant_name'], r['controller']))
    save(directory / 'episodes.json', rows)
    result = dict(purpose=spec['purpose'], actual_sumo_runs=len(rows), cells=output_cells,
                  risk_certified=False, weights_or_gates_recalibrated=False,
                  failure_accounting='Incomplete arrival capped at the original 25 s horizon; observed variation until own arrival retained. All physical failures count as harm, even if the baseline also fails.',
                  scope='Direct transfer to five declared initial distributions and four plants, uniform innovations. Per-cell descriptive results; no shifted-distribution or arbitrary-mismatch risk guarantee.',
                  episodes_sha256=digest(directory / 'episodes.json'), protocol_sha256=digest(directory / 'protocol.json'))
    save(directory / 'summary.json', result)
    save(directory / 'sha256.json', {p.relative_to(directory).as_posix(): digest(p)
         for p in sorted(directory.rglob('*')) if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps(dict(actual_sumo_runs=len(rows), cells=len(output_cells),
                          failures=sum(not r['completed'] for r in rows), risk_certified=False)), flush=True)


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
