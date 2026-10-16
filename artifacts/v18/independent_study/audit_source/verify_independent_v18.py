"""Reconstruct fixed models, physical traces and all nominal gate identities."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from coopgate.advantage_gate_v18 import JointAdvantageGate
from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.disturbance_v9 import DisturbedMergeEnv
from coopgate.fast_v8 import advance_one, parameters
from coopgate.merge_v4 import MergeConfig
from experiment_ego_v13 import run_episode
from study_independent_v18 import ROOT, factories, verify_sources
from verify_cooperative_v16 import audit_training
from verify_development_v14 import trace_audit
from verify_single_v12 import equivalent


def load(study):
    study = Path(study); torch.set_num_threads(1)
    p = json.loads((study / 'protocol.json').read_text())
    done = json.loads((study / 'completion.json').read_text())
    verify_sources(study, p)
    digest = hashlib.sha256((study / 'episodes.json').read_bytes()).hexdigest()
    assert done['episodes_sha256'] == digest
    rows = json.loads((study / 'episodes.json').read_text())
    lookup = {(r['seed'], r['disturbance'], r['controller']): r for r in rows}
    expected = {(p['seed_base'] + i, m, k) for i in range(p['n_initials']) for m in p['modes'] for k in p['methods']}
    assert len(rows) == len(lookup) == p['expected_sumo_runs'] and set(lookup) == expected
    for row in rows:
        np.testing.assert_allclose(row['per_vehicle_cost'], np.array(row['per_vehicle_finish']) +
                                   .05 * np.array(row['per_vehicle_variation']), atol=1e-10, rtol=0)
    return p, rows, lookup, digest


def trace_path(study, p, i, mode, method):
    return Path(study) / f'shard_{i % p["shards"]}' / f'trace_{i}_{mode}_{method}.json'


def nominal(study, shard, out):
    p, rows, lookup, digest = load(study)
    assert 0 <= shard < p['shards']
    fs = factories(p['models']); packed = parameters(MergeConfig())
    indices = list(range(shard, p['n_initials'], p['shards']))
    identities = actions = 0; largest_residual = 0.
    records = []
    for i in indices:
        seed = p['seed_base'] + i
        base = lookup[seed, 'clean', 'baseline']
        for model_id in p['models']:
            name = 'gate_' + model_id; row = lookup[seed, 'clean', name]
            controller = fs[name](None); state = np.array(row['initial']); prev = np.zeros(3)
            values = np.zeros((3, 2)); total = np.zeros((3, 2)); adopted = 0
            for step in range(50):
                active = state[:, 0] < 60
                if not active.any(): break
                command, info = controller.select(state, prev)
                values[:, 0] += active * (.5 + .05 * np.abs(command - prev))
                values[:, 1] += active * .5
                if info['nominal_joint_action_admitted']:
                    difference = np.array(info['nominal_difference'])
                    assert np.all(difference <= 1e-10) and difference[:, 0].mean() < -1e-9
                    total += difference; adopted += 1
                assert advance_one(state, command, packed)
                prev = command; actions += 1
            actual = np.column_stack((row['per_vehicle_cost'], row['per_vehicle_finish']))
            baseline = np.column_stack((base['per_vehicle_cost'], base['per_vehicle_finish']))
            np.testing.assert_allclose(values, actual, atol=1e-9, rtol=0)
            np.testing.assert_allclose(actual - baseline, total, atol=1e-9, rtol=0)
            assert np.all(actual - baseline <= 1e-9) and adopted == row['learned_action_steps']
            assert bool(np.all(state[:, 0] >= 60)) == row['completed']
            largest_residual = max(largest_residual, float(np.max(np.abs(actual - baseline - total))))
            identities += 1; records.append([i, model_id])
        if len(records) % 48 == 0:
            print(json.dumps(dict(shard=shard, nominal_identities=identities)), flush=True)
    result = dict(shard=shard, initial_indices=indices, identities=identities, records=records,
                  reconstructed_model_controls=actions, largest_identity_residual=largest_residual,
                  episodes_sha256=digest, additional_sumo_runs=0)
    Path(out).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('records', 'initial_indices')}), flush=True)


def physical(study, replay, out):
    p, rows, lookup, digest = load(study); fs = factories(p['models'])
    traces = actions = 0
    for row in rows:
        if not row['trace_saved']: continue
        i, mode, name = row['initial_index'], row['disturbance'], row['controller']
        trace = json.loads(trace_path(study, p, i, mode, name).read_text())
        trace_audit(trace, row); traces += 1
        assert row['learned_action_steps'] == sum(t.get('nominal_joint_action_admitted', False) for t in trace)
        controller = fs[name](None); state = np.array(row['initial']); prev = np.zeros(3)
        for item in trace:
            command, info = controller.select(state, prev)
            np.testing.assert_allclose(command, item['command'], atol=1e-9, rtol=0)
            if name.startswith('gate_'):
                assert info['nominal_joint_action_admitted'] == item['nominal_joint_action_admitted']
                np.testing.assert_allclose(info['nominal_difference'], item['nominal_difference'], atol=1e-9, rtol=0)
            state = np.array(item['x']); prev = np.array(item['command']); actions += 1
    training = {}
    for model_id, record in p['models'].items():
        path = ROOT / record['path']; _, meta = load_coupled(path)
        training[model_id] = dict(stageB=audit_training(path), stageA=audit_training(ROOT / meta['warmstart']))
    replays = 0
    if replay:
        with DisturbedMergeEnv() as env:
            for i in (0, p['n_initials'] // 2):
                seed = p['seed_base'] + i; regime = 'nominal' if i == 0 else 'dense'
                for mode in p['modes']:
                    for name, build in fs.items():
                        result, trace, _ = run_episode(env, seed, regime, mode, build, True)
                        for key in ('per_vehicle_finish', 'per_vehicle_variation', 'completed', 'collision',
                                    'aborted', 'contract_violations'):
                            equivalent(result[key], lookup[seed, mode, name][key])
                        saved = json.loads(trace_path(study, p, i, mode, name).read_text())
                        np.testing.assert_allclose([t['command'] for t in trace], [t['command'] for t in saved], atol=1e-9, rtol=0)
                        replays += 1
    result = dict(physical_traces=traces, reconstructed_controls=actions, training=training,
                  additional_sumo_replays=replays, episodes_sha256=digest)
    Path(out).write_text(json.dumps(result, indent=2) + '\n'); print(json.dumps(result), flush=True)


def assemble(study, physical_path, nominal_paths, out):
    p, rows, lookup, digest = load(study)
    phys = json.loads(Path(physical_path).read_text()); assert phys['episodes_sha256'] == digest
    parts = [json.loads(Path(path).read_text()) for path in nominal_paths]
    assert len(parts) == p['shards'] and {r['shard'] for r in parts} == set(range(p['shards']))
    records = []
    for r in parts:
        assert r['episodes_sha256'] == digest
        assert r['initial_indices'] == list(range(r['shard'], p['n_initials'], p['shards']))
        assert r['identities'] == len(r['records'])
        records.extend(tuple(a) for a in r['records'])
    assert len(records) == len(set(records)) == p['n_initials'] * len(p['models'])
    assert set(records) == {(i, s) for i in range(p['n_initials']) for s in p['models']}
    result = dict(actual_sumo_runs=len(rows), episodes_sha256=digest, physical_audit=phys,
        nominal_reconstruction=dict(identities=len(records), reconstructed_model_controls=sum(r['reconstructed_model_controls'] for r in parts),
            largest_identity_residual=max(r['largest_identity_residual'] for r in parts), additional_sumo_runs=0),
        disturbance_performance_risk_certified=False,
        verifier_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    Path(out).write_text(json.dumps(result, indent=2) + '\n'); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest='command', required=True)
    n = sub.add_parser('nominal'); n.add_argument('study'); n.add_argument('--shard', type=int, required=True); n.add_argument('--out', required=True)
    t = sub.add_parser('physical'); t.add_argument('study'); t.add_argument('--replay', action='store_true'); t.add_argument('--out', required=True)
    a = sub.add_parser('assemble'); a.add_argument('study'); a.add_argument('--physical', required=True); a.add_argument('--nominal', nargs='+', required=True); a.add_argument('--out', required=True)
    a = parser.parse_args()
    if a.command == 'nominal': nominal(a.study, a.shard, a.out)
    elif a.command == 'physical': physical(a.study, a.replay, a.out)
    else: assemble(a.study, a.physical, a.nominal, a.out)
