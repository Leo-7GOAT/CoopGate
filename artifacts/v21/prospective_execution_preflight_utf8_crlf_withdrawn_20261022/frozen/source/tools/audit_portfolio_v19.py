"""Nonvacuous finite-bank identity audit and exhaustive strict-gate harm replay.

Cases are selected after the development study for mechanism verification, not
additional independent performance testing. They are counted separately.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from coopgate.disturbance_v9 import DisturbedMergeEnv, disturbance_tape
from coopgate.disturbed_advantage_v19 import disturbed_action_difference, disturbed_suffix
from coopgate.fast_v8 import parameters
from coopgate.joint_planner_v15 import joint_rollout
from coopgate.merge_v4 import MergeConfig
from coopgate.stress_prediction_v17 import fixed_stress_bank
from experiment_ego_v13 import run_episode
from probe_disturbed_gate_v19 import digest, save
from study_gate_v19 import factories, read_protocol
from verify_development_v14 import trace_audit


def values(row):
    return np.column_stack((row['per_vehicle_cost'], row['per_vehicle_finish']))


def audit(study, out):
    torch.set_num_threads(1)
    study, protocol = read_protocol(study)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    reference = json.loads((study / 'episodes.json').read_text())
    lookup = {(r['initial_index'], r['disturbance'], r['controller']): r for r in reference}
    fs = factories(protocol)
    strict = [name for name, s in protocol['spec']['methods'].items()
              if s['kind'] == 'portfolio' and s['max_failed_scenarios'] == 0]
    selected = []
    for name in strict:
        for mode in ('clean', 'bias'):
            candidates = [r for r in reference if r['controller'] == name and r['disturbance'] == mode
                          and r['learned_action_steps'] > 0]
            if not candidates:
                raise ValueError('No nonvacuous audit example for ' + name + '/' + mode)
            row = min(candidates, key=lambda r: r['initial_index'])
            selected.append((row['initial_index'], mode, name))
    harmful = []
    for row in reference:
        if row['controller'] not in strict:
            continue
        base = lookup[row['initial_index'], row['disturbance'], 'baseline']
        if np.any(values(row) - values(base) > 1e-9) or not row['completed'] or row['aborted'] or row['collision'] or row['contract_violations']:
            harmful.append((row['initial_index'], row['disturbance'], row['controller']))
    plan = dict(purpose=__doc__, selected_nonvacuous_cases=selected, all_strict_harm_cases=harmful,
                study_protocol_sha256=digest(study / 'protocol.json'), study_episodes_sha256=digest(study / 'episodes.json'),
                audit_source_sha256=digest(__file__))
    save(out / 'plan.json', plan)
    shutil.copy2(Path(__file__), out / Path(__file__).name)
    rows, identities, replayed = [], [], {}
    packed = parameters(MergeConfig())
    names, bank = fixed_stress_bank()
    with DisturbedMergeEnv() as env:
        def replay(index, mode, method):
            key = (index, mode, method)
            if key in replayed:
                return replayed[key]
            expected = lookup[key]
            row, trace, initial = run_episode(env, expected['seed'], expected['scenario'], mode, fs[method], True)
            cost = np.array(row['per_vehicle_finish']) + .05 * np.array(row['per_vehicle_variation'])
            row.update(initial=initial.tolist(), initial_index=index, controller=method, per_vehicle_cost=cost.tolist())
            save(out / f'trace_{index}_{mode}_{method}.json', trace)
            rows.append(row)
            save(out / 'episodes.json', rows)
            trace_audit(trace, row)
            for metric in ('initial', 'per_vehicle_cost', 'per_vehicle_finish', 'per_vehicle_variation',
                           'completed', 'collision', 'contract_violations', 'aborted'):
                np.testing.assert_allclose(row[metric], expected[metric], atol=1e-8, rtol=0)
            replayed[key] = row, trace
            return row, trace

        for index, mode, method in selected:
            base, _ = replay(index, mode, 'baseline')
            row, trace = replay(index, mode, method)
            actual_tape = disturbance_tape(row['seed'], mode, .5, 250)
            assert any(np.array_equal(actual_tape, tape) for tape in bank)
            state, previous = np.array(row['initial']), np.zeros(3)
            potential, _, _ = disturbed_suffix(state, previous, 0, actual_tape, packed)
            np.testing.assert_allclose(potential, values(base), atol=1e-8, rtol=0)
            total = np.zeros((3, 2))
            admissions = 0
            for step, item in enumerate(trace):
                command = np.array(item['command'])
                if item['portfolio_joint_action_admitted']:
                    assert len(item['selected_bank_differences']) == len(bank)
                    for k, tape in enumerate(bank):
                        difference, _, _ = disturbed_action_difference(state, previous, step, command, tape, packed)
                        np.testing.assert_allclose(difference, item['selected_bank_differences'][k], atol=1e-8, rtol=0)
                        assert np.all(difference <= 1e-9)
                    actual_difference, _, _ = disturbed_action_difference(state, previous, step, command, actual_tape, packed)
                    total += actual_difference
                    admissions += 1
                else:
                    baseline_command, _, _, _ = joint_rollout(state, previous, step, packed)
                    np.testing.assert_allclose(command, baseline_command, atol=1e-9, rtol=0)
                state, previous = np.array(item['x']), command
            assert admissions > 0
            observed = values(row) - values(base)
            np.testing.assert_allclose(total, observed, atol=1e-8, rtol=0)
            assert np.all(observed <= 1e-9)
            identities.append(dict(initial_index=index, mode=mode, method=method, admissions=admissions,
                                   summed_differences=total.tolist(), observed_differences=observed.tolist(),
                                   maximum_identity_residual=float(np.max(np.abs(total - observed)))))
            print(json.dumps(dict(identity_cases=len(identities), actual_sumo_replays=len(rows))), flush=True)
        for index, mode, method in harmful:
            replay(index, mode, 'baseline')
            replay(index, mode, 'nominal_' + method.rsplit('_', 1)[1])
            replay(index, mode, method)
    result = dict(actual_sumo_replays=len(rows), nonvacuous_identity_cases=identities,
                  all_strict_harm_cases_reproduced=len(harmful), risk_certified=False,
                  episodes_sha256=digest(out / 'episodes.json'), plan_sha256=digest(out / 'plan.json'))
    save(out / 'audit.json', result)
    save(out / 'sha256.json', {p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('study')
    parser.add_argument('out')
    args = parser.parse_args()
    audit(args.study, args.out)
