"""Reconstruct completed repair commands and every admitted bank certificate.

No performance-gate select method or candidate-expansion helper is used. This audit
reuses the frozen physical projector and forecast dynamics, not the selector.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from coopgate.advantage_gate_v18 import action_then_baseline, baseline_suffix
from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.cooperative_v15 import JointPolicyGate
from coopgate.disturbed_advantage_v19 import disturbed_action_difference
from coopgate.fast_v8 import parameters
from coopgate.joint_planner_v15 import joint_rollout
from coopgate.marl_v12 import project_joint
from coopgate.merge_v4 import MergeConfig
from coopgate.robust_v9 import robust_certificate
from coopgate.temporal_portfolio_v20 import temporal_stress_bank
from probe_disturbed_gate_v19 import digest, save
from start_final_v19 import require_committed, verify_manifest
from study_gate_v19 import read_protocol
from verify_development_v14 import trace_audit

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'artifacts/v20/repair_development'
PLAN = ROOT / 'artifacts/v20/repair_closed_loop_review_plan'
OUTPUT = ROOT / 'artifacts/v20/repair_closed_loop_review'


def verify_disjoint_ranges(protocol):
    target = protocol['spec']
    start, stop = target['seed_base'], target['seed_base'] + target['n_initials']
    records = {}
    for path in sorted((ROOT / 'artifacts').rglob('protocol.json')):
        if 'source' in path.parts or path.parent == STUDY:
            continue
        data = json.loads(path.read_text(encoding='utf-8'))
        spec = data.get('spec', data)
        old_start, size = spec.get('seed_base'), spec.get('n_initials', spec.get('initials'))
        if not isinstance(old_start, int) or not isinstance(size, int):
            continue
        if max(start, old_start) < min(stop, old_start + size):
            raise ValueError('Overlapping initial range: ' + str(path))
        records[path.relative_to(ROOT).as_posix()] = dict(seed_base=old_start, initials=size, sha256=digest(path))
    return records


def prepare():
    if PLAN.exists() or OUTPUT.exists() or (STUDY / 'summary.json').exists():
        raise FileExistsError('Freeze this reviewer before completed outcomes; never overwrite it')
    _, protocol = read_protocol(STUDY)
    sources = [Path(__file__).resolve(), ROOT / 'tests/test_repair_closed_loop_review_v20.py']
    plan = dict(study_protocol_sha256=digest(STUDY / 'protocol.json'),
                source_sha256={p.relative_to(ROOT).as_posix(): digest(p) for p in sources},
                disjoint_prior_protocols=verify_disjoint_ranges(protocol),
                expected_actual_sumo_runs=protocol['spec']['expected_actual_sumo_runs'],
                scope='All physical traces, all candidate-family reconstructions, every distinct admitted 33-path forecast, and independently counted outcomes. No new independent samples.',
                policy_selection_or_threshold_changed=False, final_confirmation_outcomes_read=False)
    PLAN.mkdir(parents=True)
    save(PLAN / 'plan.json', plan)
    for source in sources:
        target = PLAN / 'source' / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    print(json.dumps(plan), flush=True)


def reconstruct_candidates(state, previous, raw, baseline, packed, repair):
    originals, seen = [], [baseline]
    for code in range(1, 8):
        proposed = baseline.copy()
        for i in range(3):
            if code & (1 << i):
                proposed[i] = raw[i]
        command, _, valid = project_joint(state, previous, proposed, packed, .5)
        if not valid or not robust_certificate(state, previous, command, packed, .5):
            raise ValueError('Uncertified original candidate')
        if any(np.array_equal(command, old) for old in seen):
            continue
        seen.append(command.copy())
        originals.append((code, 1., command.copy()))
    result = list(originals)
    if repair:
        for code, _, original in originals:
            for weight in (.75, .5, .25):
                proposed = baseline + weight * (original - baseline)
                command, _, valid = project_joint(state, previous, proposed, packed, .5)
                if not valid or not robust_certificate(state, previous, command, packed, .5):
                    raise ValueError('Uncertified additional candidate')
                if any(np.array_equal(command, old) for old in seen):
                    continue
                seen.append(command.copy())
                result.append((code, weight, command.copy()))
    return result, len(originals)


def check_selected(item, baseline, candidates, original_count):
    used = bool(item['portfolio_joint_action_admitted'])
    if item['candidate_count'] != len(candidates) or item['original_candidate_count'] != original_count:
        raise ValueError('Candidate-family coverage differs')
    if item['forecast_errors'] or item['generation_errors']:
        raise ValueError('Forecast/projection exceptions require separate investigation; do not silently omit')
    if not used:
        np.testing.assert_array_equal(item['command'], baseline)
        if item['repair_weight'] is not None or item['composition'] != 0:
            raise ValueError('Fallback has a nonbaseline candidate identity')
        return
    matches = [(code, weight, command) for code, weight, command in candidates
               if code == item['composition'] and weight == item['repair_weight']]
    if len(matches) != 1:
        raise ValueError('Selected candidate identity does not exist')
    np.testing.assert_array_equal(item['command'], matches[0][2])
    if np.array_equal(item['command'], baseline):
        raise ValueError('Baseline execution counted as learning')


def check_forecast_record(differences, recorded):
    values = np.asarray(differences)
    if values.shape != (33, 3, 2) or not np.isfinite(values).all():
        raise ValueError('Incomplete admitted 33-path certificate')
    np.testing.assert_allclose(values, recorded, atol=1e-10, rtol=0)
    if values.max() > 1e-10 or values[0, :, 0].mean() >= -1e-9:
        raise ValueError('Admitted action violates a performance condition')


def outcome_counts(rows, spec, name):
    lookup = {(r['initial_index'], r['disturbance'], r['controller']): r for r in rows}
    actual_adopted = actual_harm = bundle_adopted = bundle_harm = failed_runs = 0
    gains, bases = [], []
    for i in range(spec['n_initials']):
        mode_index = int(np.random.default_rng(np.random.SeedSequence(
            [spec['seed_base'] + i, spec['mode_selection_stream']])).integers(len(spec['modes'])))
        indicators = []
        for m, mode in enumerate(spec['modes']):
            row, baseline = lookup[i, mode, name], lookup[i, mode, 'baseline']
            used = row['learned_action_steps'] > 0
            failed = bool(not row['completed'] or row['collision'] or row['aborted'] or row['contract_violations'])
            harmful = bool(failed or np.any(np.asarray(row['per_vehicle_cost']) - baseline['per_vehicle_cost'] > 1e-9)
                           or np.any(np.asarray(row['per_vehicle_finish']) - baseline['per_vehicle_finish'] > 1e-9))
            indicators.append((used, harmful))
            failed_runs += int(failed)
            if m == mode_index:
                actual_adopted += int(used)
                actual_harm += int(used and harmful)
                gains.append(float(np.mean(np.asarray(baseline['per_vehicle_cost']) - row['per_vehicle_cost'])))
                bases.append(float(np.mean(baseline['per_vehicle_cost'])))
        a, h = any(r[0] for r in indicators), any(r[1] for r in indicators)
        bundle_adopted += int(a)
        bundle_harm += int(a and h)
    return dict(actual_adopted=actual_adopted, actual_harmful_adopted=actual_harm,
                bundle_adopted=bundle_adopted, bundle_harmful_adopted=bundle_harm,
                failed_runs=failed_runs, actual_mean_gain=float(np.mean(gains)),
                actual_mean_gain_percent=float(100 * np.mean(gains) / np.mean(bases)))


def run():
    if OUTPUT.exists():
        raise FileExistsError('Preserve the completed review')
    plan = json.loads((PLAN / 'plan.json').read_text())
    for source, expected in plan['source_sha256'].items():
        if digest(ROOT / source) != expected or digest(PLAN / 'source' / source) != expected:
            raise ValueError('Frozen review source changed')
    require_committed([PLAN / 'plan.json', *[ROOT / p for p in plan['source_sha256']]])
    _, protocol = read_protocol(STUDY)
    if digest(STUDY / 'protocol.json') != plan['study_protocol_sha256']:
        raise ValueError('Frozen study identity changed')
    for path, record in plan['disjoint_prior_protocols'].items():
        if digest(ROOT / path) != record['sha256']:
            raise ValueError('A prior population definition changed: ' + path)
    verify_manifest(STUDY)  # Written only after both native workers and analysis complete.
    rows = json.loads((STUDY / 'episodes.json').read_text())
    summary = json.loads((STUDY / 'summary.json').read_text())
    spec = protocol['spec']
    expected = {(i, mode, name) for i in range(spec['n_initials']) for mode in spec['modes'] for name in spec['methods']}
    if len(rows) != plan['expected_actual_sumo_runs'] or {(r['initial_index'], r['disturbance'], r['controller']) for r in rows} != expected:
        raise ValueError('Missing, duplicate or unexpected closed-loop records')
    models = {}
    for seed, record in protocol['models'].items():
        if digest(ROOT / record['path']) != record['sha256']:
            raise ValueError('Frozen training model changed')
        models[seed] = load_coupled(ROOT / record['path'])[0]
    torch.set_num_threads(1)
    packed, counts, forecast_cache = parameters(MergeConfig()), Counter(), {}
    names, tapes = temporal_stress_bank('sampled')
    if len(names) != 33:
        raise ValueError('Wrong forecast bank')
    started = time.perf_counter()
    for number, row in enumerate(rows):
        name = row['controller']
        method = spec['methods'][name]
        path = STUDY / f"shard_{row['initial_index'] % spec['shards']}" / f"trace_{row['initial_index']}_{row['disturbance']}_{name}.json"
        trace = json.loads(path.read_text())
        trace_audit(trace, row)
        if sum(bool(t.get('portfolio_joint_action_admitted', False)) for t in trace) != row['learned_action_steps']:
            raise ValueError('Learning adoption count differs from actual trace')
        counts['physical_traces'] += 1
        state, previous = np.asarray(row['initial']), np.zeros(3)
        actor = JointPolicyGate(models[str(method['seed'])]) if method['kind'] != 'baseline' else None
        for step, item in enumerate(trace):
            command = np.asarray(item['command'])
            if method['kind'] in ('reference', 'repair'):
                baseline = joint_rollout(state, previous, step, packed)[0]
                actor.step = step
                _, actor_info = actor.select(state, previous)
                np.testing.assert_array_equal(item['baseline_command'], baseline)
                np.testing.assert_array_equal(item['actor_proposal'], actor_info['proposal'])
                candidates, original_count = reconstruct_candidates(state, previous, np.asarray(actor_info['proposal']), baseline,
                                                                      packed, method['kind'] == 'repair')
                check_selected(item, baseline, candidates, original_count)
                counts['candidate_families'] += 1
                counts['candidate_commands'] += len(candidates)
            if item.get('portfolio_joint_action_admitted', False):
                key = (step, state.tobytes(), previous.tobytes(), command.tobytes())
                if key not in forecast_cache:
                    base = baseline_suffix(state, previous, step, packed)[0]
                    candidate = action_then_baseline(state, previous, step, command, packed)[0]
                    differences = [candidate - base]
                    for tape in tapes[1:]:
                        differences.append(disturbed_action_difference(state, previous, step, command, tape, packed)[0])
                    forecast_cache[key] = np.asarray(differences)
                    counts['distinct_admitted_forecasts_recomputed'] += 1
                check_forecast_record(forecast_cache[key], item['selected_bank_differences'])
                counts['admitted_decisions_verified'] += 1
                counts['attenuated_decisions_verified'] += int(bool(item.get('repair_action_admitted', False)))
            state, previous = np.asarray(item['x']), command
        if number % 40 == 0:
            print(json.dumps(dict(verified_runs=number + 1, elapsed_s=time.perf_counter() - started)), flush=True)
    reconstructed = {}
    for name in spec['methods']:
        counted = outcome_counts(rows, spec, name)
        for key, value in counted.items():
            np.testing.assert_allclose(value, summary['results'][name][key], atol=1e-12, rtol=1e-12, err_msg=name + '/' + key)
        reconstructed[name] = counted
    report = dict(counts=counts, independent_outcome_counts=reconstructed,
                  study_manifest_sha256=digest(STUDY / 'sha256.json'), plan_sha256=digest(PLAN / 'plan.json'),
                  all_reconstructed_commands_and_admitted_forecasts_match=True,
                  additional_sumo_runs=0, additional_independent_samples=0, risk_certified=False,
                  final_confirmation_outcomes_read=False, scope=plan['scope'])
    OUTPUT.mkdir()
    save(OUTPUT / 'audit.json', report)
    save(OUTPUT / 'sha256.json', {p.relative_to(OUTPUT).as_posix(): digest(p) for p in OUTPUT.rglob('*')
                                 if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'run'))
    (prepare if parser.parse_args().command == 'prepare' else run)()
