"""Replay every completed temporal-study harm; future noise is diagnostic only."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from coopgate.disturbance_v9 import DisturbedMergeEnv, disturbance_tape
from coopgate.disturbed_advantage_v19 import disturbed_suffix
from coopgate.fast_v8 import parameters
from coopgate.joint_planner_v15 import joint_rollout
from coopgate.merge_v4 import MergeConfig
from diagnose_calibration_harm_v19 import harmful, identity
from experiment_ego_v13 import run_episode
from probe_disturbed_gate_v19 import digest, save
from resume_frozen_risk_v19 import build_record, compare_boundary
from start_final_v19 import require_committed, verify_manifest
from study_gate_v19 import read_protocol
from study_temporal_v20 import factories, verify_episode

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'artifacts/v20/temporal_development'
PLAN = ROOT / 'artifacts/v20/temporal_harm_replay_plan'
OUTPUT = ROOT / 'artifacts/v20/temporal_harm_replay'


def select_cases(rows):
    lookup = {identity(row): row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError('Duplicate archived outcome')
    selected = set()
    for row in rows:
        if row['controller'] == 'baseline':
            continue
        base = row['initial_index'], row['disturbance'], 'baseline'
        if harmful(row, lookup[base]):
            selected.update((identity(row), base))
    return [dict(initial_index=k[0], disturbance=k[1], controller=k[2])
            for k in sorted(selected, key=lambda k: (k[0], k[1], k[2] != 'baseline', k[2]))]


def validate_bank(item, size):
    bank = np.asarray(item['selected_bank_differences'])
    if bank.shape != (size, 3, 2) or not np.isfinite(bank).all() or np.any(bank > 1e-10):
        raise ValueError('Full admitted bank evidence is invalid')
    return bank


def prepare():
    if PLAN.exists() or OUTPUT.exists():
        raise FileExistsError('Preserve the existing diagnosis')
    verify_manifest(STUDY)
    _, protocol = read_protocol(STUDY)
    rows = json.loads((STUDY / 'episodes.json').read_text())
    cases = select_cases(rows)
    sources = dict(protocol['source_sha256'])
    for name in ('tools/diagnose_temporal_harm_v20.py', 'tools/diagnose_calibration_harm_v19.py',
                 'tests/test_temporal_harm_replay_v20.py'):
        sources[name] = digest(ROOT / name)
    plan = dict(purpose='Post-hoc diagnosis of every harmful run across all frozen temporal banks and models',
        cases=cases, expected_actual_sumo_replays=len(cases),
        harmful_runs=sum(c['controller'] != 'baseline' for c in cases),
        paired_baseline_runs=sum(c['controller'] == 'baseline' for c in cases),
        source_study_actual_sumo_runs=len(rows), source_study_independent_initials=protocol['spec']['n_initials'],
        source_sha256=sources, study_manifest_sha256=digest(STUDY / 'sha256.json'),
        study_protocol_sha256=digest(STUDY / 'protocol.json'), study_episodes_sha256=digest(STUDY / 'episodes.json'),
        outcome_dependent_selection=True, additional_independent_initials=0,
        online_controller_uses_realized_future=False, risk_certified_by_diagnosis=False,
        ongoing_repair_extension_outcomes_read=False,
        hindsight_scope='Realized disturbance is used only after each actual SUMO replay to explain baseline-value differences.')
    PLAN.mkdir(parents=True)
    save(PLAN / 'plan.json', plan)
    for source in sources:
        target = PLAN / 'source' / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source, target)
    save(PLAN / 'sha256.json', {p.relative_to(PLAN).as_posix(): digest(p) for p in PLAN.rglob('*')
                              if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps({k: plan[k] for k in ('expected_actual_sumo_replays', 'harmful_runs', 'paired_baseline_runs')}), flush=True)


def hindsight(row, trace, base, method):
    if not row['completed'] or row['collision'] or row['aborted'] or row['contract_violations']:
        return dict(available=False, reason='Physical failure retained; no completed-trajectory identity is asserted')
    config = MergeConfig()
    packed = parameters(config)
    tape = disturbance_tape(row['seed'], row['disturbance'], .5, round(config.horizon / config.dt))
    states = np.asarray([row['initial']] + [item['x'] for item in trace])
    commands = np.asarray([item['command'] for item in trace])
    previous = np.concatenate((np.zeros((1, 3)), commands))
    applied = np.asarray([error for item in trace for error in item['disturbance']])
    np.testing.assert_array_equal(applied, tape[:len(applied)])
    try:
        potentials = np.asarray([disturbed_suffix(state, previous[k], k, tape, packed)[0]
                                 for k, state in enumerate(states)])
    except RuntimeError as error:
        return dict(available=False, reason='Counterfactual baseline forecast failed', error=str(error))
    active = states[:-1, :, 0] < config.goal
    stages = np.stack((active * (config.decision_dt + .05 * np.abs(commands - previous[:-1])),
                       active * config.decision_dt), axis=-1)
    differences = stages + potentials[1:] - potentials[:-1]
    original = np.column_stack((row['per_vehicle_cost'], row['per_vehicle_finish'])) - np.column_stack(
        (base['per_vehicle_cost'], base['per_vehicle_finish']))
    base_error = float(np.max(np.abs(potentials[0] - np.column_stack((base['per_vehicle_cost'], base['per_vehicle_finish'])))))
    identity_error = float(np.max(np.abs(differences.sum(0) - original)))
    if base_error > 1e-9 or identity_error > 1e-9:
        raise ValueError('Hindsight identity does not reproduce the actual paired outcome')
    steps = []
    for k, item in enumerate(trace):
        admitted = bool(item.get('portfolio_joint_action_admitted', False))
        bank = validate_bank(item, method['bank_size']) if admitted else None
        baseline_command = joint_rollout(states[k], previous[k], k, packed)[0]
        changed = np.any(commands[k] != baseline_command)  # Whole joint command must genuinely differ.
        if admitted and not changed:
            raise ValueError('An admitted joint command equals the baseline')
        unchanged = commands[k] == baseline_command
        positive = np.any(differences[k] > 1e-9, axis=1)
        steps.append(dict(step=k, admitted=admitted, composition=item.get('composition', 0),
            command=commands[k].tolist(), baseline_command=baseline_command.tolist(),
            hindsight_bellman_difference=differences[k].tolist(),
            bank_componentwise_max=bank.max(0).tolist() if admitted else None,
            positive_hindsight_residual=bool(positive.any()),
            positive_only_on_unchanged_commands=bool(positive.any() and np.all(unchanged[positive]))))
    return dict(available=True, baseline_forecast_max_error=base_error, telescoping_max_error=identity_error,
        original_regret=original.tolist(), summed_hindsight_residual=differences.sum(0).tolist(),
        positive_admission_steps=sum(s['admitted'] and s['positive_hindsight_residual'] for s in steps),
        positive_fallback_steps=sum(not s['admitted'] and s['positive_hindsight_residual'] for s in steps),
        unchanged_command_only_positive_admissions=sum(s['admitted'] and s['positive_only_on_unchanged_commands'] for s in steps),
        steps=steps)


def run():
    torch.set_num_threads(1)
    verify_manifest(PLAN)
    verify_manifest(STUDY)
    plan = json.loads((PLAN / 'plan.json').read_text())
    require_committed([PLAN / 'plan.json', PLAN / 'sha256.json', *[ROOT / s for s in plan['source_sha256']]])
    for name, expected in plan['source_sha256'].items():
        if digest(ROOT / name) != expected or digest(PLAN / 'source' / name) != expected:
            raise ValueError('Frozen diagnostic source changed: ' + name)
    if digest(STUDY / 'sha256.json') != plan['study_manifest_sha256']:
        raise ValueError('Completed development archive changed')
    _, protocol = read_protocol(STUDY)
    original = json.loads((STUDY / 'episodes.json').read_text())
    if select_cases(original) != plan['cases']:
        raise ValueError('Diagnostic selection changed')
    lookup = {identity(r): r for r in original}
    fs = factories(protocol)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    rows, diagnostics = [], []
    with DisturbedMergeEnv() as env:
        for number, case in enumerate(plan['cases']):
            reference = lookup[case['initial_index'], case['disturbance'], case['controller']]
            method = protocol['spec']['methods'][case['controller']]
            row, trace, initial = run_episode(env, reference['seed'], reference['scenario'], case['disturbance'], fs[case['controller']], True)
            row, adopted, checks, failures = build_record(row, trace, initial, case['initial_index'], case['controller'], method, protocol['spec'])
            save(OUTPUT / f'row_{number}.json', row)
            save(OUTPUT / f'trace_{number}.json', trace)
            compare_boundary(row, reference)
            verify_episode(trace, row, method, adopted, checks, failures)
            archived = STUDY / f'shard_{case["initial_index"] % protocol["spec"]["shards"]}/trace_{case["initial_index"]}_{case["disturbance"]}_{case["controller"]}.json'
            if archived.exists():
                old = json.loads(archived.read_text())
                if len(old) != len(trace):
                    raise ValueError('Archived trajectory length differs')
                for actual, saved in zip(trace, old):
                    for field in ('command', 'x', 'disturbance'):
                        np.testing.assert_array_equal(actual[field], saved[field])
            base = lookup[case['initial_index'], case['disturbance'], 'baseline']
            mechanism = hindsight(row, trace, base, method)
            save(OUTPUT / f'mechanism_{number}.json', mechanism)
            diagnostics.append(dict(case_index=number, **case, non_timing_metrics_match=True,
                archived_trajectory_compared=archived.exists(), mechanism={k: v for k, v in mechanism.items() if k != 'steps'}))
            rows.append(row)
            print(json.dumps(dict(completed_replays=len(rows), expected_replays=len(plan['cases']))), flush=True)
    harms = [d for d in diagnostics if d['controller'] != 'baseline']
    report = dict(actual_sumo_replays=len(rows), harmful_runs=len(harms), paired_baseline_runs=len(rows) - len(harms),
        all_non_timing_metrics_match=True, all_physical_traces_checked=True,
        all_hindsight_identities_match=all(d['mechanism']['available'] for d in diagnostics),
        positive_admission_steps=sum(d['mechanism'].get('positive_admission_steps', 0) for d in harms),
        positive_fallback_steps=sum(d['mechanism'].get('positive_fallback_steps', 0) for d in harms),
        unchanged_command_only_positive_admissions=sum(d['mechanism'].get('unchanged_command_only_positive_admissions', 0) for d in harms),
        by_method=dict(Counter(d['controller'] for d in harms)), cases=diagnostics,
        outcome_dependent_selection=True, additional_independent_initials=0,
        online_controller_uses_realized_future=False, risk_certified_by_diagnosis=False,
        policy_changes=0, plan_sha256=digest(PLAN / 'plan.json'), study_manifest_sha256=digest(STUDY / 'sha256.json'))
    save(OUTPUT / 'episodes.json', rows)
    save(OUTPUT / 'audit.json', report)
    save(OUTPUT / 'sha256.json', {p.relative_to(OUTPUT).as_posix(): digest(p) for p in OUTPUT.rglob('*')
                                if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps({k: v for k, v in report.items() if k != 'cases'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'run'))
    args = parser.parse_args()
    prepare() if args.command == 'prepare' else run()
