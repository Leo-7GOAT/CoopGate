"""Paired SUMO collection and independent physical/reference-policy replay.

No CLI, automatic launch, implicit resume or partial-effect summary. The future
formal supervisor must enforce predecessor completion and freeze bindings.
"""
from pathlib import Path
import time
import numpy as np

from coopgate.disturbance_v9 import disturbance_tape
from coopgate.episode_store_v21 import EpisodeStore
from coopgate.joint_risk_data_v42 import METHODS, episode_outcome
from coopgate.matched_shift_v21 import MatchedShiftedMergeEnv, matched_plant_parameters, tape_fingerprint
from coopgate.memoized_recovery_v37 import MemoizedRecoveryGate
from coopgate.merge_v4 import MergeConfig, transition_safe
from coopgate.recovery_joint_gate_v35 import RecoveryJointGate
from audit_buffered_scope_v25 import TimedGate, check_calls
from run_fair_runtime_v41 import json_ready
import design_joint_risk_v42 as design


def schedule(case):
    if len(case['methods']) != 4 or set(case['methods']) != set(METHODS):
        raise ValueError('The complete fixed four-method family is required')
    return [[case['cluster_id'], method] for method in case['methods']]


def execute_episode(case, method, model, env):
    rule = 'baseline' if method == 'baseline' else 'bank_performance'
    wrapped = TimedGate(MemoizedRecoveryGate(model, case['plant'], rule))
    started = time.perf_counter()
    env.reset_disturbed(case['seed'], case['scenario'], case['disturbance'], .5)
    initial = env.x.copy()
    env.gate = wrapped
    while True:
        _, _, done, truncated, _ = env.step([4]*3)
        if done or truncated:
            break
    row, trace = env.metrics(), env.trace
    # Empty aborted episodes have defined zero variation and capped arrivals;
    # do not take a mean of an empty learning trace or serialize NaN diagnostics.
    previous, state, variation = np.zeros(3), initial.copy(), np.zeros(3)
    for item in trace:
        command = np.array(item['command'])
        variation += (state[:, 0] < env.c.goal)*np.abs(command-previous)
        previous, state = command, np.array(item['x'])
    finish = np.nan_to_num(env.finish, nan=env.c.horizon)
    cost = finish+.05*variation
    row.update(initial=initial.tolist(), plant_name=case['plant'], rule=rule,
        model='none' if method == 'baseline' else method,
        certificate_contract='bounded_actuator_memory_stopping', per_vehicle_cost=cost.tolist(),
        team_cost=float(cost.mean()), per_vehicle_finish=finish.tolist(),
        per_vehicle_variation=variation.tolist(), control_steps=len(trace),
        learned_action_steps=sum(bool(t['actual_learning_used']) for t in trace))
    return json_ready(dict(cluster_id=case['cluster_id'], method=method, row=row, trace=trace,
        calls=wrapped.calls, episode_wall_seconds=time.perf_counter()-started))


def audit_record(case, record, model):
    """Retain genuine collisions/violations; reject inconsistent evidence.

Unlike the old development auditor, an accurately recorded physical failure is
an outcome, not a reason to silently lose the completed study's sample.
SUMO collision flags are retained; route geometry is independently reconstructed.
"""
    row, trace = record['row'], record['trace']
    method = record['method']
    rule = 'baseline' if method == 'baseline' else 'bank_performance'
    if (method not in METHODS or record['cluster_id'] != case['cluster_id']
            or row['seed'] != case['seed'] or row['scenario'] != case['scenario']
            or row['disturbance'] != case['disturbance'] or row['plant_name'] != case['plant']
            or row['rule'] != rule or row['model'] != ('none' if method == 'baseline' else method)
            or row['control_steps'] != len(trace) or len(trace) > 50):
        raise ValueError('Executed case identity or trace length changed')
    check_calls(row, trace, record['calls'])
    plant = matched_plant_parameters(case['seed'], case['plant'], .1)
    tape = disturbance_tape(case['seed'], case['disturbance'], .1, 250)
    x = np.asarray(row['initial'], dtype=float)
    if (x.shape != (3, 2) or not np.isfinite(x).all() or row['plant'] != plant
            or design.shared.array_sha256(x, decimals=8) != case['initial_sha256']
            or design.shared.canonical_sha256(plant) != case['dynamics_sha256']
            or design.shared.array_sha256(tape) != case['disturbance_sha256']
            or row['innovation_tape_sha256'] != tape_fingerprint(tape)
            or row['physical_error_radius'] != .5 or row['external_innovation_radius'] != .1):
        raise ValueError('Physical pairing inputs changed')
    reference = RecoveryJointGate(model, case['plant'], rule)
    memory, previous, variation = np.zeros(3), np.zeros(3), np.zeros(3)
    finish, active = np.full(3, 25.), x[:, 0] < 60
    finish[~active] = 0.
    tau = np.asarray(plant['tau'])
    retention = np.zeros(3)
    retention[tau > 0] = np.exp(-.1/tau[tau > 0])
    gain, drag = np.asarray(plant['gain']), np.asarray(plant['drag'])
    cursor = violations = exceeded = adopted = 0
    max_error = max_slew = 0.
    for step, entry in enumerate(trace):
        expected, info = reference.select(x, previous)
        command = np.asarray(entry['command'])
        if (command.shape != (3,) or not np.isfinite(command).all()
                or np.any(command < -5.-1e-10) or np.any(command > 2.5+1e-10)
                or np.any(np.abs(command-previous) > 2.5+1e-10)):
            raise ValueError('Invalid executed command or slew')
        np.testing.assert_array_equal(command, expected)
        for key, value in info.items():
            if key != 'gate_ms' and entry.get(key) != value:
                raise ValueError('Uncached reference gate replay differs: '+key)
        used = bool(np.any(np.abs(command[active]-np.asarray(info['baseline_command'])[active]) > 1e-9))
        if entry['actual_learning_used'] != used:
            raise ValueError('Adoption differs from executed active commands')
        adopted += used
        variation += active*np.abs(command-previous)
        max_slew = max(max_slew, float(np.max(np.abs(command-previous)))/.5)
        for key in ('innovation', 'actuator_states', 'disturbance'):
            values = np.asarray(entry[key])
            if values.shape != (5, 3) or not np.isfinite(values).all():
                raise ValueError('Incomplete finite physical substeps')
        step_error = 0.
        for sub in range(5):
            np.testing.assert_array_equal(entry['innovation'][sub], tape[cursor])
            memory = retention*memory+(1-retention)*command
            applied = gain*memory-drag*x[:, 1]**2+tape[cursor]
            np.testing.assert_allclose(entry['actuator_states'][sub], memory, atol=1e-9, rtol=0)
            np.testing.assert_allclose(entry['disturbance'][sub], applied-command, atol=1e-9, rtol=0)
            error = float(np.max(np.abs(applied-command)))
            max_error, step_error = max(max_error, error), max(step_error, error)
            exceeded += int(error > .5+1e-9)
            before = x.copy()
            x[:, 1] = np.clip(x[:, 1]+.1*applied, 0, 20)
            x[:, 0] += .1*x[:, 1]
            violations += not bool(transition_safe(before[:, 0][None], x[:, 0][None], MergeConfig())[0])
            cursor += 1
        np.testing.assert_allclose(entry['x'], x, atol=1e-8, rtol=0)
        np.testing.assert_allclose(entry['t'], (step+1)*.5, atol=1e-10, rtol=0)
        if entry['realized_error_in_contract'] != (step_error <= .5+1e-9):
            raise ValueError('Realized error indicator changed')
        x = np.array(entry['x'])
        newly = active & (x[:, 0] >= 60)
        finish[newly], active[newly] = entry['t'], False
        previous = command
    if row['aborted']:
        try:
            reference.select(x, previous)
        except (RuntimeError, ValueError) as exc:
            if str(exc) != row['abort_reason']:
                raise ValueError('Terminal error differs from reference') from exc
        else:
            raise ValueError('Terminal error did not replay')
    elif active.any() and not row['collision'] and len(trace) != 50:
        raise ValueError('Incomplete prefix is not a full episode')
    for value, name in ((finish, 'per_vehicle_finish'), (variation, 'per_vehicle_variation'),
                        (finish+.05*variation, 'per_vehicle_cost'), (max_error, 'max_actual_error'),
                        (max_slew, 'max_command_slew'), (len(trace)*.5, 'elapsed')):
        np.testing.assert_allclose(row[name], value, atol=1e-9, rtol=0)
    if (row['contract_violations'] != violations or row['learned_action_steps'] != adopted
            or row['error_contract_exceeded_substeps'] != exceeded
            or row['nominal_physical_assumption_observed'] != (exceeded == 0)
            or row['completed'] != bool(not active.any() and not row['aborted'] and not row['collision'])):
        raise ValueError('Full episode accounting changed')
    outcome = episode_outcome(row)
    return dict(physical_substeps=cursor, reference_replayed_commands=len(trace),
                recorded_failure=outcome['failed'], genuine_failures_retained=True,
                runtime_independently_certified=False)


def collect_cluster(case, directory, models, *, plan_sha, binding_sha,
                    run_fn=execute_episode, audit_fn=audit_record, env_cls=MatchedShiftedMergeEnv):
    expected = schedule(case)
    if set(models) != set(METHODS[1:]):
        raise ValueError('All fixed models required')
    directory = Path(directory)
    directory.mkdir(exist_ok=False)
    metadata = dict(stage=case['stage'], cluster_id=case['cluster_id'],
                    plan_sha256=plan_sha, binding_sha256=binding_sha)
    store = EpisodeStore(directory/'observations.sqlite', metadata)
    reviews = []
    try:
        with env_cls(plant=case['plant']) as env:
            for method in case['methods']:
                model = None if method == 'baseline' else models[method]
                record = run_fn(case, method, model, env)
                store.append([case['cluster_id'], method],
                    {k: v for k, v in record.items() if k != 'trace'}, record['trace'])
                reviews.append(dict(method=method, **audit_fn(case, record, model)))
        store.seal(expected)
    finally:
        store.close()
    receipt = dict(**metadata, actual_sumo_runs=4, reviews=reviews,
        database_sha256=design.files.digest(directory/'observations.sqlite'),
        seal_sha256=design.files.digest(directory/'observations.sqlite.seal.json'))
    design.files.exclusive_json(directory/'complete.json', receipt)
    return receipt
