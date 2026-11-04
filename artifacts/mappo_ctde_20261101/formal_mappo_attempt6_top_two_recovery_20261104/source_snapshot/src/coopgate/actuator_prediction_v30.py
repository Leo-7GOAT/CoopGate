"""Augmented-state feedback prediction under a specified external innovation path.

This is a model evaluator, not a new safety certificate or deployed selector.
Supplying realized future innovations is permitted only for offline diagnostics.
The feedback baseline and its original physical checks remain unchanged.
"""
from numbers import Integral

import numpy as np

from .arrival_phase_v20 import crossing_time
from .joint_planner_v15 import joint_rollout
from .merge_v4 import MergeConfig, transition_safe
from .robust_v9 import robust_certificate, validate


def plant_arrays(plant, dt):
    arrays = [np.asarray(plant[key], dtype=float).copy() for key in ('gain','drag','tau')]
    if (any(a.shape != (3,) or not np.isfinite(a).all() for a in arrays)
            or np.any(arrays[0] <= 0) or np.any(arrays[1] < 0) or np.any(arrays[2] < 0)
            or not np.isfinite(dt) or dt <= 0):
        raise ValueError('Finite positive gains and nonnegative drag/time constants required')
    gain, drag, tau = arrays
    retention = np.zeros(3)
    nonzero = tau > 0
    retention[nonzero] = np.exp(-dt/tau[nonzero])
    return gain, drag, retention


def advance_actuator(state, memory, command, innovation, arrays, dt, vmax):
    """One independent semi-implicit substep; never reuse command-dependent error."""
    gain, drag, retention = arrays
    following_memory = retention*memory+(1.-retention)*command
    acceleration = gain*following_memory-drag*state[:,1]**2+innovation
    following = state.copy()
    following[:,1] = np.clip(state[:,1]+dt*acceleration, 0., vmax)
    following[:,0] = state[:,0]+dt*following[:,1]
    return following, following_memory, acceleration-command


def predict_suffix(state, previous, actuator, step, innovations, plant, first_command=None,
                   config=None, record_trace=False):
    """Predict baseline, or one candidate action followed by the same baseline.

    All branches start with the same measured actuator memory and external path,
    then propagate their own memory and applied accelerations. Incomplete and
    infeasible branches retain their prefix and original horizon-capped costs.
    These capped values cannot authorize a performance claim.
    """
    c = config or MergeConfig()
    x, prev, packed = validate(state, previous, c, .5)
    x, prev = x.copy(), prev.copy()
    memory = np.asarray(actuator, dtype=float).copy()
    if (memory.shape != (3,) or not np.isfinite(memory).all()
            or np.any(memory < -c.brake) or np.any(memory > c.accel)):
        raise ValueError('Finite bounded actuator memory required')
    steps, substeps = round(c.horizon/c.decision_dt), round(c.decision_dt/c.dt)
    if isinstance(step, bool) or not isinstance(step, Integral) or not 0 <= step <= steps:
        raise ValueError('An integer decision time within the original horizon is required')
    tape = np.asarray(innovations, dtype=float)
    if tape.shape != (steps*substeps,3) or not np.isfinite(tape).all():
        raise ValueError('A complete finite external innovation path on the global clock is required')
    first = None if first_command is None else np.asarray(first_command, dtype=float).copy()
    if first is not None and (first.shape != (3,) or not np.isfinite(first).all()
            or np.any(first < -c.brake) or np.any(first > c.accel)
            or np.any(np.abs(first-prev) > c.jerk*c.decision_dt+1e-10)):
        raise ValueError('Candidate command must satisfy the original actuation and slew bounds')
    arrays = plant_arrays(plant, c.dt)
    active = x[:,0] < c.goal
    finish = np.where(active, c.horizon-step*c.decision_dt, 0.)
    crossing = np.where(active, np.nan, 0.)
    variation = np.zeros(3)
    trace, violations, exceeded, maximum, executed = [], 0, 0, 0., 0
    error = None
    for k in range(step, steps):
        active = x[:,0] < c.goal
        if not active.any():
            break
        backup = np.maximum(prev-c.jerk*c.decision_dt, -c.brake)
        if not robust_certificate(x, prev, backup, packed, .5):
            error = dict(type='RuntimeError', message='Joint planner outside backup set', step=k)
            break
        try:
            command = first.copy() if first is not None and k == step else joint_rollout(x, prev, k, packed)[0]
        except (RuntimeError, ValueError) as exc:
            error = dict(type=type(exc).__name__, message=str(exc), step=k)
            break
        if not robust_certificate(x, prev, command, packed, .5):
            error = dict(type='RuntimeError', message='Predicted command lacks the original physical certificate', step=k)
            break
        variation += active*np.abs(command-prev)
        errors, memories, external = [], [], []
        for sub in range(substeps):
            before = x[:,0].copy()
            innovation = tape[k*substeps+sub]
            x, memory, actual_error = advance_actuator(x, memory, command, innovation, arrays, c.dt, c.vmax)
            for i in range(3):
                if np.isnan(crossing[i]):
                    crossing[i] = crossing_time(before[i], x[i,0], c.goal,
                        (k-step)*c.decision_dt+sub*c.dt, c.dt)
            violations += not bool(transition_safe(before[None], x[:,0][None], c)[0])
            magnitude = float(np.max(np.abs(actual_error)))
            maximum = max(maximum, magnitude)
            exceeded += int(magnitude > .5+1e-9)
            errors.append(actual_error.tolist())
            memories.append(memory.tolist())
            external.append(innovation.tolist())
        elapsed = (k-step+1)*c.decision_dt
        finish[active & (x[:,0] >= c.goal)] = elapsed
        prev = command.copy()
        executed += 1
        if record_trace:
            trace.append(dict(t=(k+1)*c.decision_dt, x=x.tolist(), command=command.tolist(),
                innovation=external, actuator_states=memories, disturbance=errors,
                realized_error_in_contract=bool(np.max(np.abs(errors)) <= .5+1e-9)))
    arrived = bool(np.all(x[:,0] >= c.goal))
    return dict(values=np.column_stack((finish+.05*variation, finish)).tolist(),
        crossing=[float(v) if np.isfinite(v) else None for v in crossing],
        variation=variation.tolist(), complete=bool(arrived and error is None and violations == 0),
        all_arrived=arrived, error=error, route_violations=violations, steps=executed,
        max_actual_error=maximum, error_contract_exceeded_substeps=exceeded,
        value_status='complete' if arrived and error is None and violations == 0 else 'capped_incomplete',
        final_state=x.tolist(), final_actuator=memory.tolist(), trace=trace,
        parameters_assumed_known=True, new_physical_certificate=False)


def paired_suffix(state, previous, actuator, step, innovations, plant, command, config=None):
    base = predict_suffix(state, previous, actuator, step, innovations, plant, config=config)
    candidate = predict_suffix(state, previous, actuator, step, innovations, plant, command, config)
    complete = base['complete'] and candidate['complete']
    return dict(baseline=base, single_action_then_baseline=candidate, both_complete=complete,
        difference=(np.asarray(candidate['values'])-base['values']).tolist() if complete else None,
        crossing_difference=(np.asarray(candidate['crossing'])-base['crossing']).tolist() if complete else None,
        counterfactual_errors_reused=False, parameters_assumed_known=True, new_risk_certificate=False)
