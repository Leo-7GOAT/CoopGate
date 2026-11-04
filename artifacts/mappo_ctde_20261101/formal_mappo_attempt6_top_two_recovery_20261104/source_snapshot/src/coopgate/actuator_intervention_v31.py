"""Paired interventions at recorded states, with the physical prefix retained.

This module is for mechanism diagnosis. Neither the archived action nor the
realized future innovation is a proposed deployable decision rule.
"""
from numbers import Integral
import time

import numpy as np

from .joint_planner_v15 import JointRolloutGate
from .merge_v4 import MergeConfig
from .regression_diagnostic_v28 import check_admission, observed_crossings
from .robust_v9 import robust_certificate


ROLES = ('baseline_suffix', 'candidate_then_baseline')


def intervention_state(row, trace, step):
    if isinstance(step, bool) or not isinstance(step, Integral) or not 0 <= step < len(trace):
        raise ValueError('An archived admission index is required')
    check_admission(trace[step])
    state = np.asarray(row['initial'] if step == 0 else trace[step-1]['x'], dtype=float)
    previous = np.zeros(3) if step == 0 else np.asarray(trace[step-1]['command'], dtype=float)
    memory = np.zeros(3) if step == 0 else np.asarray(trace[step-1]['actuator_states'][-1], dtype=float)
    return state.copy(), previous.copy(), memory.copy()


def prefix_values(initial, prefix, config=None):
    """Original accrued costs, stopping each vehicle's clock at its arrival."""
    c = config or MergeConfig()
    state, previous = np.asarray(initial, dtype=float), np.zeros(3)
    values = np.zeros((3, 2))
    for step, item in enumerate(prefix):
        if abs(item['t']-(step+1)*c.decision_dt) > 1e-9:
            raise ValueError('The original prefix clock is required')
        command = np.asarray(item['command'], dtype=float)
        active = state[:, 0] < c.goal
        values[:, 0] += active*(c.decision_dt+.05*np.abs(command-previous))
        values[:, 1] += active*c.decision_dt
        state, previous = np.asarray(item['x'], dtype=float), command
    return values


class RecordedPrefixBranch:
    """Replay the entire prefix, then intervene once or use feedback baseline.

    Previous learning actions are not replaced by baseline actions. The two
    branches reach the same actuator memory by physical execution, not a state
    injection, and keep the original global decision clock.
    """
    def __init__(self, row, trace, step, role):
        if role not in ROLES:
            raise ValueError('Unknown intervention branch')
        intervention_state(row, trace, step)
        self.initial = np.asarray(row['initial'], dtype=float).copy()
        self.prefix = [dict(x=np.asarray(t['x'], dtype=float).copy(),
                            command=np.asarray(t['command'], dtype=float).copy()) for t in trace[:step+1]]
        self.recorded_baseline = np.asarray(trace[step]['baseline_command'], dtype=float).copy()
        self.target, self.role, self.step = int(step), role, 0
        self.baseline = JointRolloutGate()

    def select(self, state, previous, unused=None):
        started = time.perf_counter()
        base, info = self.baseline.select(state, previous)
        if self.step <= self.target:
            expected_state = self.initial if self.step == 0 else self.prefix[self.step-1]['x']
            expected_previous = np.zeros(3) if self.step == 0 else self.prefix[self.step-1]['command']
            np.testing.assert_allclose(state, expected_state, atol=1e-8, rtol=0)
            np.testing.assert_array_equal(previous, expected_previous)
        at_boundary = self.step == self.target
        if at_boundary:
            np.testing.assert_array_equal(base, self.recorded_baseline)
        archived = self.step < self.target or (at_boundary and self.role == 'candidate_then_baseline')
        command = self.prefix[self.step]['command'].copy() if archived else base
        if not robust_certificate(np.asarray(state), np.asarray(previous), command, self.baseline.packed, .5):
            raise RuntimeError('Recorded intervention violates the original physical check')
        self.step += 1
        info.update(gate_ms=1000*(time.perf_counter()-started), baseline_command=base.tolist(),
            diagnostic_archived_prefix=bool(self.step-1 < self.target),
            diagnostic_intervention=bool(at_boundary and self.role == 'candidate_then_baseline'),
            diagnostic_only=True, production_controller_changed=False)
        return command, info


def trace_error(left, right, field):
    if len(left) != len(right):
        return None
    return max((float(np.max(np.abs(np.asarray(a[field])-b[field])))
                for a, b in zip(left, right)), default=0.)


def compare_branch(prediction, row, trace, source, source_trace, step):
    """Compare independent full SUMO execution with a boundary-state forecast."""
    prefix, suffix = trace[:step], trace[step:]
    prefix_exact = len(prefix) == step and all(
        a['command'] == b['command'] and np.allclose(a['x'], b['x'], atol=1e-8, rtol=0)
        and np.allclose(a['actuator_states'], b['actuator_states'], atol=1e-9, rtol=0)
        for a, b in zip(prefix, source_trace[:step]))
    offset = prefix_values(source['initial'], source_trace[:step])
    whole = np.column_stack((row['per_vehicle_cost'], row['per_vehicle_finish']))
    metric_error = float(np.max(np.abs(np.asarray(prediction['values'])+offset-whole)))
    pt = prediction['trace']
    errors = {name:trace_error(pt, suffix, name) for name in ('x','actuator_states','disturbance','innovation')}
    state, _, _ = intervention_state(source, source_trace, step)
    local_trace = [dict(item, t=item['t']-step*.5) for item in suffix]
    crossings = observed_crossings(state, local_trace)
    predicted_crossings = np.asarray([np.nan if v is None else v for v in prediction['crossing']])
    finite = np.isfinite(crossings)
    crossing_error = float(np.max(np.abs(crossings[finite]-predicted_crossings[finite]))) if finite.any() else 0.
    effective = [float(np.max(np.abs(item['disturbance'][j]))) for item in suffix for j in range(5)]
    expected_abort = prediction['error']
    checks = dict(prefix_exact=bool(prefix_exact), same_suffix_length=len(pt) == len(suffix),
        commands_exact=len(pt) == len(suffix) and all(a['command'] == b['command'] for a,b in zip(pt,suffix)),
        state_match=errors['x'] is not None and errors['x'] <= 1e-8,
        memory_match=errors['actuator_states'] is not None and errors['actuator_states'] <= 1e-9,
        command_error_match=errors['disturbance'] is not None and errors['disturbance'] <= 1e-9,
        external_path_match=errors['innovation'] == 0., metrics_match=metric_error <= 1e-9,
        crossing_match=bool(np.array_equal(np.isfinite(predicted_crossings),finite) and crossing_error <= 1e-8),
        completion_match=bool(prediction['complete'] == row['completed']),
        abort_match=bool((expected_abort is None) == (not row['aborted']) and
            (expected_abort is None or (expected_abort['message'] == row['abort_reason'] and
                                       expected_abort['step'] == len(trace)))),
        error_bound_count_match=prediction['error_contract_exceeded_substeps'] == sum(e > .5+1e-9 for e in effective),
        maximum_error_match=abs(prediction['max_actual_error']-max(effective,default=0.)) <= 1e-9)
    return dict(all_match=all(checks.values()), checks=checks, suffix_commands=len(suffix),
        full_commands=len(trace), state_max_abs_error=errors['x'], metric_max_abs_error=metric_error,
        crossing_max_abs_error=crossing_error, original_metrics_preserved=True)


def pair_effect(branches):
    """Do not rank capped failures as completed candidate improvements."""
    base, candidate = (branches[name] for name in ROLES)
    complete = base['complete'] and candidate['complete']
    delta = np.asarray(candidate['values'])-base['values'] if complete else None
    return dict(both_complete=bool(complete), baseline_complete=bool(base['complete']),
        candidate_complete=bool(candidate['complete']),
        difference=None if delta is None else delta.tolist(),
        local_regression=None if delta is None else bool(np.any(delta > 1e-9)),
        mean_cost_difference=None if delta is None else float(delta[:,0].mean()),
        admissible_performance_evidence=bool(complete and np.all(delta <= 1e-9)))
