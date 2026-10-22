"""Post-hoc regression mechanisms; original objectives and gates stay unchanged."""
from numbers import Integral
import time

import numpy as np

from .arrival_phase_v20 import arrival_phase_suffix
from .disturbance_v9 import disturbance_tape
from .fast_v8 import parameters
from .joint_planner_v15 import JointRolloutGate
from .matched_shift_v21 import tape_fingerprint
from .merge_v4 import MergeConfig
from .robust_v9 import robust_certificate, validate


def observed_crossings(initial, trace, config=None):
    """Substep-linear interpolation of observed motion, not a new outcome metric.

    Recorded effective error is used only for reconstructing its own observed
    path, never as a counterfactual disturbance under another actuator command.
    """
    c = config or MergeConfig()
    x = np.asarray(initial, dtype=float).copy()
    if x.shape != (3, 2) or not np.isfinite(x).all():
        raise ValueError('Three finite initial states required')
    crossing = np.full(3, np.nan)
    crossing[x[:, 0] >= c.goal] = 0.
    for step, item in enumerate(trace):
        command, errors = np.asarray(item['command']), np.asarray(item['disturbance'])
        if (command.shape != (3,) or errors.shape != (round(c.decision_dt/c.dt), 3)
                or not np.isfinite(command).all() or not np.isfinite(errors).all()):
            raise ValueError('Complete finite observed substeps required')
        if abs(item['t']-(step+1)*c.decision_dt) > 1e-9:
            raise ValueError('Observed decision clock changed')
        for sub, error in enumerate(errors):
            before = x[:, 0].copy()
            x[:, 1] = np.clip(x[:, 1]+(command+error)*c.dt, 0., c.vmax)
            x[:, 0] += x[:, 1]*c.dt
            crossed = np.isnan(crossing) & (before < c.goal) & (x[:, 0] >= c.goal)
            crossing[crossed] = step*c.decision_dt+sub*c.dt+c.dt*(
                c.goal-before[crossed])/(x[crossed, 0]-before[crossed])
        np.testing.assert_allclose(x, item['x'], atol=1e-8, rtol=0)
        x = np.asarray(item['x'], dtype=float).copy()
    return crossing


def decompose_outcome(row, crossing, baseline, baseline_crossing):
    arrays = [np.asarray(v, dtype=float) for v in (row['per_vehicle_finish'],
        baseline['per_vehicle_finish'], row['per_vehicle_variation'], baseline['per_vehicle_variation'],
        crossing, baseline_crossing)]
    if any(a.shape != (3,) or not np.isfinite(a).all() for a in arrays):
        raise ValueError('Complete finite paired arrivals required for decomposition')
    t, bt, variation, bvariation, continuous, bcontinuous = arrays
    dt, dvariation = t-bt, .05*(variation-bvariation)
    dcrossing = continuous-bcontinuous
    rounding = (t-continuous)-(bt-bcontinuous)
    delta = np.asarray(row['per_vehicle_cost'])-baseline['per_vehicle_cost']
    np.testing.assert_allclose(delta, dt+dvariation, atol=1e-12, rtol=0)
    np.testing.assert_allclose(dt, dcrossing+rounding, atol=1e-12, rtol=0)
    return dict(logged_finish_delta=dt.tolist(), interpolated_crossing_delta=dcrossing.tolist(),
        decision_rounding_delta=rounding.tolist(), variation_cost_delta=dvariation.tolist(),
        original_cost_delta=delta.tolist(), interpolated_cost_diagnostic_delta=(dcrossing+dvariation).tolist(),
        observed_crossing=continuous.tolist(), baseline_observed_crossing=bcontinuous.tolist(),
        original_outcome_definition_changed=False)


def check_admission(item):
    bank = np.asarray(item['selected_bank_differences'])
    phase = np.asarray(item['selected_phase_differences'])
    if (not item['nominal_joint_action_admitted'] or bank.shape != (33, 3, 2)
            or phase.shape != (33, 3) or not np.isfinite(bank).all() or not np.isfinite(phase).all()
            or np.any(bank > 1e-10) or np.any(phase > .05+1e-10)
            or bank[0, :, 0].mean() >= -1e-9):
        raise ValueError('A complete original phase05 admission is required')
    return bank, phase


def realized_noise_branches(row, trace, step):
    """Diagnostic oracle: the recorded external-noise future is never deployed.

    Only nominal actuators are supported. Effective error from one mismatched
    trajectory cannot be transplanted into another command-dependent plant.
    """
    plant = row['plant']
    if (row['plant_name'] != 'nominal' or plant['kind'] != 'nominal'
            or not np.array_equal(plant['gain'], np.ones(3))
            or np.any(plant['drag']) or np.any(plant['tau'])):
        raise ValueError('Counterfactual additive replay requires nominal actuator dynamics')
    if isinstance(step, bool) or not isinstance(step, Integral) or not 0 <= step < len(trace):
        raise ValueError('An existing admitted decision step is required')
    item = trace[step]
    bank, phase = check_admission(item)
    c = MergeConfig()
    tape = disturbance_tape(row['seed'], row['disturbance'], row['external_innovation_radius'], round(c.horizon/c.dt))
    if tape_fingerprint(tape) != row['innovation_tape_sha256']:
        raise ValueError('Recorded external-noise identity differs')
    x = np.asarray(row['initial'] if step == 0 else trace[step-1]['x'])
    previous = np.zeros(3) if step == 0 else np.asarray(trace[step-1]['command'])
    command = np.asarray(item['command'])
    outcomes = {}
    for role, first in (('baseline', False), ('single_action_then_baseline', True)):
        try:
            values, crossing, complete = arrival_phase_suffix(x, previous, step, tape, parameters(c), command, first)
            outcomes[role] = dict(values=values.tolist(), crossing=[float(v) if np.isfinite(v) else None for v in crossing],
                                  complete=bool(complete), error=None)
        except (ValueError, RuntimeError) as exc:
            outcomes[role] = dict(values=None, crossing=None, complete=False,
                                  error=dict(type=type(exc).__name__, message=str(exc)))
    a, b = outcomes['single_action_then_baseline'], outcomes['baseline']
    complete = a['complete'] and b['complete']
    difference = np.asarray(a['values'])-b['values'] if complete else None
    phase_delta = np.asarray(a['crossing'])-b['crossing'] if complete else None
    return dict(step=int(step), decision_time=step*c.decision_dt,
        original_bank_maximum=bank.max(axis=0).tolist(), original_phase_maximum=phase.max(axis=0).tolist(),
        branches=outcomes, both_branches_complete=complete,
        realized_difference=None if difference is None else difference.tolist(),
        realized_phase_delta=None if phase_delta is None else phase_delta.tolist(),
        realized_local_regression=None if difference is None else bool(np.any(difference > 1e-9)),
        diagnostic_future_noise_only=True)


class SingleInterventionBaseline:
    """One archived command at its recorded state; feedback baseline elsewhere."""
    def __init__(self, target_step, state, previous, command):
        if isinstance(target_step, bool) or not isinstance(target_step, Integral) or target_step < 0:
            raise ValueError('Nonnegative intervention decision required')
        self.target_step = int(target_step)
        self.target_state = np.asarray(state, dtype=float).copy()
        self.target_previous = np.asarray(previous, dtype=float).copy()
        self.target_command = np.asarray(command, dtype=float).copy()
        self.c = MergeConfig()
        self.packed = parameters(self.c)
        validate(self.target_state, self.target_previous, self.c, .5)
        if (self.target_command.shape != (3,) or not np.isfinite(self.target_command).all()
                or not robust_certificate(self.target_state, self.target_previous, self.target_command, self.packed, .5)):
            raise ValueError('Archived intervention lacks physical feasibility')
        self.baseline = JointRolloutGate(self.c)
        self.step = 0

    def select(self, state, previous, unused=None):
        started = time.perf_counter()
        base, info = self.baseline.select(state, previous)
        use = self.step == self.target_step
        if use:
            np.testing.assert_allclose(state, self.target_state, rtol=0, atol=1e-8)
            np.testing.assert_allclose(previous, self.target_previous, rtol=0, atol=1e-12)
            command = self.target_command.copy()
        else:
            command = base
        if not robust_certificate(np.asarray(state), np.asarray(previous), command, self.packed, .5):
            raise RuntimeError('Diagnostic command left the common physical contract')
        self.step += 1
        info.update(gate_ms=1000*(time.perf_counter()-started), diagnostic_frozen_action=use,
            nominal_joint_action_admitted=use, baseline_command=base.tolist(),
            diagnostic_only=True, production_controller_changed=False)
        return command, info
