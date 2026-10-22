"""Paired actuator transfer with one external-noise law for every plant.

The original controller's physical error contract remains +/-0.5. External
innovation and actual command error are different quantities under mismatch.
The auditor below reconstructs dynamics without calling the simulator's
apply_actuator helper or assuming that the realized error satisfies that contract.
"""
import hashlib

import numpy as np

from .disturbance_v9 import disturbance_tape
from .fast_v8 import parameters
from .merge_v4 import MergeConfig, transition_safe
from .robust_v9 import robust_certificate
from .shifted_sumo_v19 import ShiftedMergeEnv, plant_parameters


INNOVATION_RADIUS = .1


def matched_plant_parameters(seed, plant, innovation_radius=INNOVATION_RADIUS):
    if (isinstance(innovation_radius, (bool, np.bool_)) or
            not np.isfinite(innovation_radius) or not 0 <= innovation_radius <= .5):
        raise ValueError('Innovation radius must be finite and in [0, .5]')
    result = plant_parameters(seed, plant)
    result['innovation_radius'] = float(innovation_radius)
    return result


def tape_fingerprint(tape):
    return hashlib.sha256(np.asarray(tape, dtype='<f8').tobytes(order='C')).hexdigest()


class MatchedShiftedMergeEnv(ShiftedMergeEnv):
    def __init__(self, plant='nominal', innovation_radius=INNOVATION_RADIUS, **kwargs):
        matched_plant_parameters(0, plant, innovation_radius)
        self.innovation_radius = float(innovation_radius)
        super().__init__(plant=plant, **kwargs)

    def reset_disturbed(self, seed, scenario, mode, radius):
        super().reset_disturbed(seed, scenario, mode, radius)
        self.plant = matched_plant_parameters(seed, self.plant_kind, self.innovation_radius)
        self.tape = disturbance_tape(seed, mode, self.innovation_radius,
                                     round(self.c.horizon / self.c.dt))
        self.innovation_tape_sha256 = tape_fingerprint(self.tape)
        return self.obs()

    def metrics(self):
        result = super().metrics()
        if hasattr(self, 'innovation_tape_sha256'):
            result.update(external_innovation_radius=self.innovation_radius,
                          physical_error_radius=.5,
                          innovation_tape_sha256=self.innovation_tape_sha256)
        return result


def audit_matched_trace(trace, row, expected_radius=INNOVATION_RADIUS):
    """Check noise identity, independent actuator recurrence and full accounting.

    A recorded abort is retained. No claim is made about motion after that abort.
    SUMO collision reports are retained separately from route-geometry checks.
    """
    c = MergeConfig()
    packed = parameters(c)
    plant = matched_plant_parameters(row['seed'], row['plant_name'], expected_radius)
    if (row['plant'] != plant or row['external_innovation_radius'] != expected_radius or
            row['physical_error_radius'] != .5 or row['radius'] != .5):
        raise ValueError('Plant, external noise or controller contract changed')
    tape = disturbance_tape(row['seed'], row['disturbance'], expected_radius,
                            round(c.horizon / c.dt))
    if row['innovation_tape_sha256'] != tape_fingerprint(tape):
        raise ValueError('External innovation tape changed')
    state = np.asarray(row['initial'], dtype=float).copy()
    if state.shape != (3, 2) or not np.isfinite(state).all():
        raise ValueError('Invalid initial state')
    previous, actuator = np.zeros(3), np.zeros(3)
    finish, active, variation = np.full(3, c.horizon), np.ones(3, bool), np.zeros(3)
    tau = np.asarray(plant['tau'])
    retention = np.zeros(3)
    retention[tau > 0] = np.exp(-c.dt / tau[tau > 0])
    gain, drag = np.asarray(plant['gain']), np.asarray(plant['drag'])
    maximum, exceeded, violations, cursor, max_slew = 0., 0, 0, 0, 0.
    substeps = round(c.decision_dt / c.dt)
    if len(trace) > round(c.horizon / c.decision_dt):
        raise ValueError('Trace exceeds the declared horizon')
    for step, item in enumerate(trace):
        command = np.asarray(item['command'])
        if (command.shape != (3,) or not np.isfinite(command).all() or
                np.any(command < -5. - 1e-10) or np.any(command > 2.5 + 1e-10) or
                np.any(np.abs(command - previous) > 2.5 + 1e-10)):
            raise ValueError('Invalid commanded actuation or slew')
        if not robust_certificate(state, previous, command, packed, .5):
            raise ValueError('Executed command lacks its declared nominal physical certificate')
        np.testing.assert_allclose(item['t'], (step + 1) * c.decision_dt, atol=1e-10, rtol=0)
        variation += active * np.abs(command - previous)
        max_slew = max(max_slew, float(np.max(np.abs(command - previous))) / c.decision_dt)
        for key in ('innovation', 'actuator_states', 'disturbance'):
            values = np.asarray(item[key])
            if values.shape != (substeps, 3) or not np.isfinite(values).all():
                raise ValueError('Incomplete substep evidence: ' + key)
        step_maximum = 0.
        for j in range(substeps):
            innovation = tape[cursor]
            cursor += 1
            np.testing.assert_array_equal(innovation, item['innovation'][j])
            # Independent recurrence: do not import/call apply_actuator here.
            actuator = retention * actuator + (1. - retention) * command
            applied = gain * actuator - drag * state[:, 1] ** 2 + innovation
            error = applied - command
            np.testing.assert_allclose(actuator, item['actuator_states'][j], atol=1e-9, rtol=0)
            np.testing.assert_allclose(error, item['disturbance'][j], atol=1e-9, rtol=0)
            error_size = float(np.max(np.abs(error)))
            maximum, step_maximum = max(maximum, error_size), max(step_maximum, error_size)
            exceeded += int(error_size > .5 + 1e-9)
            before = state[:, 0].copy()
            state[:, 1] = np.clip(state[:, 1] + applied * c.dt, 0., c.vmax)
            state[:, 0] += state[:, 1] * c.dt
            violations += not bool(transition_safe(before[None], state[:, 0][None], c)[0])
        if item['realized_error_in_contract'] != (step_maximum <= .5 + 1e-9):
            raise ValueError('Realized-error indicator differs')
        np.testing.assert_allclose(state, item['x'], atol=1e-8, rtol=0)
        state = np.asarray(item['x']).copy()
        newly = active & (state[:, 0] >= c.goal)
        finish[newly] = item['t']
        active[newly] = False
        previous = command
    for actual, key in ((finish, 'per_vehicle_finish'), (variation, 'per_vehicle_variation'),
                        (maximum, 'max_actual_error'), (max_slew, 'max_command_slew'),
                        (len(trace) * c.decision_dt, 'elapsed')):
        np.testing.assert_allclose(actual, row[key], atol=1e-9, rtol=0)
    if (exceeded != row['error_contract_exceeded_substeps'] or
            violations != row['contract_violations'] or
            row['nominal_physical_assumption_observed'] != (exceeded == 0)):
        raise ValueError('Physical mismatch/violation accounting differs')
    completed = bool(not active.any() and not row['collision'] and not row['aborted'])
    if completed != row['completed'] or bool(row['abort_reason']) != row['aborted']:
        raise ValueError('Completion/abort accounting differs')
    cost = finish + .05 * variation
    if 'per_vehicle_cost' in row:
        np.testing.assert_allclose(cost, row['per_vehicle_cost'], atol=1e-9, rtol=0)
        np.testing.assert_allclose(cost.mean(), row['team_cost'], atol=1e-9, rtol=0)
    return dict(substeps=cursor, error_contract_exceeded_substeps=exceeded,
                maximum_actual_error=maximum, completed=completed, aborted=row['aborted'])
