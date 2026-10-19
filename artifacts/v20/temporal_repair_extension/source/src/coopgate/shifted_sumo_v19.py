"""SUMO execution with explicit initial-distribution and actuator-model shifts.

All methods share plant parameters and external innovations. Realized errors
are action dependent under gain/drag/lag, so are NOT forced equal across methods.
The controller receives measured kinematics and previous command only.
"""
import numpy as np

from .disturbance_v9 import DisturbedMergeEnv, disturbance_tape
from .fast_v8 import parameters
from .merge_v4 import ACTIONS, BRANCH, advance, transition_safe
from .robust_v9 import robust_certificate


SCENARIOS = ('nominal', 'dense', 'slow_yield', 'asymmetric_speed', 'tight_headway')
PLANTS = ('nominal', 'gain_drag', 'lag_mild', 'lag_strong')


def shifted_initial(seed, scenario):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 19031]))
    if scenario == 'slow_yield':
        velocity = rng.uniform(2., 6., 3)
        position = -rng.uniform(45., 80., 3)
        position[2] = position[0] - rng.uniform(18., 32.)
    elif scenario == 'asymmetric_speed':
        velocity = np.array([rng.uniform(8., 11.), rng.uniform(16., 19.), rng.uniform(8., 12.)])
        position = -rng.uniform(85., 110., 3)
        position[2] = position[0] - rng.uniform(30., 45.)
    elif scenario == 'tight_headway':
        velocity = rng.uniform(12., 15., 3)
        velocity[2] = velocity[0] + rng.uniform(-.5, .5)
        position = -rng.uniform(70., 85., 3)
        position[2] = position[0] - rng.uniform(18., 25.)
    else:
        raise ValueError('Use original SUMO sampler for nominal/dense')
    return np.column_stack((position, velocity))


def plant_parameters(seed, plant):
    if plant not in PLANTS:
        raise ValueError('Unknown actuator plant')
    rng = np.random.default_rng(np.random.SeedSequence([seed, 19073]))
    gain, drag, tau = np.ones(3), np.zeros(3), np.zeros(3)
    if plant == 'gain_drag':
        gain = rng.uniform(.96, 1.04, 3)
        drag = rng.uniform(.00025, .0005, 3)
    elif plant == 'lag_mild':
        tau = rng.uniform(.03, .05, 3)
    elif plant == 'lag_strong':
        tau = rng.uniform(.3, .5, 3)
    return dict(kind=plant, gain=gain.tolist(), drag=drag.tolist(), tau=tau.tolist(),
                innovation_radius=.5 if plant == 'nominal' else .1)


def apply_actuator(command, velocity, previous_actuator, innovation, plant, dt=.1):
    command = np.asarray(command, dtype=float)
    tau = np.array(plant['tau'])
    retain = np.zeros(3)
    positive = tau > 0
    retain[positive] = np.exp(-dt / tau[positive])
    actuator = retain * previous_actuator + (1. - retain) * command
    applied = np.array(plant['gain']) * actuator - np.array(plant['drag']) * velocity ** 2 + innovation
    return applied, actuator


class ShiftedMergeEnv(DisturbedMergeEnv):
    def __init__(self, plant='nominal', **kwargs):
        if plant not in PLANTS:
            raise ValueError('Unknown actuator plant')
        self.plant_kind = plant
        super().__init__(**kwargs)

    def reset_disturbed(self, seed, scenario, mode, radius):
        if scenario not in SCENARIOS or radius != .5:
            raise ValueError('Unknown shift or unsupported nominal contract')
        super().reset_disturbed(seed, scenario if scenario in ('nominal', 'dense') else 'dense', mode, radius)
        if scenario not in ('nominal', 'dense'):
            target = shifted_initial(seed, scenario)
            # Place vehicles before the bookkeeping time origin, then verify the
            # actual simulator state. No state injection occurs during a rollout.
            for i, ident in enumerate(self.ids):
                route = 'a' if BRANCH[i] == 0 else 'b'
                length = self.conn.lane.getLength(route + '_0')
                self.conn.vehicle.moveTo(ident, route + '_0', float(length + target[i, 0] - target[i, 1] * self.c.dt))
                self.conn.vehicle.setSpeed(ident, float(target[i, 1]))
            self.conn.simulationStep()
            self.offsets = []
            for i, ident in enumerate(self.ids):
                route = 'a' if BRANCH[i] == 0 else 'b'
                position = self.conn.vehicle.getLanePosition(ident) - self.conn.lane.getLength(route + '_0')
                self.offsets.append(position - self.conn.vehicle.getDistance(ident))
            self.x = self.read()
            np.testing.assert_allclose(self.x, target, atol=1e-8, rtol=0)
            self.initial = self.x.copy()
        self.scenario = scenario
        if not robust_certificate(self.x, np.zeros(3), np.full(3, -2.5), parameters(self.c), .5):
            raise RuntimeError('Generated shifted state outside nominal recoverable set; no seed rejection')
        self.plant = plant_parameters(seed, self.plant_kind)
        self.tape = disturbance_tape(seed, mode, self.plant['innovation_radius'], round(self.c.horizon / self.c.dt))
        self.actuator = np.zeros(3)
        self.max_actual_error = 0.
        self.error_contract_exceeded_substeps = 0
        return self.obs()

    def step(self, actions):
        c = self.c
        previous = self.prev.copy()
        try:
            command, info = self.gate.select(self.x, self.prev, ACTIONS[np.asarray(actions, dtype=int)])
        except (RuntimeError, ValueError) as exc:
            self.abort_reason = str(exc)
            return self.obs(), 0., True, False, self.metrics()
        predicted = self.x[None].copy()
        advance(predicted, command[None], c)
        errors, innovations, actuator_states = [], [], []
        for _ in range(round(c.decision_dt / c.dt)):
            before = self.x.copy()
            innovation = self.tape[self.substep]
            self.substep += 1
            applied, self.actuator = apply_actuator(command, self.x[:, 1], self.actuator, innovation, self.plant, c.dt)
            error = applied - command
            errors.append(error.tolist())
            innovations.append(innovation.tolist())
            actuator_states.append(self.actuator.tolist())
            self.max_actual_error = max(self.max_actual_error, float(np.max(np.abs(error))))
            self.error_contract_exceeded_substeps += int(np.max(np.abs(error)) > .5 + 1e-9)
            for i, ident in enumerate(self.ids):
                self.conn.vehicle.setSpeed(ident, float(np.clip(self.x[i, 1] + applied[i] * c.dt, 0, c.vmax)))
            self.conn.simulationStep()
            self.x = self.read()
            self.collision |= bool(self.conn.simulation.getCollidingVehiclesIDList())
            self.violations += not bool(transition_safe(before[:, 0][None], self.x[:, 0][None], c)[0])
        self.t += c.decision_dt
        self.prev = command
        self.error = max(self.error, float(np.max(np.abs(predicted[0] - self.x))))
        self.outside += not info['in_set']
        self.uncertified += not info['certified']
        self.interventions.append(info['intervention'])
        self.jerks.append(float(np.mean(np.abs(command - previous))) / c.decision_dt)
        self.max_command_slew = max(self.max_command_slew, float(np.max(np.abs(command - previous))) / c.decision_dt)
        newly = (self.x[:, 0] >= c.goal) & np.isnan(self.finish)
        self.finish[newly] = self.t
        self.trace.append(dict(t=self.t, x=self.x.tolist(), command=command.tolist(), disturbance=errors,
                               innovation=innovations, actuator_states=actuator_states,
                               realized_error_in_contract=bool(np.max(np.abs(errors)) <= .5 + 1e-9), **info))
        terminal = self.collision or bool(np.all(np.isfinite(self.finish)))
        truncated = self.t >= c.horizon and not terminal
        return self.obs(), 0., terminal, truncated, self.metrics()

    def metrics(self):
        result = super().metrics()
        if hasattr(self, 'plant'):
            result.update(plant=self.plant, max_actual_error=self.max_actual_error,
                          error_contract_exceeded_substeps=self.error_contract_exceeded_substeps,
                          nominal_physical_assumption_observed=self.error_contract_exceeded_substeps == 0)
        return result


def audit_shifted_trace(trace, row):
    """Reconstruct actuator memory and every SUMO substep, including mismatches."""
    state = np.array(row['initial'])
    previous, actuator = np.zeros(3), np.zeros(3)
    finish, active, variation = np.full(3, 25.), np.ones(3, bool), np.zeros(3)
    from .merge_v4 import MergeConfig
    c = MergeConfig()
    packed = parameters(c)
    exceed = 0
    maximum = 0.
    violations = 0
    for item in trace:
        command = np.array(item['command'])
        assert robust_certificate(state, previous, command, packed, .5)
        variation += active * np.abs(command - previous)
        for j, innovation in enumerate(item['innovation']):
            applied, actuator = apply_actuator(command, state[:, 1], actuator, np.array(innovation), row['plant'])
            np.testing.assert_allclose(actuator, item['actuator_states'][j], atol=1e-9, rtol=0)
            np.testing.assert_allclose(applied - command, item['disturbance'][j], atol=1e-9, rtol=0)
            maximum = max(maximum, float(np.max(np.abs(applied - command))))
            exceed += int(np.max(np.abs(applied - command)) > .5 + 1e-9)
            old = state[:, 0].copy()
            state[:, 1] = np.clip(state[:, 1] + applied * .1, 0., 20.)
            state[:, 0] += state[:, 1] * .1
            violations += not bool(transition_safe(old[None], state[:, 0][None], c)[0])
        np.testing.assert_allclose(state, item['x'], atol=1e-8, rtol=0)
        state = np.array(item['x'])
        newly = active & (state[:, 0] >= 60.)
        finish[newly] = item['t']
        active[newly] = False
        previous = command
    np.testing.assert_allclose(finish, row['per_vehicle_finish'], atol=1e-9, rtol=0)
    np.testing.assert_allclose(variation, row['per_vehicle_variation'], atol=1e-9, rtol=0)
    np.testing.assert_allclose(maximum, row['max_actual_error'], atol=1e-9, rtol=0)
    assert exceed == row['error_contract_exceeded_substeps']
    assert violations == row['contract_violations']
