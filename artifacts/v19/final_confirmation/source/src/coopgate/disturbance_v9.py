"""SUMO actuator-error harness; identical exogenous tape for paired methods."""
import numpy as np
from .merge_v4 import MergeEnv, ACTIONS, advance, transition_safe


def disturbance_tape(seed, mode, radius, steps):
    rng = np.random.default_rng(seed+901)
    if mode == 'clean': return np.zeros((steps, 3))
    if mode == 'uniform': return rng.uniform(-radius, radius, (steps, 3))
    if mode == 'bias': return np.tile(rng.choice([-radius, radius], 3), (steps, 1))
    if mode == 'pulse':
        tape = np.zeros((steps, 3))
        # Independent of the controller; repeated 1 s extreme pulses, 1 s gaps.
        for start in range(10, steps, 20):
            tape[start:min(start+10, steps)] = rng.choice([-radius, radius], 3)
        return tape
    raise ValueError(mode)


class DisturbedMergeEnv(MergeEnv):
    def reset_disturbed(self, seed, scenario, mode, radius):
        observation = super().reset(seed, scenario)
        self.disturbance_mode = mode; self.disturbance_radius = radius
        self.tape = disturbance_tape(seed, mode, radius, round(self.c.horizon/self.c.dt))
        self.substep = 0; self.abort_reason = ''; self.max_command_slew = 0.; self.applied_errors = []
        return observation

    def step(self, actions):
        c = self.c; previous = self.prev.copy()
        try:
            command, info = self.gate.select(self.x, self.prev, ACTIONS[np.asarray(actions, dtype=int)])
        except (RuntimeError, ValueError) as exc:
            self.abort_reason = str(exc)
            return self.obs(), 0., True, False, self.metrics()
        predicted = self.x[None].copy(); advance(predicted, command[None], c)
        errors = []
        for _ in range(round(c.decision_dt/c.dt)):
            before = self.x.copy(); error = self.tape[self.substep]; self.substep += 1
            errors.append(error.tolist())
            for i, ident in enumerate(self.ids):
                self.conn.vehicle.setSpeed(ident, float(np.clip(self.x[i, 1]+(command[i]+error[i])*c.dt, 0, c.vmax)))
            self.conn.simulationStep(); self.x = self.read()
            self.collision |= bool(self.conn.simulation.getCollidingVehiclesIDList())
            self.violations += not bool(transition_safe(before[:, 0][None], self.x[:, 0][None], c)[0])
        self.t += c.decision_dt; self.prev = command
        self.error = max(self.error, float(np.max(np.abs(predicted[0]-self.x))))
        self.outside += not info['in_set']; self.uncertified += not info['certified']; self.interventions.append(info['intervention'])
        self.jerks.append(float(np.mean(np.abs(command-previous)))/c.decision_dt)
        self.max_command_slew = max(self.max_command_slew, float(np.max(np.abs(command-previous)))/c.decision_dt)
        newly = (self.x[:, 0] >= c.goal) & np.isnan(self.finish); self.finish[newly] = self.t
        self.trace.append({'t': self.t, 'x': self.x.tolist(), 'command': command.tolist(), 'disturbance': errors, **info})
        terminal = self.collision or bool(np.all(np.isfinite(self.finish)))
        truncated = self.t >= c.horizon and not terminal
        return self.obs(), 0., terminal, truncated, self.metrics()

    def metrics(self):
        result = super().metrics()
        result.update(disturbance=self.disturbance_mode, radius=self.disturbance_radius,
                      aborted=bool(self.abort_reason), abort_reason=self.abort_reason, max_command_slew=self.max_command_slew)
        if self.abort_reason:
            result['completed'] = False
            # Administrative aborts are penalized, never quietly omitted.
            result['mean_capped_finish_s'] = self.c.horizon
        return result
