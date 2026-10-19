"""Diagnostic arrival phase within the existing discrete control period.

Same dynamics, feedback baseline, finite horizon, stage cost and path checks as
the frozen predictor. Interpolated event times are extra diagnostics; existing
evaluation metrics and controllers are unchanged.
"""
import numpy as np
from numba import njit

from .fast_v8 import transition_values
from .joint_planner_v15 import joint_rollout


@njit(cache=True, fastmath=False)
def crossing_time(before, after, goal, elapsed, dt):
    if before < goal <= after:
        return elapsed + dt * (goal - before) / (after - before)
    return np.nan


@njit(cache=True, fastmath=False)
def arrival_phase_suffix(state, previous, step, tape, packed, first_command, use_first):
    x, prev = state.copy(), previous.copy()
    values = np.zeros((3, 2))
    arrivals = np.zeros(3)
    for i in range(3):
        if x[i, 0] < packed[11]:
            arrivals[i] = np.nan
    substeps = round(packed[1] / packed[0])
    for k in range(step, round(packed[2] / packed[1])):
        active = x[:, 0] < packed[11]
        if not np.any(active):
            break
        if use_first and k == step:
            command = first_command.copy()
        else:
            command = joint_rollout(x, prev, k, packed)[0]
        for i in range(3):
            if active[i]:
                values[i, 0] += packed[1] + .05 * abs(command[i] - prev[i])
                values[i, 1] += packed[1]
        for sub in range(substeps):
            positions = x[:, 0].copy()
            for i in range(3):
                acceleration = command[i] + tape[k * substeps + sub, i]
                x[i, 1] = min(packed[3], max(0., x[i, 1] + acceleration * packed[0]))
                x[i, 0] += x[i, 1] * packed[0]
                if np.isnan(arrivals[i]):
                    arrivals[i] = crossing_time(positions[i], x[i, 0], packed[11],
                                                (k - step) * packed[1] + sub * packed[0], packed[0])
            if not transition_values(positions[0], positions[1], positions[2], x[0, 0], x[1, 0], x[2, 0], packed):
                raise RuntimeError('Unsafe arrival-phase forecast transition')
        prev = command
    return values, arrivals, bool(np.all(np.isfinite(arrivals)))
