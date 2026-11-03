"""Full-episode, finite-portfolio planning under the frozen V4 route contract.

This is a deterministic planning teacher, not a learned policy or global optimum.
The original controller is an incumbent. Every planned action has its own joint
braking witness. Planning cost is reported separately from cached execution.
"""
from dataclasses import dataclass
from itertools import product
import time

import numpy as np

from .merge_v4 import MergeConfig, MergeGate, admissible, advance, backup, certify


BITS = np.array(list(product((0, 1), repeat=3)), dtype=bool)


def batch_certify(states, previous, commands, c):
    """One distinct joint state per command; reject actuator-bound violations."""
    states = np.asarray(states, dtype=float)
    previous = np.asarray(previous, dtype=float)
    commands = np.asarray(commands, dtype=float)
    if states.shape != (len(commands), 3, 2) or previous.shape != commands.shape:
        raise ValueError('Expected states [N,3,2] and commands/previous [N,3]')
    x = states.copy()
    valid = np.all(np.isfinite(x), axis=(1, 2))
    valid &= np.all(np.isfinite(commands), axis=1)
    valid &= np.all(np.abs(admissible(commands, previous, c)-commands) < 1e-9, axis=1)
    valid &= advance(x, commands, c)
    tail = commands.copy()
    for _ in range(c.tail_steps):
        tail = backup(tail, c)
        valid &= advance(x, tail, c)
        if np.all(x[:, :, 1] <= 1e-10):
            break
    return valid & np.all(x[:, :, 1] <= 1e-10, axis=1) & np.all(tail <= 0, axis=1)


def portfolio():
    """37 predeclared feedback schedules; index zero is exactly V4 constant.

    Orders respect the fixed same-road order 0 before 2. Weight ratio trades
    interventions between cars; a short coasting prefix gives the first car room.
    All policies continue to use joint certification; an order is a preference.
    """
    result = [{'name': 'incumbent', 'weights': [1., 1., 1.], 'first': 0, 'coast_steps': 0}]
    for order in ((0, 1, 2), (0, 2, 1), (1, 0, 2)):
        for ratio in (2., 8., 32.):
            weights = np.ones(3)
            weights[list(order)] = [ratio**2, ratio, 1.]
            for coast_steps in (0, 2, 4, 6):
                result.append({'name': f'{order[0]}{order[1]}{order[2]}_r{int(ratio)}_c{coast_steps}',
                               'weights': weights.tolist(), 'first': order[0], 'coast_steps': coast_steps})
    return result


@dataclass
class Plan:
    commands: np.ndarray
    states: np.ndarray
    previous: np.ndarray
    costs: np.ndarray
    profiles: list
    selected: int
    planning_ms: float
    short_scores: np.ndarray
    selection: str

    @property
    def predicted_cost(self):
        return float(self.costs[self.selected])


def plan_episode(x, previous, c=None, profiles=None, selection='full'):
    """Evaluate all profiles through the goal/deadline, without simulator resets.

    The objective is mean capped arrival time, not a short-horizon proxy.
    Intended for the initial state of an episode; not a mid-episode cost API.
    """
    start = time.perf_counter()
    c = c or MergeConfig()
    if selection not in ('full', 'short', 'no_coast'):
        raise ValueError(selection)
    profiles = portfolio() if profiles is None else profiles
    if not profiles or profiles[0] != portfolio()[0]:
        raise ValueError('The exact baseline incumbent must be candidate zero')
    x = np.asarray(x, dtype=float)
    previous = np.asarray(previous, dtype=float)
    if not batch_certify(x[None], previous[None], backup(previous, c)[None], c)[0]:
        raise ValueError('Initial state is outside the declared backup contract')
    n = len(profiles)
    state = np.repeat(x[None], n, axis=0)
    prev = np.repeat(previous[None], n, axis=0)
    weights = np.array([p['weights'] for p in profiles])
    coast_steps = np.array([p['coast_steps'] for p in profiles])
    first = np.array([p['first'] for p in profiles])
    arrived = state[:, :, 0] >= c.goal
    finish = np.where(arrived, 0., c.horizon)
    command_history, state_history, previous_history = [], [], []
    short_scores = -np.minimum(state[:, :, 0], c.goal).sum(axis=1)
    for step in range(round(c.horizon/c.decision_dt)):
        done = arrived.all(axis=1)
        if done.all():
            break
        raw = np.full((n, 3), c.accel)
        coasting = (step < coast_steps)[:, None] & (np.arange(3)[None] != first[:, None])
        raw[coasting & ~arrived] = 0.
        proposal = admissible(raw, prev, c)
        b = backup(prev, c)
        candidates = np.where(BITS[None], proposal[:, None], b[:, None])
        valid = batch_certify(np.repeat(state, 8, axis=0), np.repeat(prev, 8, axis=0),
                              candidates.reshape(-1, 3), c).reshape(n, 8)
        if not valid.any(axis=1).all():
            raise RuntimeError('Lost recursive recoverability during model rollout')
        cost = np.sum(weights[:, None]*(candidates-proposal[:, None])**2, axis=2)
        cost -= 1e-8*candidates.sum(axis=2)
        chosen = candidates[np.arange(n), np.argmin(np.where(valid, cost, np.inf), axis=1)]
        # Completed candidates no longer accrue cost or require execution.
        command_history.append(chosen.copy())
        state_history.append(state.copy())
        previous_history.append(prev.copy())
        next_state = state.copy()
        if not advance(next_state, chosen, c).all():
            raise RuntimeError('Certified first step failed transition check')
        state[~done] = next_state[~done]
        prev[~done] = chosen[~done]
        new = (state[:, :, 0] >= c.goal) & ~arrived
        finish[new] = (step+1)*c.decision_dt
        arrived |= new
        if step < round(2./c.decision_dt):
            short_scores = -np.minimum(state[:, :, 0], c.goal).sum(axis=1)
    costs = finish.mean(axis=1)
    if selection == 'full':
        selected = int(np.argmin(costs))  # incumbent wins exact ties
    elif selection == 'short':
        selected = int(np.argmin(short_scores))
    else:
        selected = int(np.argmin(np.where(coast_steps == 0, costs, np.inf)))
    # Save only the executable witness; retain all profile costs for auditing.
    steps = round(float(finish[selected].max())/c.decision_dt)
    shape = (0, 3)
    commands = np.array(command_history)[:steps, selected] if steps else np.empty(shape)
    states = np.array(state_history)[:steps, selected] if steps else np.empty((0, 3, 2))
    history = np.array(previous_history)[:steps, selected] if steps else np.empty(shape)
    return Plan(commands, states, history, costs, profiles, selected,
                (time.perf_counter()-start)*1000, short_scores, selection)


class PlannedGate:
    """Execute an independently certified plan, falling back on stale plans.

    A mismatch invalidates the suffix permanently. The fallback is the original
    feedback controller, not an unchecked cached command. Safety remains
    conditional on the measured state being in K_B and the declared exact model.
    """
    def __init__(self, c, plan, state_tolerance=1e-6):
        self.c, self.plan = c, plan
        self.state_tolerance = state_tolerance
        self.step = 0
        self.invalidated = False
        self.fallback_count = 0
        self.fallback = MergeGate(c, 'constant')

    def select(self, x, prev, proposal):
        start = time.perf_counter()
        k, c = self.step, self.c
        self.step += 1
        matched = k < len(self.plan.commands)
        if matched:
            matched = (np.max(np.abs(x-self.plan.states[k])) <= self.state_tolerance
                       and np.max(np.abs(prev-self.plan.previous[k])) <= self.state_tolerance)
        self.invalidated |= not matched
        command = self.plan.commands[k] if not self.invalidated else None
        if command is not None:
            valid = batch_certify(x[None], prev[None], command[None], c)[0]
            self.invalidated |= not bool(valid)
        if self.invalidated:
            self.fallback_count += 1
            command, info = self.fallback.select(x, prev, proposal)
        else:
            cruise = admissible(np.full(3, c.accel), prev, c)
            info = {'in_set': bool(certify(x, prev, backup(prev, c), c)[0]),
                    'certified': True, 'intervention': float(np.mean(np.abs(command-cruise) > 1e-5))}
        info.update(gate_ms=(time.perf_counter()-start)*1000, plan_fallback=self.invalidated)
        return command, info
