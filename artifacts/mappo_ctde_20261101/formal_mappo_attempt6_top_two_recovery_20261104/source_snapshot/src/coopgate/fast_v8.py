"""Compiled exact-model finite-portfolio verification and admissible pruning.

Strict float64 arithmetic, no fastmath, parallel workers or relaxed certificate.
Learning may order candidates; it never authorizes a command or prunes a plan.
"""
import time
import numpy as np
from numba import njit
import torch

from .merge_v4 import MergeConfig, MergeGate, admissible
from .learning_v6 import features
from .planning_v5 import Plan, portfolio


def parameters(c):
    return np.array([c.dt, c.decision_dt, c.horizon, c.vmax, c.brake, c.accel,
                     c.jerk, c.separation, c.zone_start, c.zone_end, c.tail_steps, c.goal], dtype=np.float64)


@njit(cache=True, fastmath=False)
def clip(value, lo, hi):
    return min(max(value, lo), hi)


@njit(cache=True, fastmath=False, inline='always')
def pair_safe(pi, pj, qi, qj, c):
    ei, li, ej, lj = -np.inf, np.inf, -np.inf, np.inf
    di, dj = qi-pi, qj-pj
    if di > 1e-12:
        ei, li = (c[8]-pi)/di, (c[9]-pi)/di
    if dj > 1e-12:
        ej, lj = (c[8]-pj)/dj, (c[9]-pj)/dj
    ei, li, ej, lj = max(0., ei), min(1., li), max(0., ej), min(1., lj)
    if qi >= c[8] and pi <= c[9] and qj >= c[8] and pj <= c[9] and max(ei, ej) <= min(li, lj):
        return False
    start = clip(max(ei, ej), 0., 1.)
    d0 = (pi-pj)+start*(di-dj); d1 = qi-qj
    gap = 0. if d0*d1 <= 0 else min(abs(d0), abs(d1))
    return not (qi >= c[8] and qj >= c[8] and gap < c[7]-1e-8)


@njit(cache=True, fastmath=False, inline='always')
def transition_values(p0, p1, p2, q0, q1, q2, c):
    return (min(p0-p2, q0-q2) >= c[7]-1e-8 and pair_safe(p0, p1, q0, q1, c)
            and pair_safe(p1, p2, q1, q2, c))


@njit(cache=True, fastmath=False)
def transition(p, q, c):
    return transition_values(p[0], p[1], p[2], q[0], q[1], q[2], c)


@njit(cache=True, fastmath=False)
def advance_one(state, command, c):
    for _ in range(round(c[1]/c[0])):
        p0, p1, p2 = state[0, 0], state[1, 0], state[2, 0]
        for i in range(3):
            state[i, 1] = clip(state[i, 1]+command[i]*c[0], 0., c[3])
            state[i, 0] += state[i, 1]*c[0]
        if not transition_values(p0, p1, p2, state[0, 0], state[1, 0], state[2, 0], c):
            return False
    return True


@njit(cache=True, fastmath=False)
def certificate(state, previous, command, c):
    for i in range(3):
        bounded = clip(command[i], max(-c[4], previous[i]-c[6]*c[1]), min(c[5], previous[i]+c[6]*c[1]))
        if abs(command[i]-bounded) >= 1e-9:
            return False
    future = state.copy(); tail = command.copy()
    if not advance_one(future, command, c):
        return False
    for _ in range(int(c[10])):
        for i in range(3):
            tail[i] = max(tail[i]-c[6]*c[1], -c[4])
        if not advance_one(future, tail, c):
            return False
        if future[0, 1] <= 1e-10 and future[1, 1] <= 1e-10 and future[2, 1] <= 1e-10:
            break
    return (future[0, 1] <= 1e-10 and future[1, 1] <= 1e-10 and future[2, 1] <= 1e-10
            and tail[0] <= 0 and tail[1] <= 0 and tail[2] <= 0)


@njit(cache=True, fastmath=False)
def optimistic_cost(state, previous, finish, arrived, step, c):
    """Exact earliest independent arrival ignoring all inter-vehicle constraints.

    Monotonicity of saturated forward dynamics and the slew envelope implies
    every admissible continuation has arrival time at least this value.
    """
    future = state.copy(); prev = previous.copy(); times = finish.copy(); done = arrived.copy()
    for k in range(step, round(c[2]/c[1])):
        for i in range(3):
            prev[i] = min(c[5], prev[i]+c[6]*c[1])
        for _ in range(round(c[1]/c[0])):
            for i in range(3):
                future[i, 1] = clip(future[i, 1]+prev[i]*c[0], 0., c[3])
                future[i, 0] += future[i, 1]*c[0]
        for i in range(3):
            if not done[i] and future[i, 0] >= c[11]:
                times[i] = (k+1)*c[1]; done[i] = True
        if np.all(done):
            break
    return (times[0]+times[1]+times[2])/3.


@njit(cache=True, fastmath=False)
def rollout(initial, previous, c, weights, first, coast, incumbent, pruning):
    length = round(c[2]/c[1]); x = initial.copy(); prev = previous.copy()
    commands = np.zeros((length, 3)); states = np.zeros((length, 3, 2)); memories = np.zeros((length, 3))
    arrived = x[:, 0] >= c[11]; finish = np.full(3, c[2])
    for i in range(3):
        if arrived[i]:
            finish[i] = 0.
    checks = 0
    for step in range(length):
        if np.all(arrived):
            return (finish[0]+finish[1]+finish[2])/3., step, commands, states, memories, checks, False
        if pruning and optimistic_cost(x, prev, finish, arrived, step, c) > incumbent+1e-10:
            return np.inf, step, commands, states, memories, checks, True
        proposal = np.empty(3); backup = np.empty(3)
        for i in range(3):
            raw = 0. if step < coast and i != first and not arrived[i] else c[5]
            proposal[i] = clip(raw, max(-c[4], prev[i]-c[6]*c[1]), min(c[5], prev[i]+c[6]*c[1]))
            backup[i] = max(prev[i]-c[6]*c[1], -c[4])
        best_cost = np.inf; chosen = np.empty(3)
        for mode in range(8):
            candidate = np.empty(3)
            for i in range(3):
                candidate[i] = proposal[i] if (mode >> (2-i)) & 1 else backup[i]
            checks += 1
            if certificate(x, prev, candidate, c):
                score = 0.; total = 0.
                for i in range(3):
                    score += weights[i]*(candidate[i]-proposal[i])**2; total += candidate[i]
                score -= 1e-8*total
                if score < best_cost:
                    best_cost = score; chosen = candidate.copy()
        if not np.isfinite(best_cost):
            raise ValueError('No certified continuation in the frozen model')
        commands[step] = chosen; states[step] = x; memories[step] = prev
        if not advance_one(x, chosen, c):
            raise ValueError('Certified first step failed the route contract')
        prev = chosen
        for i in range(3):
            if not arrived[i] and x[i, 0] >= c[11]:
                finish[i] = (step+1)*c[1]; arrived[i] = True
    return (finish[0]+finish[1]+finish[2])/3., length, commands, states, memories, checks, False


@njit(cache=True, fastmath=False)
def search(initial, previous, c, weights, first, coast, order, pruning):
    n = len(order); costs = np.full(n, np.nan); checks = np.zeros(n, dtype=np.int64)
    pruned = np.zeros(n, dtype=np.bool_); best = np.inf; selected = 0
    length = round(c[2]/c[1]); best_length = 0
    best_commands = np.zeros((length, 3)); best_states = np.zeros((length, 3, 2)); best_memories = np.zeros((length, 3))
    for index in order:
        cost, steps, commands, states, memories, count, stopped = rollout(initial, previous, c, weights[index], first[index], coast[index], best, pruning)
        costs[index] = cost; checks[index] = count; pruned[index] = stopped
        if cost < best or (cost == best and index < selected):
            best = cost; selected = index; best_length = steps
            best_commands = commands; best_states = states; best_memories = memories
    return costs, selected, best_length, best_commands, best_states, best_memories, checks, pruned


def plan_fast(state, previous, config=None, indices=None, order=None, pruning=False):
    start = time.perf_counter(); config = config or MergeConfig(); c = parameters(config)
    x = np.ascontiguousarray(state, dtype=np.float64); prev = np.ascontiguousarray(previous, dtype=np.float64)
    if x.shape != (3, 2) or prev.shape != (3,) or not np.isfinite(x).all() or not np.isfinite(prev).all():
        raise ValueError('Finite state (3,2) and memory (3,) required')
    if not np.isfinite(c).all() or min(config.dt, config.decision_dt, config.horizon, config.vmax, config.brake, config.accel, config.jerk) <= 0:
        raise ValueError('Positive finite dynamics parameters required')
    if abs(config.decision_dt/config.dt-round(config.decision_dt/config.dt)) > 1e-10 or abs(config.horizon/config.decision_dt-round(config.horizon/config.decision_dt)) > 1e-10:
        raise ValueError('Time scales must contain integer substeps')
    profiles = portfolio(); indices = list(range(37)) if indices is None else list(indices)
    if not indices or indices[0] != 0 or any(not isinstance(i, (int, np.integer)) or not 0 <= i < 37 for i in indices):
        raise ValueError('Candidate zero must be the incumbent; indices must be valid')
    chosen_profiles = [profiles[i] for i in indices]; n = len(indices)
    order = np.arange(n, dtype=np.int64) if order is None else np.asarray(order, dtype=np.int64)
    if order.shape != (n,) or sorted(order.tolist()) != list(range(n)) or order[0] != 0:
        raise ValueError('Order must be a permutation with incumbent first')
    if not certificate(x, prev, np.maximum(prev-config.jerk*config.decision_dt, -config.brake), c):
        raise ValueError('Initial state outside backup contract')
    weights = np.array([p['weights'] for p in chosen_profiles], dtype=np.float64)
    first = np.array([p['first'] for p in chosen_profiles], dtype=np.int64)
    coast = np.array([p['coast_steps'] for p in chosen_profiles], dtype=np.int64)
    costs, selected, steps, commands, states, memories, checks, pruned = search(x, prev, c, weights, first, coast, order, pruning)
    elapsed = (time.perf_counter()-start)*1000
    plan = Plan(commands[:steps], states[:steps], memories[:steps], costs, chosen_profiles, int(selected), elapsed,
                np.full(n, np.nan), 'compiled_bound' if pruning else 'compiled_full')
    return plan, {'certificate_calls': int(checks.sum()), 'pruned_profiles': int(pruned.sum()),
                  'profile_calls': checks.tolist(), 'pruned': pruned.tolist(), 'order': order.tolist()}


def learned_order(models, state):
    """Same ensemble as V6; rank all proposals, retaining baseline first."""
    x = torch.as_tensor(features(np.asarray(state)[None]))
    with torch.no_grad():
        probabilities = torch.stack([model(x)[0].softmax(-1) for model in models]).mean(0)[0].numpy()
    return np.array([0, *[int(i) for i in np.argsort(-probabilities, kind='stable') if i != 0]], dtype=np.int64)


class FastPlannedGate:
    """Same cached-plan checks and supported-state fallback, compiled witnesses."""
    def __init__(self, config, plan, state_tolerance=1e-6):
        self.c, self.plan, self.packed = config, plan, parameters(config)
        self.state_tolerance = state_tolerance
        self.step = 0; self.invalidated = False; self.fallback_count = 0
        self.fallback = MergeGate(config, 'constant')

    def select(self, state, previous, proposal):
        start = time.perf_counter(); k = self.step; self.step += 1
        state = np.ascontiguousarray(state, dtype=np.float64)
        previous = np.ascontiguousarray(previous, dtype=np.float64)
        if state.shape != (3, 2) or previous.shape != (3,) or not np.isfinite(state).all() or not np.isfinite(previous).all():
            raise ValueError('Finite measured state and memory required')
        backup_command = np.maximum(previous-self.c.jerk*self.c.decision_dt, -self.c.brake)
        in_set = bool(certificate(state, previous, backup_command, self.packed))
        if not in_set:
            raise RuntimeError('Measured state outside supported backup contract')
        matched = k < len(self.plan.commands)
        if matched:
            matched = (np.max(np.abs(state-self.plan.states[k])) <= self.state_tolerance
                       and np.max(np.abs(previous-self.plan.previous[k])) <= self.state_tolerance)
        self.invalidated |= not matched
        command = None if self.invalidated else self.plan.commands[k]
        if command is not None:
            self.invalidated |= not bool(certificate(state, previous, command, self.packed))
        if self.invalidated:
            self.fallback_count += 1; command, info = self.fallback.select(state, previous, proposal)
        else:
            cruise = admissible(np.full(3, self.c.accel), previous, self.c)
            info = {'in_set': in_set, 'certified': True,
                    'intervention': float(np.mean(np.abs(command-cruise) > 1e-5))}
        info.update(plan_fallback=self.invalidated, gate_ms=(time.perf_counter()-start)*1000)
        return command, info
