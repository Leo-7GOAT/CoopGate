"""Interval backup certificates for bounded actuator error and feedback plans.

The disturbance bound is deterministic, not a learned confidence score. Route
contract only; exact measured state, fixed routes and cooperative vehicles.
"""
import time
import numpy as np
from numba import njit
from .fast_v8 import clip, parameters, advance_one, optimistic_cost
from .merge_v4 import MergeConfig, admissible
from .planning_v5 import portfolio


@njit(cache=True, fastmath=False)
def occupancy(pl, pu, ql, qu, c):
    """Outer interval of all possible conflict occupancy times in one substep."""
    if qu < c[8] or pl > c[9]:
        return 1., 0., False
    enter = 0. if pu >= c[8] else (c[8]-pu)/max(qu-pu, 1e-300)
    leave = 1. if ql <= c[9] else (c[9]-pl)/max(ql-pl, 1e-300)
    return max(0., enter), min(1., leave), True


@njit(cache=True, fastmath=False)
def interval_transition(lo, hi, qlo, qhi, c):
    if min(lo[0, 0]-hi[2, 0], qlo[0, 0]-qhi[2, 0]) < c[7]-1e-8:
        return False
    for i, j in ((0, 1), (1, 2)):
        a, b, present_i = occupancy(lo[i, 0], hi[i, 0], qlo[i, 0], qhi[i, 0], c)
        e, f, present_j = occupancy(lo[j, 0], hi[j, 0], qlo[j, 0], qhi[j, 0], c)
        if present_i and present_j and max(a, e) <= min(b, f):
            return False
        if qhi[i, 0] >= c[8] and qhi[j, 0] >= c[8]:
            # Both could have entered the guarded approach from this time on.
            ti = 0. if hi[i, 0] >= c[8] else (c[8]-hi[i, 0])/max(qhi[i, 0]-hi[i, 0], 1e-300)
            tj = 0. if hi[j, 0] >= c[8] else (c[8]-hi[j, 0])/max(qhi[j, 0]-hi[j, 0], 1e-300)
            t = max(ti, tj)
            low_start = lo[i, 0]+t*(qlo[i, 0]-lo[i, 0])-hi[j, 0]-t*(qhi[j, 0]-hi[j, 0])
            high_start = hi[i, 0]+t*(qhi[i, 0]-hi[i, 0])-lo[j, 0]-t*(qlo[j, 0]-lo[j, 0])
            ahead = min(low_start, qlo[i, 0]-qhi[j, 0]) >= c[7]-1e-8
            behind = max(high_start, qhi[i, 0]-qlo[j, 0]) <= -c[7]+1e-8
            if not (ahead or behind):
                return False
    return True


@njit(cache=True, fastmath=False)
def interval_substep(lo, hi, command, c, radius):
    qlo, qhi = lo.copy(), hi.copy()
    for i in range(3):
        qlo[i, 1] = clip(lo[i, 1]+(command[i]-radius)*c[0], 0., c[3])
        qhi[i, 1] = clip(hi[i, 1]+(command[i]+radius)*c[0], 0., c[3])
        qlo[i, 0] += qlo[i, 1]*c[0]; qhi[i, 0] += qhi[i, 1]*c[0]
    return qlo, qhi


@njit(cache=True, fastmath=False)
def interval_advance(lo, hi, command, c, radius):
    for _ in range(round(c[1]/c[0])):
        qlo, qhi = interval_substep(lo, hi, command, c, radius)
        if not interval_transition(lo, hi, qlo, qhi, c):
            return qlo, qhi, False
        lo, hi = qlo, qhi
    return lo, hi, True


@njit(cache=True, fastmath=False)
def robust_certificate(state, previous, command, c, radius):
    for i in range(3):
        bounded = clip(command[i], max(-c[4], previous[i]-c[6]*c[1]), min(c[5], previous[i]+c[6]*c[1]))
        if abs(command[i]-bounded) >= 1e-9:
            return False
    lo, hi, valid = interval_advance(state.copy(), state.copy(), command, c, radius)
    if not valid:
        return False
    tail = command.copy()
    for _ in range(int(c[10])):
        for i in range(3):
            tail[i] = max(tail[i]-c[6]*c[1], -c[4])
        lo, hi, valid = interval_advance(lo, hi, tail, c, radius)
        if not valid:
            return False
        if np.all(hi[:, 1] == 0.) and np.all(tail <= -radius):
            return True
    return False


@njit(cache=True, fastmath=False)
def feedback(state, previous, c, radius, weights, first, coast, step):
    proposal, backup = np.empty(3), np.empty(3)
    for i in range(3):
        raw = 0. if step < coast and i != first and state[i, 0] < c[11] else c[5]
        proposal[i] = clip(raw, max(-c[4], previous[i]-c[6]*c[1]), min(c[5], previous[i]+c[6]*c[1]))
        backup[i] = max(previous[i]-c[6]*c[1], -c[4])
    best, chosen = np.inf, backup.copy()
    for mode in range(8):
        command = np.empty(3)
        for i in range(3):
            command[i] = proposal[i] if (mode >> (2-i)) & 1 else backup[i]
        if robust_certificate(state, previous, command, c, radius):
            score, total = 0., 0.
            for i in range(3):
                score += weights[i]*(command[i]-proposal[i])**2; total += command[i]
            score -= 1e-8*total
            if score < best:
                best, chosen = score, command.copy()
    if not np.isfinite(best):
        raise ValueError('No robustly certified continuation')
    return chosen


@njit(cache=True, fastmath=False)
def profile_cost(initial, previous, c, radius, weights, first, coast, incumbent):
    state, prev = initial.copy(), previous.copy()
    arrived = state[:, 0] >= c[11]; finish = np.full(3, c[2])
    for i in range(3):
        if arrived[i]: finish[i] = 0.
    for step in range(round(c[2]/c[1])):
        if np.all(arrived): break
        if optimistic_cost(state, prev, finish, arrived, step, c) > incumbent+1e-10:
            return np.inf
        command = feedback(state, prev, c, radius, weights, first, coast, step)
        if not advance_one(state, command, c):
            raise RuntimeError('Robust certificate failed at its nominal center')
        prev = command
        for i in range(3):
            if not arrived[i] and state[i, 0] >= c[11]:
                finish[i] = (step+1)*c[1]; arrived[i] = True
    return (finish[0]+finish[1]+finish[2])/3.


def validate(state, previous, c, radius):
    x = np.ascontiguousarray(state, dtype=np.float64); prev = np.ascontiguousarray(previous, dtype=np.float64)
    if x.shape != (3, 2) or prev.shape != (3,) or not np.isfinite(x).all() or not np.isfinite(prev).all():
        raise ValueError('Finite state (3,2) and memory (3,) required')
    packed = parameters(c)
    if not np.isfinite(packed).all() or not np.isfinite(radius) or not 0 <= radius < c.brake:
        raise ValueError('Finite disturbance bound in [0, brake) required')
    if min(c.dt, c.decision_dt, c.horizon, c.vmax, c.brake, c.accel, c.jerk) <= 0 or c.tail_steps <= 0:
        raise ValueError('Positive dynamics and tail length required')
    if abs(c.decision_dt/c.dt-round(c.decision_dt/c.dt)) > 1e-10 or abs(c.horizon/c.decision_dt-round(c.horizon/c.decision_dt)) > 1e-10:
        raise ValueError('Integer time ratios required')
    if np.any(x[:, 1] < 0) or np.any(x[:, 1] > c.vmax) or np.any(prev < -c.brake) or np.any(prev > c.accel):
        raise ValueError('State or command memory outside physical bounds')
    return x, prev, packed


class RobustGate:
    """Select a feedback profile once; recertify it from every measured state."""
    def __init__(self, initial, previous, config=None, radius=.5, strategy='full', proposal=0):
        self.c = config or MergeConfig(); self.radius = radius; self.step = 0
        x, prev, self.packed = validate(initial, previous, self.c, radius)
        b = np.maximum(prev-self.c.jerk*self.c.decision_dt, -self.c.brake)
        if not robust_certificate(x, prev, b, self.packed, radius):
            raise ValueError('Initial state outside robust backup contract')
        if strategy not in ('full', 'constant', 'pair') or not isinstance(proposal, (int, np.integer)) or not 0 <= proposal < 37:
            raise ValueError('Invalid profile strategy or proposal')
        profiles = portfolio(); indices = range(37) if strategy == 'full' else ([0, proposal] if strategy == 'pair' and proposal else [0])
        self.costs = np.full(37, np.nan); self.selected = 0; best = np.inf
        for i in indices:
            p = profiles[i]
            cost = profile_cost(x, prev, self.packed, radius, np.array(p['weights']), p['first'], p['coast_steps'], best)
            self.costs[i] = cost
            if cost < best: self.selected, best = i, cost
        self.profile = profiles[self.selected]; self.predicted_cost = best

    def select(self, state, previous, proposal):
        start = time.perf_counter(); x, prev, _ = validate(state, previous, self.c, self.radius)
        b = np.maximum(prev-self.c.jerk*self.c.decision_dt, -self.c.brake)
        in_set = bool(robust_certificate(x, prev, b, self.packed, self.radius))
        if not in_set: raise RuntimeError('Measured state outside robust backup contract')
        p = self.profile
        command = feedback(x, prev, self.packed, self.radius, np.array(p['weights']), p['first'], p['coast_steps'], self.step)
        self.step += 1
        cruise = admissible(np.full(3, self.c.accel), prev, self.c)
        return command, {'in_set': in_set, 'certified': True, 'intervention': float(np.mean(np.abs(command-cruise) > 1e-5)),
                         'gate_ms': (time.perf_counter()-start)*1000, 'plan_fallback': False}
