"""Certificate-constrained coordinate rollout with an exact base-policy tail.

The learned profile supplies a base policy, not a safety probability. Every
replacement command retains the V9 robust backup certificate. Integer arrival
ticks and an admissible lower bound make early stopping independent of floats.
"""
import time
import numpy as np
from numba import njit
from .fast_v8 import advance_one
from .robust_v9 import RobustGate, feedback, robust_certificate, validate
from .merge_v4 import admissible
from .planning_v5 import portfolio


@njit(cache=True, fastmath=False)
def active_count(state, c):
    count = 0
    for i in range(3):
        count += state[i, 0] < c[11]
    return count


@njit(cache=True, fastmath=False)
def earliest_ticks(state, previous, step, c):
    """Sum of independent earliest remaining arrival ticks, deadline capped."""
    x, prev = state.copy(), previous.copy()
    ticks = 0
    for _ in range(step, round(c[2]/c[1])):
        count = active_count(x, c)
        if count == 0:
            break
        ticks += count
        for i in range(3):
            prev[i] = min(c[5], prev[i]+c[6]*c[1])
        for _ in range(round(c[1]/c[0])):
            for i in range(3):
                x[i, 1] = min(c[3], max(0., x[i, 1]+prev[i]*c[0]))
                x[i, 0] += x[i, 1]*c[0]
    return ticks


@njit(cache=True, fastmath=False)
def tail_ticks(state, previous, step, c, radius, weights, first, coast,
               cutoff, pruning):
    """Exact remaining base cost, or a rejected lower bound >= cutoff."""
    x, prev = state.copy(), previous.copy()
    ticks, evaluations = 0, 0
    for k in range(step, round(c[2]/c[1])):
        count = active_count(x, c)
        if count == 0:
            break
        if pruning and ticks+earliest_ticks(x, prev, k, c) >= cutoff:
            return cutoff, evaluations, True
        command = feedback(x, prev, c, radius, weights, first, coast, k)
        evaluations += 1
        ticks += count
        if not advance_one(x, command, c):
            raise RuntimeError('Certified base tail failed at nominal center')
        prev = command
    return ticks, evaluations, False


@njit(cache=True, fastmath=False)
def improve(state, previous, step, c, radius, weights, first, coast,
            levels, order, pruning=True, passes=1, known_base=-1):
    """One sequential coordinate sweep; incumbent retained on every tie.

    At most 3*levels*passes trial commands. No global grid optimality claim.
    Each trial is certified jointly before its nominal cost is considered.
    """
    chosen = feedback(state, previous, c, radius, weights, first, coast, step)
    base, evaluations = known_base, 0
    if base < 0:
        base, evaluations, _ = tail_ticks(state, previous, step, c, radius,
                                        weights, first, coast, 10**9, False)
    best = base
    checks, rejected, pruned, accepted = 0, 0, 0, 0
    stage = active_count(state, c)
    if stage == 0:
        return chosen, base, best, evaluations, checks, rejected, pruned, accepted
    for index in range(passes*len(order)):
        i = order[index % len(order)]
        if state[i, 0] >= c[11]:
            continue  # Preserve the base controller's post-goal lifecycle.
        lower = max(-c[4], previous[i]-c[6]*c[1])
        upper = min(c[5], previous[i]+c[6]*c[1])
        # High acceleration first is only a heuristic ordering, not pruning.
        for j in range(levels-1, -1, -1):
            candidate = chosen.copy()
            candidate[i] = lower+(upper-lower)*j/(levels-1)
            if abs(candidate[i]-chosen[i]) < 1e-12:
                continue
            checks += 1
            if not robust_certificate(state, previous, candidate, c, radius):
                rejected += 1
                continue
            x = state.copy()
            if not advance_one(x, candidate, c):
                raise RuntimeError('Certified trial failed at nominal center')
            ticks, used, stopped = tail_ticks(x, candidate, step+1, c, radius,
                                            weights, first, coast, best-stage, pruning)
            evaluations += used
            pruned += stopped
            score = stage+ticks
            if not stopped and score < best:
                chosen, best = candidate, score
                accepted += 1
    return chosen, base, best, evaluations, checks, rejected, pruned, accepted


class RolloutGate(RobustGate):
    """Recompare base policies, then improve the best certified continuation."""
    def __init__(self, initial, previous, config=None, radius=.5,
                 strategy='pair', proposal=0, levels=5, pruning=True,
                 adaptive=True, passes=1):
        if not isinstance(levels, (int, np.integer)) or levels < 2:
            raise ValueError('At least two action levels required')
        if not isinstance(passes, (int, np.integer)) or passes < 1:
            raise ValueError('Positive integer coordinate passes required')
        if strategy not in ('constant', 'pair'):
            raise ValueError('Rollout supports constant or paired base policies')
        super().__init__(initial, previous, config, radius, strategy, proposal)
        self.levels, self.pruning, self.passes = levels, pruning, passes
        indices = ([0, proposal] if proposal else [0]) if strategy == 'pair' else [0]
        self.profiles = [(i, portfolio()[i]) for i in indices] if adaptive else [(self.selected, self.profile)]

    def select(self, state, previous, proposal):
        start = time.perf_counter()
        x, prev, _ = validate(state, previous, self.c, self.radius)
        backup = np.maximum(prev-self.c.jerk*self.c.decision_dt, -self.c.brake)
        in_set = bool(robust_certificate(x, prev, backup, self.packed, self.radius))
        if not in_set:
            raise RuntimeError('Measured state outside robust backup contract')
        if self.step >= round(self.c.horizon/self.c.decision_dt):
            raise ValueError('Rollout horizon exhausted')
        base_ticks, base_used = 10**9, 0
        chosen_profile, active_profile = self.profile, self.selected
        for i, p in self.profiles:
            ticks, used, stopped = tail_ticks(x, prev, self.step, self.packed,
                self.radius, np.asarray(p['weights'], dtype=np.float64), p['first'],
                p['coast_steps'], base_ticks, self.pruning)
            base_used += used
            if not stopped and ticks < base_ticks:
                base_ticks, chosen_profile, active_profile = ticks, p, i
        weights = np.asarray(chosen_profile['weights'], dtype=np.float64)
        order = np.argsort(-weights, kind='stable').astype(np.int64)
        command, base, best, evaluations, checks, rejected, pruned, accepted = improve(
            x, prev, self.step, self.packed, self.radius, weights,
            chosen_profile['first'], chosen_profile['coast_steps'], self.levels,
            order, self.pruning, self.passes, base_ticks)
        self.step += 1
        cruise = admissible(np.full(3, self.c.accel), prev, self.c)
        return command, {'in_set': in_set, 'certified': True,
            'intervention': float(np.mean(np.abs(command-cruise) > 1e-5)),
            'gate_ms': (time.perf_counter()-start)*1000, 'plan_fallback': False,
            'base_ticks': int(base), 'selected_ticks': int(best),
            'tail_feedback_steps': int(evaluations+base_used), 'trial_checks': int(checks),
            'unsafe_trials': int(rejected), 'pruned_trials': int(pruned),
            'accepted_updates': int(accepted), 'active_profile': int(active_profile)}
