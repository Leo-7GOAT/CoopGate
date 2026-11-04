"""Certified lexicographic rollout on a fixed command lattice.

Arrival ticks are primary and command total variation is secondary. Binary and
hold-enabled feedback families each retain their best profile and receive one
coordinate sweep. Neither performance objective relaxes the V9 certificate.
"""
import time
import numpy as np
from numba import njit
from .fast_v8 import advance_one, clip
from .robust_v9 import RobustGate, robust_certificate, validate
from .rollout_v10 import active_count, earliest_ticks
from .planning_v5 import portfolio
from .merge_v4 import admissible


@njit(cache=True, fastmath=False)
def binary_feedback_ordered(state, previous, c, radius, weights, first, coast, step):
    """Same eight-vertex minimizer and tie rule as frozen V9 feedback."""
    proposal, backup = np.empty(3), np.empty(3)
    for i in range(3):
        raw = 0. if step < coast and i != first and state[i, 0] < c[11] else c[5]
        proposal[i] = clip(raw,max(-c[4],previous[i]-c[6]*c[1]),min(c[5],previous[i]+c[6]*c[1]))
        backup[i] = max(previous[i]-c[6]*c[1],-c[4])
    commands, scores = np.empty((8,3)), np.empty(8)
    for mode in range(8):
        for i in range(3):
            commands[mode,i] = proposal[i] if (mode >> (2-i)) & 1 else backup[i]
        score,total=0.,0.
        for i in range(3):
            score+=weights[i]*(commands[mode,i]-proposal[i])**2
            total+=commands[mode,i]
        scores[mode]=score-1e-8*total
    for mode in np.argsort(scores,kind='mergesort'):
        if robust_certificate(state,previous,commands[mode],c,radius):
            return commands[mode].copy()
    raise ValueError('No robustly certified continuation')


@njit(cache=True, fastmath=False)
def ternary_feedback(state, previous, c, radius, weights, first, coast, step):
    """Joint proposal/hold/backup choices; all commands are certified."""
    proposal, backup = np.empty(3), np.empty(3)
    for i in range(3):
        raw = 0. if step < coast and i != first and state[i, 0] < c[11] else c[5]
        proposal[i] = clip(raw, max(-c[4], previous[i]-c[6]*c[1]), min(c[5], previous[i]+c[6]*c[1]))
        backup[i] = max(previous[i]-c[6]*c[1], -c[4])
    commands, scores = np.empty((27,3)), np.empty(27)
    for mode in range(27):
        command, digits = np.empty(3), mode
        for i in range(2, -1, -1):
            digit = digits % 3; digits //= 3
            command[i] = backup[i] if digit == 0 else previous[i] if digit == 1 else proposal[i]
        score = 0.
        for i in range(3):
            score += weights[i]*(command[i]-proposal[i])**2 - 1e-8*command[i]
        commands[mode], scores[mode] = command, score
    # Stable ascending score order exactly preserves the original tie rule.
    for mode in np.argsort(scores, kind='mergesort'):
        if robust_certificate(state, previous, commands[mode], c, radius):
            return commands[mode].copy()
    raise ValueError('No robustly certified continuation')


@njit(cache=True, fastmath=False)
def base_command(x, prev, k, c, radius, weights, first, coast, ternary):
    if ternary:
        return ternary_feedback(x, prev, c, radius, weights, first, coast, k)
    return binary_feedback_ordered(x, prev, c, radius, weights, first, coast, k)


@njit(cache=True, fastmath=False)
def variation_units(command, previous, quantum):
    units = 0
    for i in range(3):
        units += round(abs(command[i]-previous[i])/quantum)
    return units


@njit(cache=True, fastmath=False)
def less(ticks, variation, other_ticks, other_variation, smooth):
    return ticks < other_ticks or (smooth and ticks == other_ticks and variation < other_variation)


@njit(cache=True, fastmath=False)
def tail_value(state, previous, step, c, radius, weights, first, coast,
               quantum, ternary, smooth, cutoff_ticks, cutoff_variation, pruning):
    x, prev = state.copy(), previous.copy()
    ticks, variation, evaluations = 0, 0, 0
    for k in range(step, round(c[2]/c[1])):
        count = active_count(x, c)
        if count == 0:
            break
        if pruning:
            lower = ticks+earliest_ticks(x, prev, k, c)
            if not less(lower, variation, cutoff_ticks, cutoff_variation, smooth):
                return cutoff_ticks, cutoff_variation, evaluations, True
        command = base_command(x, prev, k, c, radius, weights, first, coast, ternary)
        evaluations += 1
        ticks += count
        variation += variation_units(command, prev, quantum)
        if not advance_one(x, command, c):
            raise RuntimeError('Certified base tail failed at nominal center')
        prev = command
    return ticks, variation, evaluations, False


@njit(cache=True, fastmath=False)
def improve_smooth(state, previous, step, c, radius, weights, first, coast,
                   quantum, ternary, smooth, order, base_ticks, base_variation, pruning):
    chosen = base_command(state, previous, step, c, radius, weights, first, coast, ternary)
    best_ticks, best_variation = base_ticks, base_variation
    evaluations, checks, rejected, pruned, accepted = 0, 0, 0, 0, 0
    stage = active_count(state, c)
    if not stage:
        return chosen, best_ticks, best_variation, evaluations, checks, rejected, pruned, accepted
    for i in order:
        if state[i, 0] >= c[11]:
            continue
        lo = round(max(-c[4], previous[i]-c[6]*c[1])/quantum)
        hi = round(min(c[5], previous[i]+c[6]*c[1])/quantum)
        for level in range(hi, lo-1, -1):
            candidate = chosen.copy(); candidate[i] = level*quantum
            if candidate[i] == chosen[i]:
                continue
            checks += 1
            if not robust_certificate(state, previous, candidate, c, radius):
                rejected += 1
                continue
            x = state.copy()
            if not advance_one(x, candidate, c):
                raise RuntimeError('Certified trial failed at nominal center')
            variation = variation_units(candidate, previous, quantum)
            ticks, tail_variation, used, stopped = tail_value(x, candidate, step+1, c, radius,
                weights, first, coast, quantum, ternary, smooth,
                best_ticks-stage, best_variation-variation, pruning)
            evaluations += used; pruned += stopped
            score_ticks, score_variation = stage+ticks, variation+tail_variation
            if not stopped and less(score_ticks, score_variation, best_ticks, best_variation, smooth):
                chosen, best_ticks, best_variation = candidate, score_ticks, score_variation
                accepted += 1
    return chosen, best_ticks, best_variation, evaluations, checks, rejected, pruned, accepted


class SmoothGate(RobustGate):
    """Jointly certify commands and improve an exact two-component tail value.

    The command quantum is 0.625 m/s^2 in the frozen experiment. All candidate
    commands, the memory and the physical slew increment share this lattice,
    making both components of the objective integer valued. No scalar comfort
    weight or online tuning from held-out disturbance tapes is used.
    """
    def __init__(self, initial, previous, config=None, radius=.5, strategy='pair',
                 proposal=0, quantum=.625, ternary=True, smooth=True, pruning=True,
                 retain_binary=True, family_rollouts=True):
        if strategy not in ('constant','pair'):
            raise ValueError('Constant or paired profiles required')
        super().__init__(initial, previous, config, radius, strategy, proposal)
        if not np.isfinite(quantum) or quantum <= 0:
            raise ValueError('Positive finite command quantum required')
        for v in [self.c.accel, self.c.brake, self.c.jerk*self.c.decision_dt, *previous]:
            if abs(v/quantum-round(v/quantum)) > 1e-9:
                raise ValueError('Dynamics and command memory must lie on command lattice')
        self.quantum, self.ternary, self.smooth, self.pruning = quantum, ternary, smooth, pruning
        self.family_rollouts = family_rollouts
        ids = [0, proposal] if strategy == 'pair' and proposal else [0]
        families = [False, True] if ternary and retain_binary else [ternary]
        self.profiles = [(i, portfolio()[i], family) for family in families for i in ids]

    def select(self, state, previous, proposal):
        start = time.perf_counter()
        x, prev, _ = validate(state, previous, self.c, self.radius)
        if np.any(np.abs(prev/self.quantum-np.rint(prev/self.quantum)) > 1e-9):
            raise ValueError('Command memory must lie on command lattice')
        backup = np.maximum(prev-self.c.jerk*self.c.decision_dt, -self.c.brake)
        if not robust_certificate(x, prev, backup, self.packed, self.radius):
            raise RuntimeError('Measured state outside robust backup contract')
        if self.step >= round(self.c.horizon/self.c.decision_dt):
            raise ValueError('Rollout horizon exhausted')
        base_ticks, base_variation, base_used = 10**9, 10**9, 0
        chosen_profile, active_profile, active_family = self.profile, self.selected, False
        leaders = {}
        for i, p, family in self.profiles:
            cutoff = leaders.get(family, (10**9,10**9,None,-1))[:2] if self.family_rollouts else (base_ticks,base_variation)
            ticks, variation, used, stopped = tail_value(x, prev, self.step, self.packed,
                self.radius, np.asarray(p['weights'],dtype=np.float64), p['first'], p['coast_steps'],
                self.quantum, family, self.smooth, *cutoff, self.pruning)
            base_used += used
            if not stopped and less(ticks, variation, *cutoff, self.smooth):
                leaders[family] = (ticks,variation,p,i)
            if not stopped and less(ticks, variation, base_ticks, base_variation, self.smooth):
                base_ticks, base_variation, chosen_profile, active_profile, active_family = ticks, variation, p, i, family
        if not self.family_rollouts:
            leaders = {active_family:(base_ticks,base_variation,chosen_profile,active_profile)}
        command = base_command(x,prev,self.step,self.packed,self.radius,
            np.asarray(chosen_profile['weights'],dtype=np.float64),chosen_profile['first'],chosen_profile['coast_steps'],active_family)
        ticks, variation = base_ticks, base_variation
        used, checks, rejected, pruned, accepted = 0,0,0,0,0
        for family,(local_t,local_v,p,profile_id) in leaders.items():
            weights=np.asarray(p['weights'],dtype=np.float64)
            order=np.argsort(-weights,kind='stable').astype(np.int64)
            u,t,v,e,ch,rj,pr,ac=improve_smooth(x,prev,self.step,self.packed,self.radius,
                weights,p['first'],p['coast_steps'],self.quantum,family,self.smooth,order,
                local_t,local_v,self.pruning)
            used+=e;checks+=ch;rejected+=rj;pruned+=pr;accepted+=ac
            if less(t,v,ticks,variation,self.smooth):
                command,ticks,variation,active_profile,active_family=u,t,v,profile_id,family
        self.step += 1
        cruise = admissible(np.full(3,self.c.accel),prev,self.c)
        return command, {'in_set':True, 'certified':True, 'intervention':float(np.mean(np.abs(command-cruise)>1e-5)),
            'gate_ms':(time.perf_counter()-start)*1000, 'plan_fallback':False,
            'base_ticks':int(base_ticks),'selected_ticks':int(ticks),
            'base_variation_units':int(base_variation),'selected_variation_units':int(variation),
            'tail_feedback_steps':int(used+base_used),'trial_checks':int(checks),
            'unsafe_trials':int(rejected),'pruned_trials':int(pruned),
            'accepted_updates':int(accepted),'active_profile':int(active_profile),
            'active_family':int(active_family),'family_sweeps':len(leaders)}
