"""Experimental exact baseline kernel with reusable interval-state buffers.

Only allocation changes in the physical certificate. Every arithmetic update,
transition predicate, jerk bound, backup step and stopping condition is kept.
The three downstream functions below copy their frozen original bodies exactly;
their local project_joint reference resolves to this same certificate rewrite.
This module is not used by the prospective calibration controller.
"""
import numpy as np
from numba import njit

from .fast_v8 import advance_one, clip
from .marl_v12 import LEVELS
from .robust_v9 import interval_transition
from .single_v12 import constant_proposal


@njit(cache=True, fastmath=False)
def robust_certificate(state, previous, command, c, radius):
    for i in range(3):
        bounded = clip(command[i], max(-c[4], previous[i]-c[6]*c[1]), min(c[5], previous[i]+c[6]*c[1]))
        if abs(command[i]-bounded) >= 1e-9:
            return False
    lo, hi = state.copy(), state.copy()
    qlo, qhi = np.empty_like(lo), np.empty_like(hi)
    tail = command.copy()
    # Stage zero is the proposed decision; subsequent stages are the exact
    # original backup. Test stopping only after a full backup decision.
    for stage in range(1+int(c[10])):
        if stage > 0:
            for i in range(3):
                tail[i] = max(tail[i]-c[6]*c[1], -c[4])
        for _ in range(round(c[1]/c[0])):
            for i in range(3):
                qlo[i, 1] = clip(lo[i, 1]+(tail[i]-radius)*c[0], 0., c[3])
                qhi[i, 1] = clip(hi[i, 1]+(tail[i]+radius)*c[0], 0., c[3])
                qlo[i, 0] = lo[i, 0]+qlo[i, 1]*c[0]
                qhi[i, 0] = hi[i, 0]+qhi[i, 1]*c[0]
            if not interval_transition(lo, hi, qlo, qhi, c):
                return False
            lo, qlo = qlo, lo
            hi, qhi = qhi, hi
        if stage > 0 and np.all(hi[:, 1] == 0.) and np.all(tail <= -radius):
            return True
    return False


@njit(cache=True, fastmath=False)
def project_joint(x, previous, proposed, c, radius):
    """Nearest proposed/hold/backup joint vertex; no rollout optimizer."""
    commands = np.empty((27,3)); scores = np.empty(27)
    for mode in range(27):
        code = mode; score = 0.
        for i in range(2,-1,-1):
            option = code % 3; code //= 3
            backup = max(previous[i]-c[6]*c[1],-c[4])
            v = backup if option == 0 else (previous[i] if option == 1 else proposed[i])
            commands[mode,i] = v
            score += (v-proposed[i])**2
        scores[mode] = score
    order = np.argsort(scores,kind='mergesort')
    for m in order:
        if robust_certificate(x,previous,commands[m],c,radius):
            return commands[m].copy(), int(m), True
    return commands[0].copy(), 0, False


@njit(cache=True,fastmath=False)
def team_rule_tail(state,previous,step,c,cutoff=1e12):
    x=state.copy();prev=previous.copy();value=0.;transitions=0
    for k in range(step,round(c[2]/c[1])):
        active=x[:,0]<c[11]
        if not np.any(active):break
        command,_,ok=project_joint(x,prev,constant_proposal(prev,c),c,.5)
        if not ok:raise RuntimeError('Rule continuation outside physical contract')
        for i in range(3):
            if active[i]:value+=(c[1]+.05*abs(command[i]-prev[i]))/3.
        if value>=cutoff:return value,transitions
        if not advance_one(x,command,c):raise RuntimeError('Unsafe rule continuation')
        transitions+=1;prev=command
    return value,transitions


@njit(cache=True, fastmath=False)
def deduplicated_joint_rollout(state, previous, step, c, sweeps=2):
    chosen, _, ok = project_joint(state, previous, constant_proposal(previous, c), c, .5)
    if not ok:
        raise RuntimeError('No certified baseline command')
    best, transitions = team_rule_tail(state, previous, step, c)
    base_value = best
    if np.all(state[:, 0] >= c[11]):
        return chosen, best, base_value, transitions
    seen = np.empty((1 + max(0, sweeps) * 3 * len(LEVELS), 3))
    n_seen = 0
    for _ in range(sweeps):
        improved = False
        for i in (1, 0, 2):
            if state[i, 0] >= c[11]:
                continue
            center = chosen.copy()
            for level in LEVELS:
                if abs(level - previous[i]) > c[6] * c[1] + 1e-8:
                    continue
                proposal = center.copy()
                proposal[i] = level
                for j in range(3):
                    if state[j, 0] >= c[11]:
                        proposal[j] = min(c[5], previous[j] + c[6] * c[1])
                command, _, valid = project_joint(state, previous, proposal, c, .5)
                if not valid:
                    raise RuntimeError('No certificate for candidate')
                if np.array_equal(command, chosen):
                    continue
                duplicate = False
                for k in range(n_seen):
                    if np.array_equal(command, seen[k]):
                        duplicate = True
                        break
                if duplicate:
                    continue
                seen[n_seen] = command
                n_seen += 1
                stage = 0.
                for j in range(3):
                    if state[j, 0] < c[11]:
                        stage += (c[1] + .05 * abs(command[j] - previous[j])) / 3.
                if stage >= best:
                    continue
                following = state.copy()
                if not advance_one(following, command, c):
                    raise RuntimeError('Unsafe planning transition')
                tail, count = team_rule_tail(following, command, step + 1, c, best - stage)
                transitions += 1 + count
                if stage + tail < best - 1e-10:
                    chosen = command.copy()
                    best = stage + tail
                    improved = True
        if not improved:
            break
    return chosen, best, base_value, transitions
