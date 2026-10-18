"""Skip duplicate current commands in the unchanged strong rollout objective.

Within one joint_rollout call, state/previous/step and rule continuation are
fixed. The incumbent objective only decreases. A command already evaluated
cannot later improve that incumbent: a full evaluation already competed, and
an early-pruned tail has a lower bound at least as large as the then-incumbent.
The original command ordering, arithmetic, 1e-10 comparison and physical
certificate are preserved. Only the informational work counter changes.
"""
import numpy as np
from numba import njit

from .fast_v8 import advance_one
from .joint_planner_v15 import team_rule_tail
from .marl_v12 import LEVELS, project_joint
from .single_v12 import constant_proposal


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
