"""Existing physical-switching rules adapted to the common driving certificate.

MAMPS Algorithm 1 (Zhang, Bastani, Kumar, arXiv:1910.12639v2) motivates the
monotone per-agent mask update. This is a common-baseline/common-oracle adapter,
not the original robot benchmark or its deterministic IsRecoverable routine.
It makes no claim about individual performance relative to the full baseline.
"""
import time

import numpy as np
from numba import njit

from .ego_v13 import ego_observations
from .fast_v8 import clip, parameters
from .joint_planner_v15 import JointRolloutGate
from .marl_v12 import LEVELS, action_mask
from .merge_v4 import MergeConfig
from .robust_v9 import interval_substep, occupancy, robust_certificate, validate


@njit(cache=True, fastmath=False)
def interval_transition_flags(lo, hi, qlo, qhi, c):
    """Assign every violated pairwise check to both incident vehicles."""
    valid = np.ones(3, np.bool_)
    if min(lo[0, 0]-hi[2, 0], qlo[0, 0]-qhi[2, 0]) < c[7]-1e-8:
        valid[0] = valid[2] = False
    for i, j in ((0, 1), (1, 2)):
        a, b, present_i = occupancy(lo[i, 0], hi[i, 0], qlo[i, 0], qhi[i, 0], c)
        e, f, present_j = occupancy(lo[j, 0], hi[j, 0], qlo[j, 0], qhi[j, 0], c)
        if present_i and present_j and max(a, e) <= min(b, f):
            valid[i] = valid[j] = False
        if qhi[i, 0] >= c[8] and qhi[j, 0] >= c[8]:
            ti = 0. if hi[i, 0] >= c[8] else (c[8]-hi[i, 0])/max(qhi[i, 0]-hi[i, 0], 1e-300)
            tj = 0. if hi[j, 0] >= c[8] else (c[8]-hi[j, 0])/max(qhi[j, 0]-hi[j, 0], 1e-300)
            t = max(ti, tj)
            low_start = lo[i, 0]+t*(qlo[i, 0]-lo[i, 0])-hi[j, 0]-t*(qhi[j, 0]-hi[j, 0])
            high_start = hi[i, 0]+t*(qhi[i, 0]-hi[i, 0])-lo[j, 0]-t*(qlo[j, 0]-lo[j, 0])
            ahead = min(low_start, qlo[i, 0]-qhi[j, 0]) >= c[7]-1e-8
            behind = max(high_start, qhi[i, 0]-qlo[j, 0]) <= -c[7]+1e-8
            if not (ahead or behind):
                valid[i] = valid[j] = False
    return valid


@njit(cache=True, fastmath=False)
def certificate_flags(state, previous, command, c, radius):
    """The conjunction equals the existing global robust certificate.

Unlike original MAMPS Algorithm 2, a vehicle's flag does not latch upon an
earlier individual stop. All incident pair checks persist to the common
stopped terminal set. No physical constraint or backup horizon is weakened.
"""
    valid = np.ones(3, np.bool_)
    for i in range(3):
        bounded = clip(command[i], max(-c[4], previous[i]-c[6]*c[1]), min(c[5], previous[i]+c[6]*c[1]))
        if abs(command[i]-bounded) >= 1e-9:
            valid[i] = False
    lo, hi, current = state.copy(), state.copy(), command.copy()
    for decision in range(int(c[10])+1):
        if decision:
            for i in range(3):
                current[i] = max(current[i]-c[6]*c[1], -c[4])
        for _ in range(round(c[1]/c[0])):
            qlo, qhi = interval_substep(lo, hi, current, c, radius)
            valid &= interval_transition_flags(lo, hi, qlo, qhi, c)
            lo, hi = qlo, qhi
        if decision and np.all(hi[:, 1] == 0.) and np.all(current <= -radius):
            return valid
    # The existing global certificate rejects when the joint terminal set was
    # not reached within the declared horizon. Retain that same joint condition.
    return np.zeros(3, np.bool_)


def recovery_search(proposal, baseline, checker, rule):
    """Switch only 1->0 within a call; recheck the final joint command."""
    proposal, baseline = np.asarray(proposal), np.asarray(baseline)
    if (proposal.shape != (3,) or baseline.shape != (3,) or
            not np.isfinite(proposal).all() or not np.isfinite(baseline).all()):
        raise ValueError("Three finite joint commands required")
    if rule not in ("mamps_shared", "mps_shared"):
        raise ValueError("Unknown physical switching rule")
    mask = np.ones(3, bool)
    masks, outcomes = [], []
    for _ in range(5):
        command = np.where(mask, proposal, baseline)
        flags = np.asarray(checker(command))
        if flags.shape != (3,) or flags.dtype != np.dtype(bool):
            raise ValueError("Checker must return three explicit Boolean flags")
        masks.append(mask.tolist())
        outcomes.append(flags.tolist())
        if flags.all():
            return command, mask.copy(), masks, outcomes
        if not mask.any():
            raise RuntimeError("Common baseline failed its physical certificate")
        following = mask & flags if rule == "mamps_shared" else np.zeros(3, bool)
        mask = np.zeros(3, bool) if np.array_equal(following, mask) else following
    raise RuntimeError("Monotone physical switching did not terminate")


class PhysicalSwitchGate:
    def __init__(self, model, rule="mamps_shared", config=None):
        if rule not in ("mamps_shared", "mps_shared"):
            raise ValueError("Unknown physical switching rule")
        self.model, self.rule = model, rule
        self.c = config or MergeConfig()
        self.packed = parameters(self.c)
        self.baseline = JointRolloutGate(self.c)
        self.step = 0
        self.learning_used_ever = False

    def select(self, state, previous, unused=None):
        started = time.perf_counter()
        x, prev, _ = validate(state, previous, self.c, .5)
        emergency = np.maximum(prev-self.c.jerk*self.c.decision_dt, -self.c.brake)
        if not robust_certificate(x, prev, emergency, self.packed, .5):
            raise RuntimeError("Physical switch outside common backup set")
        indices = self.model.proposal(ego_observations(x, prev, self.step, self.c), action_mask(prev, self.c))
        proposal = LEVELS[indices].copy()
        cleared = x[:, 0] >= self.c.goal
        proposal[cleared] = np.minimum(self.c.accel, prev[cleared]+self.c.jerk*self.c.decision_dt)
        self.baseline.step = self.step
        base, base_info = self.baseline.select(x, prev)
        checker = lambda command: certificate_flags(x, prev, command, self.packed, .5)
        command, mask, masks, flags = recovery_search(proposal, base, checker, self.rule)
        # Check the unchanged authoritative common certificate before execution.
        if not robust_certificate(x, prev, command, self.packed, .5):
            raise RuntimeError("Local and global physical checks disagree")
        used = bool(np.any((np.abs(command-base) > 1e-9) & ~cleared))
        self.learning_used_ever |= used
        self.step += 1
        return command, dict(in_set=True, certified=True,
            intervention=float(np.mean(np.abs(command-proposal)>1e-8)),
            gate_ms=1000*(time.perf_counter()-started), agent_actions=indices.tolist(),
            proposal=proposal.tolist(), common_baseline=base.tolist(), physical_switch_rule=self.rule,
            learning_mode_mask=mask.tolist(), mode_search_masks=masks, mode_search_flags=flags,
            physical_candidate_checks=len(masks), actual_learning_used=used,
            actual_learning_used_ever=self.learning_used_ever,
            baseline_model_transitions=base_info["performance_model_transitions"],
            individual_performance_checked=False, conditional_performance_risk_certified=False)
