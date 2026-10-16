"""Explicit component-wise switching ablation with the same physical shield.

A partial profile is a distinct closed-loop controller. Its outcomes cannot be
replaced by the matching columns of either complete controller's outcomes.
The neural proposals are local; the baseline and physical filter are joint.
"""
import time
import numpy as np
from .cooperative_v15 import JointPolicyGate
from .joint_planner_v15 import JointRolloutGate
from .ego_v13 import ego_observations
from .marl_v12 import LEVELS, action_mask, project_joint
from .merge_v4 import MergeConfig
from .fast_v8 import parameters
from .robust_v9 import robust_certificate, validate


class ComposedProfile:
    def __init__(self, model, learned_mask):
        mask = np.asarray(learned_mask)
        if mask.shape != (3,) or mask.dtype != np.dtype(bool):
            raise ValueError('Exactly three explicit boolean mode choices required')
        self.mask = mask.copy()
        self.model = model
        self.c = MergeConfig()
        self.packed = parameters(self.c)
        self.baseline = JointRolloutGate()
        self.learned = JointPolicyGate(model)
        self.step = 0

    def select(self, state, previous, unused=None):
        started = time.perf_counter()
        x, prev, _ = validate(state, previous, self.c, .5)
        self.baseline.step = self.learned.step = self.step
        if not self.mask.any():
            command, info = self.baseline.select(x, prev)
        elif self.mask.all():
            command, info = self.learned.select(x, prev)
        else:
            backup = np.maximum(prev-self.c.jerk*self.c.decision_dt, -self.c.brake)
            if not robust_certificate(x, prev, backup, self.packed, .5):
                raise RuntimeError('Mixed profile outside recoverable set')
            base, _ = self.baseline.select(x, prev)
            actions = self.model.proposal(ego_observations(x, prev, self.step), action_mask(prev))
            proposal = np.where(self.mask, LEVELS[actions], base)
            cleared = x[:, 0] >= self.c.goal
            proposal[cleared] = np.minimum(self.c.accel, prev[cleared]+self.c.jerk*self.c.decision_dt)
            command, vertex, valid = project_joint(x, prev, proposal, self.packed, .5)
            if not valid:
                raise RuntimeError('No physically certified mixed command')
            info = dict(in_set=True, certified=True, proposal=proposal.tolist(),
                        intervention=float(np.mean(np.abs(command-proposal)>1e-8)), shield_vertex=vertex)
        self.step += 1
        info = dict(info, component_learning_mask=self.mask.tolist(),
                    gate_ms=(time.perf_counter()-started)*1000.)
        return command, info


def nominal_independent_decision(predicted_isolated_difference, minimum_own_gain=.05):
    """Naive ablation: each row assumes the other two cars use the baseline.

    This is a deterministic nominal performance test, not a risk certificate.
    Joint execution can invalidate the other-policy assumption in every row.
    """
    values = np.asarray(predicted_isolated_difference, dtype=float)
    if values.shape != (3, 2) or not np.isfinite(values).all():
        raise ValueError('Three finite isolated cost/time differences required')
    if not np.isfinite(minimum_own_gain) or minimum_own_gain < 0:
        raise ValueError('Invalid gain threshold')
    return (values[:, 0] <= -minimum_own_gain) & (values[:, 1] <= 0.)
