"""Independent engineering supplement: verify the proposal against incumbent.

This is online two-profile model planning, not a probabilistic confidence gate.
The nondegradation property requires exact cost prediction and matching state.
"""
import numpy as np
from .learning_v6 import predict
from .planning_v5 import plan_episode, portfolio


def verify_proposal(models, state, previous, config):
    action, estimate = predict(models, np.asarray(state)[None])
    proposal = int(action[0]); profiles = portfolio()
    plan = plan_episode(state, previous, config, profiles=[profiles[0], profiles[proposal]])
    # Ties go to the incumbent through the frozen planner's argmin rule.
    selected = proposal if plan.selected == 1 else 0
    return selected, plan, {'proposal': proposal, 'estimated_excess_s': float(estimate[0]),
                            'verified_excess_s': float(plan.costs[1]-plan.costs[0])}
