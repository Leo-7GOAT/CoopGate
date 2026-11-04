"""Frozen method adapters; no evaluation costs passed into online decisions."""
import numpy as np
from .fast_v8 import FastPlannedGate, learned_order, plan_fast
from .learning_v6 import ProfileGate, predict
from .planning_v5 import PlannedGate, plan_episode
from .verify_v7 import verify_proposal


def prepare(method, models, state, previous, config):
    info = {'proposal': None, 'certificate_calls': None, 'pruned_profiles': None}
    if method == 'constant':
        return 0, ProfileGate(config, 0), info
    if method == 'reference_pair':
        selected, plan, details = verify_proposal(models, state, previous, config)
        info['proposal'] = details['proposal']
        return selected, PlannedGate(config, plan), info
    if method == 'reference_full':
        plan = plan_episode(state, previous, config)
        return plan.selected, PlannedGate(config, plan), info
    if method == 'fast_pair':
        action, _ = predict(models, state[None]); proposal = int(action[0])
        plan, stats = plan_fast(state, previous, config, indices=[0, proposal])
        selected = proposal if plan.selected else 0; info['proposal'] = proposal
    elif method in ('fast_full', 'bb_natural', 'bb_learned'):
        order = learned_order(models, state) if method == 'bb_learned' else None
        plan, stats = plan_fast(state, previous, config, order=order, pruning=method != 'fast_full')
        selected = plan.selected
    else:
        raise ValueError(method)
    info.update(certificate_calls=stats['certificate_calls'], pruned_profiles=stats['pruned_profiles'])
    return selected, FastPlannedGate(config, plan), info
