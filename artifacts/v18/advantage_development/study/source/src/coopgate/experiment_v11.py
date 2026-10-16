"""Frozen matched V11 comparison and mechanism/learning ablations."""
import numpy as np
from .experiment_v10 import prepare as old_prepare
from .learning_v6 import predict
from .smooth_v11 import SmoothGate

METHODS=('v9_learned_pair','v10_rollout','smooth_constant','smooth_fixed_pair',
    'smooth_random_pair','smooth_binary','smooth_time_only','smooth_single_family',
    'smooth_learned')


def prepare(method,models,state,previous,config,radius,seed):
    if method not in METHODS:
        raise ValueError(method)
    if method in ('v9_learned_pair','v10_rollout'):
        gate,info=old_prepare('rollout_learned' if method=='v10_rollout' else method,
            models,state,previous,config,radius,seed)
        info['initial_v9_base_cost_s']=info.pop('initial_base_cost_s')
        return gate,info
    proposal=0
    if method=='smooth_fixed_pair':
        proposal=1
    elif method=='smooth_random_pair':
        proposal=int(np.random.default_rng(seed+1002).integers(1,37))
    elif method!='smooth_constant':
        proposal=int(predict(models,state[None])[0][0])
    gate=SmoothGate(state,previous,config,radius,
        'constant' if method=='smooth_constant' else 'pair',proposal,
        quantum=.625,ternary=method!='smooth_binary',smooth=method!='smooth_time_only',
        pruning=True,retain_binary=True,family_rollouts=method!='smooth_single_family')
    return gate,{'selected':int(gate.selected),'proposal':proposal,
        'initial_v9_base_cost_s':float(gate.predicted_cost)}
