"""Matched old controllers and coordinate-rollout ablations."""
import numpy as np
from .learning_v6 import predict
from .robust_v9 import RobustGate
from .rollout_v10 import RolloutGate

METHODS = ('v9_constant', 'v9_learned_pair', 'v9_full', 'rollout_constant',
           'rollout_fixed_pair', 'rollout_random_pair', 'rollout_learned_binary',
           'rollout_learned')


def prepare(method, models, state, previous, config, radius, seed):
    if method not in METHODS: raise ValueError(method)
    proposal = 0
    if 'learned' in method: proposal = int(predict(models, state[None])[0][0])
    if method == 'rollout_fixed_pair': proposal = 1
    if method == 'rollout_random_pair': proposal = int(np.random.default_rng(seed+1002).integers(1,37))
    strategy = 'full' if method == 'v9_full' else 'constant' if method.endswith('constant') else 'pair'
    cls = RolloutGate if method.startswith('rollout') else RobustGate
    options = {'levels': 2 if method.endswith('binary') else 5, 'passes': 1, 'adaptive': True} if cls == RolloutGate else {}
    gate = cls(state, previous, config, radius, strategy, proposal, **options)
    return gate, {'selected': int(gate.selected), 'proposal': proposal,
                  'initial_base_cost_s': float(gate.predicted_cost)}
