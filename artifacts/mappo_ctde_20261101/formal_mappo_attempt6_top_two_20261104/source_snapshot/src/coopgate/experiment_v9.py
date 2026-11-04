"""Seven matched controllers; no disturbance tape is visible to a controller."""
import numpy as np
from .fast_v8 import plan_fast, FastPlannedGate
from .learning_v6 import predict
from .robust_v9 import RobustGate

METHODS = ('v8_cached', 'nominal_feedback', 'robust_constant', 'robust_fixed_pair',
           'robust_random_pair', 'robust_learned_pair', 'robust_full')


def prepare(method, models, state, previous, config, radius, seed):
    if method not in METHODS: raise ValueError(method)
    proposal = -1
    if method == 'v8_cached':
        plan, _ = plan_fast(state, previous, config, pruning=True)
        gate = FastPlannedGate(config, plan); selected = plan.selected; predicted = plan.predicted_cost
    else:
        strategy = 'full'
        if method == 'robust_constant': strategy = 'constant'
        if method.endswith('_pair'):
            strategy = 'pair'
            if method == 'robust_fixed_pair': proposal = 1  # V6 training-only best fixed profile.
            if method == 'robust_random_pair': proposal = int(np.random.default_rng(seed+902).integers(1, 37))
            if method == 'robust_learned_pair': proposal = int(predict(models, state[None])[0][0])
        gate = RobustGate(state, previous, config, radius=0. if method == 'nominal_feedback' else radius,
                          strategy=strategy, proposal=max(0, proposal))
        selected, predicted = gate.selected, gate.predicted_cost
    return gate, {'selected': int(selected), 'proposal': proposal, 'nominal_predicted_cost_s': float(predicted)}
