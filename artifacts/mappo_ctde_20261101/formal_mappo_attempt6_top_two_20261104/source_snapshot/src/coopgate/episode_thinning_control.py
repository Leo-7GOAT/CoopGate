"""State-independent episode abstention control, fixed before confirmation.

The externally supplied coin is controller randomness,not an environment seed
or an online confidence estimate. Actual adoption is inherited from executed
commands; choosing the nominal branch alone is never counted as adoption.
"""
import numpy as np


def matching_probability(target_adopted,nominal_adopted,n):
    if any(type(x) is not int for x in (target_adopted,nominal_adopted,n)) or not 0<=target_adopted<=nominal_adopted<=n or n<=0:
        raise ValueError('Complete common calibration counts,with target no larger than nominal adoption,required')
    if nominal_adopted==0:raise ValueError('Cannot identify a thinning probability with zero nominal adoption')
    return target_adopted/nominal_adopted


def independent_coins(probability,n,*,stream,model_index,stage):
    if not np.isfinite(probability) or not 0<=probability<=1 or type(n) is not int or n<=0:
        raise ValueError('A fixed valid probability and positive count are required')
    if any(type(v) is not int or v<0 for v in (stream,model_index,stage)):
        raise ValueError('Explicit nonnegative independent controller stream identifiers required')
    rng=np.random.default_rng(np.random.SeedSequence([stream,model_index,stage]))
    return rng.random(n)<probability


class EpisodeThinnedNominal:
    def __init__(self,model,profile,admit_nominal,*,factory=None):
        if type(admit_nominal) is not bool:raise ValueError('Preassigned independent Boolean episode coin required')
        if factory is None:
            from .mappo_deployment import make_recovery_gate
            factory=make_recovery_gate
        self.admit_nominal=admit_nominal
        self.inner=factory(model if admit_nominal else None,profile,'nominal_performance' if admit_nominal else 'baseline')

    def __getattr__(self,name):return getattr(self.inner,name)

    def select(self,*args,**kwargs):
        command,info=self.inner.select(*args,**kwargs)
        # Keep the truthful actual inner rule for physical/performance checking.
        # The study method ID and these fields identify the randomized selector.
        info=dict(info,random_abstention_control=True,episode_coin_admit_nominal=self.admit_nominal,
                  coin_is_actual_adoption=False,controller_coin_independent_of_environment=True)
        if not self.admit_nominal and info['actual_learning_used']:
            raise RuntimeError('Rejected episode coin cannot execute learning')
        return command,info
