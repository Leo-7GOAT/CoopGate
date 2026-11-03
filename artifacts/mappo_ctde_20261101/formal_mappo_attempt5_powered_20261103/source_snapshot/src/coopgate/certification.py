"""Finite-family episode-level risk tests, not per-action safety probabilities."""
import math
from scipy.stats import beta


def binomial_upper(failures, trials, delta=.05, family_size=1):
    """One-sided exact Clopper-Pearson bound with Bonferroni correction.

    Each trial must be an independent episode from the specified distribution.
    Dependence BETWEEN fixed candidate controllers on paired scenes is allowed.
    The family must be fixed independently of the certification episodes.
    """
    if not 0 <= failures <= trials or trials < 0 or not 0 < delta < 1 or family_size < 1:
        raise ValueError('Invalid count or confidence parameters')
    if failures == trials:
        return 1.0
    return float(beta.ppf(1-delta/family_size, failures+1, trials-failures))


def zero_failure_sample_size(risk=.05, delta=.05, family_size=1):
    if not 0 < risk < 1 or not 0 < delta < 1 or family_size < 1:
        raise ValueError('Invalid risk/confidence parameters')
    return math.ceil(math.log(delta/family_size)/math.log1p(-risk))


def certify(failures, trials, target=.05, delta=.05, family_size=1):
    bound=binomial_upper(failures,trials,delta,family_size)
    return {'failures':int(failures),'episodes':int(trials),'risk_upper':bound,
            'target':target,'delta_family':delta,'family_size':family_size,
            'certified':bool(bound<=target)}
