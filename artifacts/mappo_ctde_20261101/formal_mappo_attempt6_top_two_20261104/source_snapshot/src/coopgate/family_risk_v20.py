"""All-or-none risk certification for three fixed models and two targets.

Policies, families, sample sizes and target definitions must be fixed before
collecting independent calibration data. This is an intersection-union test,
not simultaneous coverage of each separate pointwise interval and not a
statewise safety probability. No existing experiment is re-certified here.
"""
from numbers import Integral

import numpy as np
from scipy.stats import beta, binom

MODELS = (1802, 1812, 1822)
TARGETS = ('actual', 'bundle')
MEMBERS = tuple(f'{model}:{target}' for model in MODELS for target in TARGETS)


def family_certificate(counts, risk_limit=.05, family_error=.05):
    if set(counts) != set(MEMBERS):
        raise ValueError('All three fixed models and both targets must be present exactly once')
    if not np.isfinite([risk_limit, family_error]).all() or not 0. < risk_limit < 1. or not 0. < family_error < 1.:
        raise ValueError('Risk limit and family error must be probabilities strictly between zero and one')
    members = {}
    for name in MEMBERS:
        pair = counts[name]
        if len(pair) != 2:
            raise ValueError('Each member requires (harmful_adoptions, adoptions)')
        k, n = pair
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in pair) or not 0 <= k <= n:
            raise ValueError('Counts must be integers satisfying 0 <= harmful <= adopted')
        p = float(binom.cdf(k, n, risk_limit)) if n else 1.
        upper = float(beta.ppf(1. - family_error, k + 1, n - k)) if k < n else 1.
        members[name] = dict(harmful_adoptions=int(k), adoptions=int(n), pointwise_p_value=p,
                             pointwise_risk_upper=upper, conditional_risk_observed=bool(n))
    p_value = max(row['pointwise_p_value'] for row in members.values())
    maximum_risk_upper = max(row['pointwise_risk_upper'] for row in members.values())
    certified = bool(p_value <= family_error and maximum_risk_upper <= risk_limit)
    return dict(members=members, risk_limit=risk_limit, family_error=family_error,
                family_p_value=p_value, maximum_conditional_risk_upper=maximum_risk_upper,
                all_fixed_models_both_targets_certified=certified,
                claim_scope='One common upper bound on the maximum of all six fixed conditional risks; all-or-none decision',
                separate_member_intervals_simultaneously_covered=False,
                statewise_risk_certified=False, new_training_models_covered=False)


def certify_declared_families(counts_by_family, declared_families, risk_limit=.05, total_error=.05):
    names = tuple(declared_families)
    if not names or len(set(names)) != len(names) or set(counts_by_family) != set(names):
        raise ValueError('Every predeclared family must be supplied, without additions or omissions')
    if not np.isfinite(total_error) or not 0. < total_error < 1.:
        raise ValueError('Total error must be strictly between zero and one')
    level = total_error / len(names)
    return dict(total_error=total_error, family_error=level, declared_families=list(names),
        families={name: family_certificate(counts_by_family[name], risk_limit, level) for name in names},
        guarantee='Simultaneous common maximum-risk upper bounds across the predeclared families, by a union bound',
        current_experiments_recertified=False)
