"""Exact prospective conditional-risk rules for the fixed recovery gate.

The unit of sampling is one independent initial-condition cluster.  Each
cluster is assigned one scenario, actuator class and realized disturbance mode
before execution.  A fixed learned-policy seed is either adopted at least once
in the episode or it is not.  Harm retains controller failure and every
vehicle's paired cost/arrival regression.

The primary statement is an intersection-union claim: all three fixed models
must satisfy both the declared mixture and strong-lag conditional-risk targets.
Consequently the family p-value is the maximum constituent p-value.  Selecting
among controller families or repeating attempts requires a separate error
allocation; this module uses the project's summable attempt schedule.
"""
from numbers import Integral

import numpy as np
from scipy.stats import beta, binom


MODELS = (1802, 1812, 1822)
TARGETS = ('mixture', 'strong_lag')
MEMBERS = tuple(f'{model}:{target}' for model in MODELS for target in TARGETS)
RISK_LIMIT = .05
TOTAL_ATTEMPT_ERROR = .05
CALIBRATION_N = 3072
CONFIRMATION_N = 1024


def attempt_error(attempt, total_error=TOTAL_ATTEMPT_ERROR):
    if (isinstance(attempt, (bool, np.bool_)) or not isinstance(attempt, Integral)
            or attempt < 1 or not np.isfinite(total_error) or not 0. < total_error < 1.):
        raise ValueError('A positive integer attempt and probability error budget are required')
    return float(total_error / (attempt * (attempt + 1)))


def _count_pair(pair):
    if len(pair) != 2:
        raise ValueError('Each target requires (harmful_adoptions, adoptions)')
    k, n = pair
    if (any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in pair)
            or not 0 <= k <= n):
        raise ValueError('Counts must be integers satisfying 0 <= harmful <= adopted')
    return int(k), int(n)


def recovery_family_certificate(counts, *, attempt=2, risk_limit=RISK_LIMIT,
                                total_error=TOTAL_ATTEMPT_ERROR):
    """Certify one predeclared six-member family without model selection."""
    if set(counts) != set(MEMBERS):
        raise ValueError('All three fixed models and both targets are required exactly once')
    if not np.isfinite(risk_limit) or not 0. < risk_limit < 1.:
        raise ValueError('Risk limit must be a probability strictly between zero and one')
    level = attempt_error(attempt, total_error)
    members = {}
    for name in MEMBERS:
        k, n = _count_pair(counts[name])
        p_value = float(binom.cdf(k, n, risk_limit)) if n else 1.
        upper = float(beta.ppf(1. - level, k + 1, n - k)) if k < n else 1.
        members[name] = dict(harmful_adoptions=k, adoptions=n,
            pointwise_p_value=p_value, pointwise_risk_upper=upper,
            conditional_risk_observed=bool(n))
    family_p = max(item['pointwise_p_value'] for item in members.values())
    maximum_upper = max(item['pointwise_risk_upper'] for item in members.values())
    return dict(members=members, risk_limit=float(risk_limit), attempt=int(attempt),
        allocated_error=level, total_attempt_error=float(total_error),
        family_p_value=family_p, maximum_conditional_risk_upper=maximum_upper,
        all_fixed_models_both_targets_certified=bool(
            family_p <= level and maximum_upper <= risk_limit),
        claim_scope=('One common upper bound on the maximum conditional harm risk '
                     'over three fixed models and the mixture/strong-lag targets'),
        intersection_union_test=True, constituent_bonferroni_used=False,
        statewise_probability_certified=False, out_of_distribution_risk_certified=False)


def target_counts(adopted, harmful, plants):
    """Build six fixed counts from complete cluster-level observations."""
    a, h, p = np.asarray(adopted), np.asarray(harmful), np.asarray(plants)
    if (a.shape != (CALIBRATION_N, len(MODELS)) or h.shape != a.shape
            or a.dtype != np.bool_ or h.dtype != np.bool_
            or p.shape != (CALIBRATION_N,) or p.dtype.kind not in 'USO'):
        raise ValueError('Complete Boolean cluster/model arrays and plant labels are required')
    if any(value not in ('nominal', 'gain_drag', 'lag_mild', 'lag_strong') for value in p):
        raise ValueError('Unknown actuator class in the fixed calibration mixture')
    counts = {}
    masks = dict(mixture=np.ones(len(p), dtype=bool), strong_lag=p == 'lag_strong')
    for column, model in enumerate(MODELS):
        for target, mask in masks.items():
            selected = mask & a[:, column]
            counts[f'{model}:{target}'] = (int((selected & h[:, column]).sum()),
                                           int(selected.sum()))
    return counts


def certify_complete_observations(adopted, harmful, failed, plants, *, attempt=2):
    """Keep conditional risk and unconditional execution failures distinct."""
    a, h, f = map(np.asarray, (adopted, harmful, failed))
    if f.shape != (CALIBRATION_N, len(MODELS)) or f.dtype != np.bool_:
        raise ValueError('Complete Boolean failure observations are required')
    if a.shape != f.shape or h.shape != f.shape or a.dtype != np.bool_ or h.dtype != np.bool_:
        raise ValueError('Adoption, harm and failure arrays must share the complete shape')
    if np.any(f & ~h):
        raise ValueError('Every controller failure must be retained as harm')
    certificate = recovery_family_certificate(target_counts(a, h, plants), attempt=attempt)
    return dict(primary_certificate=certificate,
        nonadopted_controller_failures=int((f & ~a).sum()),
        all_controller_executions_complete=not bool(f.any()),
        initial_count=CALIBRATION_N, model_count=len(MODELS),
        one_realized_disturbance_per_independent_cluster=True,
        calibration_outcomes_used_for_controller_selection=False)
