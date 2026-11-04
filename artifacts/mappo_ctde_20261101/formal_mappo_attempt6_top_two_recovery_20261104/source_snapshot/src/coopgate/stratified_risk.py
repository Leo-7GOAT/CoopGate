"""Finite-sample conditional risk bounds for a fixed stratified design.

Rows must be independent and identically distributed within each stratum.
Stratum sizes and target population weights are fixed before outcomes exist.
No homogeneity of adoption or harm probabilities across strata is assumed.
"""
import numpy as np
from scipy.stats import beta


def _inputs(harmful_adoptions, adoptions, totals, weights):
    arrays = [np.asarray(v) for v in (harmful_adoptions, adoptions, totals)]
    k, a, n = arrays
    w = np.asarray(weights, dtype=float)
    if (n.ndim != 1 or not n.size or any(x.shape != n.shape for x in (*arrays, w))
            or any(x.dtype.kind not in 'iu' for x in arrays)
            or np.any(n <= 0) or np.any(k < 0) or np.any(k > a) or np.any(a > n)
            or not np.isfinite(w).all() or np.any(w <= 0)
            or not np.isclose(w.sum(), 1., atol=1e-12, rtol=0.)):
        raise ValueError('Require complete integer 0 <= harm <= adopted <= total and fixed positive weights summing to one')
    return k, a, n, w


def conditional_risk_upper(harmful_adoptions, adoptions, totals, weights, *, alpha):
    """Bound sum(w P(A and H)) / sum(w P(A)) via 2S exact bounds.

    A union bound inside this ONE inferential target allocates alpha/(2S)
    to each numerator upper and denominator lower bound. Dependence between
    these two counts, or between strata, does not affect the union bound.
    IID sampling within strata is still required for each binomial bound.
    """
    k, a, n, w = _inputs(harmful_adoptions, adoptions, totals, weights)
    if not np.isfinite(alpha) or not 0. < alpha < 1.:
        raise ValueError('alpha must lie strictly between zero and one')
    delta = float(alpha / (2 * len(n)))
    q_upper = np.ones(len(n))
    mask = k < n
    q_upper[mask] = beta.isf(delta, k[mask] + 1, n[mask] - k[mask])
    a_lower = np.zeros(len(n))
    mask = a > 0
    a_lower[mask] = beta.ppf(delta, a[mask], n[mask] - a[mask] + 1)
    numerator, denominator = float(w @ q_upper), float(w @ a_lower)
    upper = min(1., numerator / denominator) if denominator > 0. else 1.
    return dict(risk_upper=upper, joint_harm_upper=numerator,
        adoption_lower=denominator, stratum_joint_harm_uppers=q_upper.tolist(),
        stratum_adoption_lowers=a_lower.tolist(), component_alpha=delta,
        allocated_error=float(alpha), target_weights=w.tolist(),
        finite_sample_valid_under_within_stratum_iid=True,
        pooled_adoptions_are_binomial=False)


def conditional_risk_certificate(harmful_adoptions, adoptions, totals, weights,
                                 *, alpha, risk_limit):
    """Invert the nested risk bounds to obtain a conservative p-value.

    Bisection returns the upper endpoint of the numerical bracket, conservatively
    rounding the inferred p-value upward rather than downward.
    """
    result = conditional_risk_upper(harmful_adoptions, adoptions, totals, weights,
                                    alpha=alpha)
    if not np.isfinite(risk_limit) or not 0. < risk_limit < 1.:
        raise ValueError('risk_limit must lie strictly between zero and one')
    low, high = 0., 1.
    for _ in range(48):
        middle = (low + high) / 2
        bound = conditional_risk_upper(harmful_adoptions, adoptions, totals,
                                      weights, alpha=middle)['risk_upper']
        if bound <= risk_limit:
            high = middle
        else:
            low = middle
    result.update(pointwise_p_value=high, risk_limit=float(risk_limit),
                  certified=bool(result['risk_upper'] <= risk_limit and high <= alpha))
    return result
