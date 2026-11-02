"""Established statistical gates, with their different targets kept explicit.

LTT: exact binomial p-values on actual-adoption subsets and fixed-sequence or
Bonferroni familywise error control (Angelopoulos et al., arXiv:2110.01052).
CSPI-style full-information adaptation: simultaneous Gaussian mean-gain bands
for paired simulator outcomes (Cho et al., arXiv:2408.12004, Section 4.3).
This is NOT their observational doubly-robust estimator or cutoff-search rule.
Neither construction is introduced here as a new statistical theorem.
"""
import numpy as np
from scipy.stats import beta, binom, norm


def _design(initial_ids, candidate_ids, values):
    initial_ids, candidate_ids = tuple(initial_ids), tuple(candidate_ids)
    array = np.asarray(values)
    if not initial_ids or len(set(initial_ids)) != len(initial_ids):
        raise ValueError('Unique independent initial IDs required')
    if not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError('Unique previously frozen candidate IDs required')
    if array.shape != (len(initial_ids), len(candidate_ids)):
        raise ValueError('One row per independent scene and column per candidate required')
    return initial_ids, candidate_ids, array


def conditional_ltt(initial_ids, candidate_ids, admitted, harmful, alpha=.05, delta=.05,
                    procedure='bonferroni', order=None):
    ids, names, admitted = _design(initial_ids, candidate_ids, admitted)
    _, _, harmful = _design(ids, names, harmful)
    if admitted.dtype != np.bool_ or harmful.dtype != np.bool_:
        raise ValueError('Boolean adoption and harm indicators required')
    if not 0 < alpha < 1 or not 0 < delta < 1:
        raise ValueError('Risk and error levels must lie in (0,1)')
    n = admitted.sum(0)
    k = (admitted & harmful).sum(0)
    # Conditional on n, k is binomial for IID independent scene tuples (A,H).
    # A may include the realized closed-loop admission; it need not be X-only.
    p = np.where(n > 0, binom.cdf(k, n, alpha), 1.)
    if procedure == 'bonferroni':
        if order is not None:
            raise ValueError('Bonferroni does not use an ordering')
        levels = np.full(len(names), delta / len(names))
        certified = (n > 0) & (p <= levels)
        checked = np.ones(len(names), bool)
    elif procedure == 'fixed_sequence':
        if order is None or sorted(order) != list(range(len(names))):
            raise ValueError('A full ordering frozen before calibration is required')
        levels = np.full(len(names), delta)
        certified, checked = np.zeros(len(names), bool), np.zeros(len(names), bool)
        for index in order:
            checked[index] = True
            if n[index] == 0 or p[index] > delta:
                break
            certified[index] = True
    else:
        raise ValueError('Unknown LTT familywise procedure')
    upper = np.ones(len(names))
    valid = (n > 0) & (k < n)
    upper[valid] = beta.ppf(1. - levels[valid], k[valid] + 1, n[valid] - k[valid])
    return dict(procedure=procedure, independent_initials=len(ids), alpha=alpha, delta=delta,
                order=list(order) if order is not None else None,
                candidates=[dict(candidate=name, adopted=int(n[j]), harmful=int(k[j]),
                                 empirical_conditional_risk=float(k[j] / n[j]) if n[j] else None,
                                 binomial_pvalue=float(p[j]), tested=bool(checked[j]),
                                 threshold_upper=float(upper[j]), certified=bool(certified[j]))
                            for j, name in enumerate(names)],
                statement='Finite-sample familywise control for IID scene-level tuples and a frozen family; fixed-sequence bounds are only certified for the accepted prefix, not simultaneous free selection of unchecked candidates.')


def paired_mean_improvement_bands(initial_ids, candidate_ids, gains, delta=.05,
                                  simulations=50000, seed=19241):
    """CSPI-MT-style simultaneous mean testing under paired full information.

    Both potential outcomes are available in a controlled paired simulation, so
    the influence contribution is the observed paired gain, not an observational
    propensity estimate. The Gaussian approximation is ASYMPTOTIC. Zero empirical
    variance is treated as untestable, rather than proof of certainty.
    """
    ids, names, gain = _design(initial_ids, candidate_ids, gains)
    gain = np.asarray(gain, dtype=float)
    if len(ids) < 3 or not np.isfinite(gain).all() or not 0 < delta < 1:
        raise ValueError('Finite independent scene gains and a valid level required')
    if type(simulations) is not int or simulations < 1000:
        raise ValueError('At least 1000 Gaussian simulations required')
    mean = gain.mean(0)
    covariance = np.atleast_2d(np.cov(gain, rowvar=False, ddof=1))
    sd = np.sqrt(np.maximum(0., np.diag(covariance)))
    active = sd > 1e-12
    lower = np.full(len(names), -np.inf)
    critical = None
    if active.any():
        corr = covariance[np.ix_(active, active)] / np.outer(sd[active], sd[active])
        eigval, eigvec = np.linalg.eigh(corr)
        if eigval.min() < -1e-8:
            raise ValueError('Invalid empirical covariance')
        root = eigvec * np.sqrt(np.maximum(eigval, 0.))
        normal = np.random.default_rng(seed).standard_normal((simulations, int(active.sum()))) @ root.T
        maximum = normal.max(1)
        critical = float(np.quantile(maximum, 1. - delta, method='higher'))
        # At least the analytic univariate critical value. A singular duplicate
        # candidate must not acquire a more permissive critical value by MC noise.
        critical = max(critical, float(norm.ppf(1. - delta)))
        lower[active] = mean[active] - critical * sd[active] / np.sqrt(len(ids))
    return dict(independent_initials=len(ids), delta=delta, simulations=simulations, seed=seed,
                critical=critical, finite_sample_guarantee=False,
                candidates=[dict(candidate=name, mean_gain=float(mean[j]), standard_error=float(sd[j] / np.sqrt(len(ids))),
                                 lower=float(lower[j]) if np.isfinite(lower[j]) else None,
                                 testable=bool(active[j]), mean_improvement_selected=bool(active[j] and lower[j] > 0.))
                            for j, name in enumerate(names)],
                target='Positive population MEAN paired team gain, not harm risk among admitted scenes; Gaussian joint band is an asymptotic, full-information adaptation, not a reproduction of observational CSPI-MT.')
