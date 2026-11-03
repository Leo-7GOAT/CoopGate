"""Finite-grid selective performance risk control for declared traffic regimes.

No collision probability, arbitrary-shift guarantee, or new policy training.
The same scalar threshold is deployed without observing the regime label.
"""
import numpy as np
from scipy.stats import beta

from .learning_v6 import predict


def binomial_upper(failures, trials, delta):
    """One-sided exact Clopper-Pearson bound, including empty acceptance."""
    if not (isinstance(trials, (int, np.integer)) and isinstance(failures, (int, np.integer))):
        raise ValueError('Counts must be integers')
    if not 0 <= failures <= trials or not 0 < delta < 1:
        raise ValueError('Invalid binomial counts or confidence parameter')
    if trials == 0 or failures == trials:
        return 1.0
    return float(beta.ppf(1-delta, failures+1, trials-failures))


def calibrate_threshold(groups, thresholds, risk=.1, delta=.05, tolerance=1e-6):
    """Simultaneously certify every fixed threshold in every fixed regime.

    groups maps regime to aligned proposal, predicted, actual arrays. Models,
    score, threshold grid and regimes must be fixed before calibration access.
    The selected threshold maximizes observed acceptance among certified ones.
    None means reject every nonbaseline proposal; no empty-set risk estimate.
    """
    grid = np.asarray(thresholds, dtype=float)
    if grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all() or len(set(grid)) != len(grid):
        raise ValueError('A finite nonempty distinct threshold grid is required')
    if not groups or not 0 < risk < 1 or not 0 < delta < 1 or tolerance < 0:
        raise ValueError('Invalid calibration configuration')
    per_test_delta = delta/(len(groups)*len(grid))
    candidates = []
    for threshold in sorted(grid):
        stats = {}
        for name, values in groups.items():
            proposal, predicted, actual = (np.asarray(values[k]) for k in ('proposal', 'predicted', 'actual'))
            if proposal.ndim != 1 or proposal.shape != predicted.shape or proposal.shape != actual.shape or not len(proposal):
                raise ValueError('Each regime requires aligned nonempty vectors')
            if not np.isfinite(predicted).all() or not np.isfinite(actual).all():
                raise ValueError('Finite calibration scores required')
            accept = (proposal != 0) & (predicted <= threshold)
            n = int(accept.sum()); k = int(np.sum(accept & (actual > tolerance)))
            stats[name] = {'accepted': n, 'failures': k, 'upper': binomial_upper(k, n, per_test_delta)}
        candidates.append({'threshold_s': float(threshold), 'regimes': stats,
                           'total_accepted': sum(v['accepted'] for v in stats.values()),
                           'certified': all(v['upper'] <= risk for v in stats.values())})
    valid = [c for c in candidates if c['certified']]
    # For tied acceptance choose the stricter threshold, predeclared.
    selected = max(valid, key=lambda c: (c['total_accepted'], -c['threshold_s'])) if valid else None
    return {'risk': risk, 'delta': delta, 'tolerance_s': tolerance, 'per_test_delta': per_test_delta,
            'threshold_s': selected['threshold_s'] if selected else None, 'selected': selected,
            'candidates': candidates}


def choose(proposal, estimate, threshold):
    """A failed/absent certificate always falls back, including nonfinite scores."""
    accept = (proposal != 0 and np.isfinite(estimate) and threshold is not None
              and np.isfinite(threshold) and estimate <= threshold)
    return int(proposal) if accept else 0


def decide(models, state, method, calibration, old_quantile):
    if method == 'constant':
        return 0, {'proposal': 0, 'estimated_excess_s': None, 'threshold_s': None}
    proposal, estimate = predict(models, np.asarray(state)[None])
    proposal, estimate = int(proposal[0]), float(estimate[0])
    if method == 'learned':
        selected, threshold = proposal, None
    elif method == 'v6_conformal':
        threshold = -old_quantile; selected = choose(proposal, estimate, threshold)
    elif method == 'robust_conformal':
        threshold = -calibration['max_conformal_quantile_s']; selected = choose(proposal, estimate, threshold)
    elif method == 'robust_pac':
        threshold = calibration['pac']['threshold_s']; selected = choose(proposal, estimate, threshold)
    else:
        raise ValueError(method)
    return selected, {'proposal': proposal, 'estimated_excess_s': estimate, 'threshold_s': threshold}
