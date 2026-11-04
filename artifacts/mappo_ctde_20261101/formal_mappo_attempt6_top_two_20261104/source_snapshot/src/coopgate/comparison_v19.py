"""Frozen episode-level adapters for established statistical comparison targets.

Nominal forecasts are complete clean SUMO closed loops, run from the same
observed initial with no future disturbance. They are computed before selecting
a profile and can be expensive. A selected profile remains fixed for the whole
episode. No mixing of cached trajectories or per-vehicle profile outcomes.
"""
import math

import numpy as np

from .joint_admission_v15 import JointBound


def conformal_episode_bound(actual, predicted, failed, profile_ids, alpha=.05):
    """Simultaneous split-conformal upper bound including physical failures.

    A failed candidate/mode gives the entire calibration cluster score +inf.
    For a finite q, score <= q then implies both no failure and all component
    residual bounds. Thus selection cannot hide a collision behind a low cost.
    The guarantee is marginal over calibration and a new IID initial bundle,
    NOT 95%-confidence control of harm conditional on actual learning adoption.
    """
    actual = np.asarray(actual, float)
    predicted = np.asarray(predicted, float)
    failed = np.asarray(failed)
    names = tuple(profile_ids)
    if actual.ndim != 5 or min(actual.shape) < 1:
        raise ValueError('Expected initial x mode x candidate x vehicle x metric')
    if actual.shape[-2:] != (3, 2) or predicted.shape != (len(actual), *actual.shape[2:]):
        raise ValueError('Prediction/outcome shape mismatch')
    if failed.shape != actual.shape[:3] or failed.dtype != np.bool_:
        raise ValueError('Boolean failure indicator per initial/mode/candidate required')
    if len(names) != actual.shape[2] or len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError('Unique frozen complete-profile identities required')
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all() or not 0 < alpha < 1:
        raise ValueError('Finite capped outcomes and valid alpha required')
    scores = (actual - predicted[:, None]).reshape(len(actual), -1).max(1)
    scores[failed.any((1, 2))] = np.inf
    rank = math.ceil((len(actual) + 1) * (1 - alpha))
    q = float(np.partition(scores, rank - 1)[rank - 1]) if rank <= len(scores) else math.inf
    bound = JointBound(names, ((1., 1.),) * 3, q, len(actual), alpha)
    return bound, dict(calibration_initials=len(actual), rank=rank,
                       failed_initial_bundles=int(failed.any((1, 2)).sum()),
                       q=q if np.isfinite(q) else None, infinite_q=not np.isfinite(q),
                       finite_sample_marginal_coverage=True, adoption_conditional_risk_certified=False)


def conformal_profile(bound, predicted, names, seed, forecast_failed=None):
    """Select within one fixed training replicate, keeping the joint family bound."""
    upper = bound.upper(predicted, names)
    eligible = np.array([name.endswith('_' + str(seed)) for name in names])
    if not eligible.any():
        raise ValueError('Unknown fixed training replicate')
    if forecast_failed is not None:
        failures = np.asarray(forecast_failed)
        if failures.shape != (len(names),) or failures.dtype != np.bool_:
            raise ValueError('Boolean forecast-failure flags required')
        eligible &= ~failures
    team = upper[:, :, 0].mean(1)
    eligible &= np.all(upper <= 0, axis=(1, 2)) & (team < -1e-9)
    if not eligible.any():
        return 'baseline'
    return names[int(np.argmin(np.where(eligible, team, np.inf)))]


def select_fixed_replicates(names, eligible, means, seeds):
    """After simultaneous calibration, select within each replicate, never a seed."""
    if set(eligible) - set(names) or set(means) != set(names):
        raise ValueError('Candidate family mismatch')
    if not all(np.isfinite(value) for value in means.values()):
        raise ValueError('Finite calibration means required')
    result = {}
    for seed in seeds:
        allowed = [name for name in names if name in eligible and name.endswith('_' + str(seed))]
        result[str(seed)] = max(allowed, key=lambda name: means[name]) if allowed else 'baseline'
    return result


def conditional_counts(adopted, harmful):
    adopted, harmful = np.asarray(adopted), np.asarray(harmful)
    if adopted.shape != harmful.shape or adopted.dtype != np.bool_ or harmful.dtype != np.bool_:
        raise ValueError('Matching Boolean episode indicators required')
    n, k = int(adopted.sum()), int((adopted & harmful).sum())
    return dict(adopted=n, harmful_adopted=k, conditional_harm=k / n if n else None,
                adoption_fraction=n / adopted.size if adopted.size else None)
