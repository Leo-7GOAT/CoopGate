"""Experimental episode-level admission for a fixed family of joint policies.

No MARL controller is trained here. This is a split-conformal implementation,
not a new conformal theorem. Each calibration unit is an independent initial
scene; modes, candidates, vehicles and metrics remain inside that cluster.
Physical command certification is a separate, mandatory controller operation.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class JointBound:
    profile_ids: tuple[str, ...]
    scales: tuple[tuple[float, ...], ...]
    q: float
    n_initials: int
    alpha: float

    def upper(self, predicted, profile_ids):
        if tuple(profile_ids) != self.profile_ids:
            raise ValueError('Joint policy family changed after calibration')
        predicted = np.asarray(predicted, dtype=float)
        scales = np.asarray(self.scales)
        if predicted.shape != (len(self.profile_ids), *scales.shape):
            raise ValueError('Expected candidate x vehicle x metric predictions')
        if not np.isfinite(predicted).all():
            raise ValueError('Nonfinite performance prediction')
        return predicted + self.q * scales

    def select(self, predicted, profile_ids, margins=None, minimum_team_gain=.05):
        """Return a whole-profile index or None for the baseline profile.

        Metric zero is individual cost in seconds; remaining metrics have
        prespecified units/scales. The mean upper cost selects among eligible
        whole profiles. Commands from different profiles are never composed.
        A wrapper must also catch validation exceptions and use its baseline.
        """
        upper = self.upper(predicted, profile_ids)
        limits = np.zeros(upper.shape[1:]) if margins is None else np.asarray(margins, dtype=float)
        if limits.shape != upper.shape[1:] or not np.isfinite(limits).all():
            raise ValueError('Invalid individual non-degradation margins')
        if not np.isfinite(minimum_team_gain) or minimum_team_gain < 0:
            raise ValueError('Invalid minimum gain')
        team = upper[..., 0].mean(axis=-1)
        eligible = np.all(upper <= limits, axis=(-2, -1)) & (team <= -minimum_team_gain)
        if not eligible.any():
            return None, upper
        return int(np.argmin(np.where(eligible, team, np.inf))), upper


def calibrate_joint(actual_differences, predicted_differences, profile_ids,
                    alpha=.05, scales=None):
    """Calibrate simultaneous upper bounds for actual cost minus baseline.

    actual: initial x disturbance mode x candidate x vehicle x metric.
    prediction: initial x candidate x vehicle x metric, fixed before calibration.
    IDs must bind each complete frozen policy profile (e.g. to weight hashes).
    All outcomes use the SAME initial and exogenous tape as their baseline.
    """
    actual = np.asarray(actual_differences, dtype=float)
    predicted = np.asarray(predicted_differences, dtype=float)
    ids = tuple(profile_ids)
    if actual.ndim != 5 or min(actual.shape) < 1:
        raise ValueError('Expected nonempty clustered joint outcomes')
    if predicted.shape != (actual.shape[0], *actual.shape[2:]):
        raise ValueError('Prediction/outcome shapes differ')
    if len(ids) != actual.shape[2] or len(set(ids)) != len(ids) or any(not s for s in ids):
        raise ValueError('Unique frozen joint-profile IDs required')
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError('Failed/nonfinite outcomes must be resolved before calibration')
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('alpha must be between zero and one')
    scale = np.ones(actual.shape[-2:]) if scales is None else np.asarray(scales, dtype=float)
    if scale.shape != actual.shape[-2:] or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Positive fixed scales required for every vehicle/metric')
    residual = (actual - predicted[:, None]) / scale
    scores = residual.reshape(len(actual), -1).max(axis=1)
    k = math.ceil((len(scores) + 1) * (1 - alpha))
    q = float(np.partition(scores, k - 1)[k - 1]) if k <= len(scores) else math.inf
    return JointBound(ids, tuple(tuple(float(v) for v in row) for row in scale),
                      q, len(scores), float(alpha))
