"""Causal state envelopes from timestamped measurements and past commands.

Separate development component. It is not installed in any frozen controller.
Bounds cover the complete declared V39 compound family, rather than the older
lag_strong class. Floating-point padding is engineering, not interval arithmetic.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PublicEnvelope:
    dt: float = .1
    substeps: int = 5
    gain_lower: float = .78
    gain_upper: float = 1.22
    drag_lower: float = .00010
    drag_upper: float = .00100
    tau_lower: float = .12
    tau_upper: float = 1.25
    innovation_radius: float = .1
    maximum_delay: int = 3
    vmax: float = 20.
    command_lower: float = -5.
    command_upper: float = 2.5
    state_padding: float = 1e-8
    memory_padding: float = 1e-12

    def __post_init__(self):
        if (not np.isfinite(list(vars(self).values())).all()
                or self.dt <= 0 or type(self.substeps) is not int or self.substeps <= 0
                or not 0 < self.gain_lower <= self.gain_upper
                or not 0 <= self.drag_lower <= self.drag_upper
                or not 0 < self.tau_lower <= self.tau_upper
                or self.innovation_radius < 0 or type(self.maximum_delay) is not int
                or self.maximum_delay < 1 or self.vmax <= 0
                or not self.command_lower <= 0 <= self.command_upper
                or min(self.state_padding, self.memory_padding) < 0):
            raise ValueError('Invalid public envelope')


def _vector(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError('Finite '+name+' required')
    return result.copy()


def timestamp_vector(source, decision):
    raw = np.asarray(source)
    if raw.dtype.kind not in 'iu' or raw.shape not in ((), (3,)):
        raise ValueError('Integer scalar or three-vehicle source timestamps required')
    result = np.full(3, raw.item(), dtype=int) if raw.shape == () else raw.astype(int)
    if np.any(result < 0) or np.any(result > decision):
        raise ValueError('Source timestamp outside the received prefix')
    return result


def noise_bounds(profile):
    if profile in ('clean_reference', 'stale_dropout'):
        return np.zeros(2)
    if profile == 'bounded_bias_jitter':
        return np.array([1.50+.25, .60+.15])
    raise ValueError('Unknown public measurement profile')


def delayed_interval(history, maximum_delay):
    # history[-1] is the current physical decision's already executed command.
    index = len(history)-1
    values = [history[index-delay] if index-delay >= 0 else np.zeros(3)
              for delay in range(1, maximum_delay+1)]
    return np.min(values, axis=0), np.max(values, axis=0)


def memory_interval(lower, upper, input_lower, input_upper, bounds):
    rlo, rhi = np.exp(-bounds.dt/bounds.tau_lower), np.exp(-bounds.dt/bounds.tau_upper)
    lo = np.minimum(rlo*lower+(1-rlo)*input_lower,
                    rhi*lower+(1-rhi)*input_lower)-bounds.memory_padding
    hi = np.maximum(rlo*upper+(1-rlo)*input_upper,
                    rhi*upper+(1-rhi)*input_upper)+bounds.memory_padding
    return np.maximum(bounds.command_lower, lo), np.minimum(bounds.command_upper, hi)


def state_interval(lower, upper, memory_lower, memory_upper, bounds):
    lower, upper = np.asarray(lower), np.asarray(upper)
    alow = (bounds.gain_upper if memory_lower < 0 else bounds.gain_lower)*memory_lower
    ahigh = (bounds.gain_lower if memory_upper < 0 else bounds.gain_upper)*memory_upper
    alow -= bounds.drag_upper*upper[1]**2+bounds.innovation_radius
    ahigh += -bounds.drag_lower*lower[1]**2+bounds.innovation_radius
    vlo = np.clip(lower[1]+bounds.dt*alow, 0., bounds.vmax)
    vhi = np.clip(upper[1]+bounds.dt*ahigh, 0., bounds.vmax)
    lo = np.array([lower[0]+bounds.dt*vlo, vlo])-bounds.state_padding
    hi = np.array([upper[0]+bounds.dt*vhi, vhi])+bounds.state_padding
    lo[1], hi[1] = max(0., lo[1]), min(bounds.vmax, hi[1])
    return lo, hi


class AgeAwareStateObserver:
    """Only public measurements, source indices and executed commands enter.

    The reset contract is zero actuator memory and zero prior command. The
    sensor supplies a source timestamp for every vehicle. Constant hidden delay
    is relaxed to an independent {1,...,maximum_delay} choice each decision.
    """
    def __init__(self, profile, bounds=None):
        self.profile = profile
        self.error = noise_bounds(profile)
        self.bounds = bounds or PublicEnvelope()
        self.index = 0
        self.commands = []
        self.memory_lower = [np.zeros(3)]
        self.memory_upper = [np.zeros(3)]
        self.last = None

    def update(self, observed, source, previous):
        b, step = self.bounds, self.index
        observed = _vector(observed, (3, 2), 'measurement')
        previous = _vector(previous, (3,), 'previous command')
        stamps = timestamp_vector(source, step)
        if np.any(previous < b.command_lower) or np.any(previous > b.command_upper):
            raise ValueError('Previous command outside the public envelope')
        if step == 0 and np.any(previous != 0):
            raise ValueError('Zero-reset command contract required')
        if step:
            self.commands.append(previous)
            qlo, qhi = delayed_interval(self.commands, b.maximum_delay)
            for _ in range(b.substeps):
                lo, hi = memory_interval(self.memory_lower[-1], self.memory_upper[-1], qlo, qhi, b)
                self.memory_lower.append(lo); self.memory_upper.append(hi)
        assert len(self.memory_lower) == step*b.substeps+1
        lower = observed-self.error-b.state_padding
        upper = observed+self.error+b.state_padding
        lower[:, 1] = np.maximum(0., lower[:, 1])
        upper[:, 1] = np.minimum(b.vmax, upper[:, 1])
        if np.any(lower > upper):
            raise ValueError('Measurement has no state in the declared speed domain')
        for actor in range(3):
            for sub in range(int(stamps[actor])*b.substeps, step*b.substeps):
                lower[actor], upper[actor] = state_interval(lower[actor], upper[actor],
                    self.memory_lower[sub+1][actor], self.memory_upper[sub+1][actor], b)
        # Intersect with the last state's propagated box; both contain the same
        # current state under the assumptions, even when a frame is repeated.
        if self.last is not None:
            prior_lo, prior_hi = (x.copy() for x in self.last)
            for sub in range((step-1)*b.substeps, step*b.substeps):
                for actor in range(3):
                    prior_lo[actor], prior_hi[actor] = state_interval(prior_lo[actor], prior_hi[actor],
                        self.memory_lower[sub+1][actor], self.memory_upper[sub+1][actor], b)
            lower = np.maximum(lower, prior_lo); upper = np.minimum(upper, prior_hi)
            if np.any(lower > upper):
                raise ValueError('Received measurement contradicts the public model/history bounds')
        self.last = (lower.copy(), upper.copy())
        self.index += 1
        return dict(lower=lower.tolist(), upper=upper.tolist(),
            memory_lower=self.memory_lower[-1].tolist(), memory_upper=self.memory_upper[-1].tolist(),
            source_decision=stamps.tolist(), age_decisions=(step-stamps).tolist(),
            original_exact_state_certificate_restored=False,
            numerically_verified_interval_arithmetic=False, performance_risk_certified=False)


def estimate_frames(frames, profile):
    """Public-only facade, also usable with truth-denying mapping fixtures."""
    observer = AgeAwareStateObserver(profile)
    estimates = []
    for decision, frame in enumerate(frames):
        metadata = frame['perception']
        if metadata['decision'] != decision or metadata['profile'] != profile:
            raise ValueError('Received frame order/profile differs')
        estimates.append(observer.update(frame['observed_state'], metadata['source_decision'],
                                         frame['previous_command']))
    return estimates
