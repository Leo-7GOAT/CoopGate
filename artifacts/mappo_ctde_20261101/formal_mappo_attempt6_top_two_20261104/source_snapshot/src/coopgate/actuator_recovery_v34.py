"""Interval stopping recovery with bounded actuator memory, gain and drag.

This is a separate development controller. It does not change the frozen
performance gate, its statistical calibration, or its physical error contract.
Only the declared plant class and actually executed commands reach the memory
observer. No realized coefficients, future innovations or simulator memory do.
The certificate concerns the benchmark's fixed-route geometric contract.
"""
import time

import numpy as np
from numba import njit

from .buffered_baseline_v25 import deduplicated_joint_rollout
from .fast_v8 import parameters
from .merge_v4 import MergeConfig
from .robust_v9 import interval_transition


def class_bounds(profile, dt=.1, innovation_radius=.1):
    """Rows are [gain lo/hi, drag lo/hi, retention lo/hi, innovation radius]."""
    if dt != .1 or innovation_radius != .1:
        raise ValueError('This development protocol fixes dt=.1 and innovation radius=.1')
    intervals = {
        'nominal': (1., 1., 0., 0., 0., 0.),
        'gain_drag': (.96, 1.04, .00025, .0005, 0., 0.),
        'lag_mild': (1., 1., 0., 0., .03, .05),
        'lag_strong': (1., 1., 0., 0., .3, .5),
    }
    if profile not in intervals:
        raise ValueError('Unknown declared actuator class')
    glo, ghi, dlo, dhi, tlo, thi = intervals[profile]
    rlo = 0. if tlo == 0. else np.exp(-dt/tlo)
    rhi = 0. if thi == 0. else np.exp(-dt/thi)
    return np.tile(np.array([glo, ghi, dlo, dhi, rlo, rhi, innovation_radius]), (3, 1))


@njit(cache=True, fastmath=False)
def memory_step(lower, upper, command, bounds):
    following_lower, following_upper = np.empty(3), np.empty(3)
    for i in range(3):
        rlo, rhi = bounds[i, 4], bounds[i, 5]
        u = command[i]
        following_lower[i] = min(rlo*lower[i]+(1.-rlo)*u,
                                 rhi*lower[i]+(1.-rhi)*u)
        following_upper[i] = max(rlo*upper[i]+(1.-rlo)*u,
                                 rhi*upper[i]+(1.-rhi)*u)
    return following_lower, following_upper


@njit(cache=True, fastmath=False)
def interval_substep(lower, upper, memory_lower, memory_upper, command, bounds, packed):
    mlo, mhi = memory_step(memory_lower, memory_upper, command, bounds)
    following_lower, following_upper = np.empty_like(lower), np.empty_like(upper)
    for i in range(3):
        glo, ghi, dlo, dhi, radius = bounds[i, 0], bounds[i, 1], bounds[i, 2], bounds[i, 3], bounds[i, 6]
        # Positive gains; memory may have either sign. The velocity intervals
        # are nonnegative, so squared drag is monotone on this domain.
        applied_lower = (ghi if mlo[i] < 0. else glo)*mlo[i] - dhi*upper[i, 1]**2 - radius
        applied_upper = (glo if mhi[i] < 0. else ghi)*mhi[i] - dlo*lower[i, 1]**2 + radius
        following_lower[i, 1] = min(packed[3], max(0., lower[i, 1]+applied_lower*packed[0]))
        following_upper[i, 1] = min(packed[3], max(0., upper[i, 1]+applied_upper*packed[0]))
        following_lower[i, 0] = lower[i, 0]+following_lower[i, 1]*packed[0]
        following_upper[i, 0] = upper[i, 0]+following_upper[i, 1]*packed[0]
    return following_lower, following_upper, mlo, mhi


@njit(cache=True, fastmath=False)
def stopping_certificate(state, previous, command, memory_lower, memory_upper, bounds, packed):
    """Return validity, reason code, substeps and terminal interval endpoints.

    Codes: 0 = stopped invariant set, 1 = input/slew, 2 = route geometry,
    3 = terminal set not reached. Failure is not a proof of unavoidability.
    Python entry points validate finite shapes and parameter domains first.
    """
    lo, hi = state.copy(), state.copy()
    mlo, mhi = memory_lower.copy(), memory_upper.copy()
    for i in range(3):
        if (command[i] < max(-packed[4], previous[i]-packed[6]*packed[1])-1e-10
                or command[i] > min(packed[5], previous[i]+packed[6]*packed[1])+1e-10):
            return False, 1, 0, lo, hi, mlo, mhi
    tail = command.copy()
    count = 0
    for stage in range(1+int(packed[10])):
        if stage > 0:
            tail = np.maximum(tail-packed[6]*packed[1], -packed[4])
        for _ in range(round(packed[1]/packed[0])):
            qlo, qhi, nmlo, nmhi = interval_substep(lo, hi, mlo, mhi, tail, bounds, packed)
            count += 1
            if not interval_transition(lo, hi, qlo, qhi, packed):
                return False, 2, count, qlo, qhi, nmlo, nmhi
            lo, hi, mlo, mhi = qlo, qhi, nmlo, nmhi
        invariant = stage > 0 and np.all(hi[:, 1] == 0.)
        for i in range(3):
            # At zero speed drag vanishes. Both current memory and the held
            # command keep worst-case positive innovation from restarting.
            limit = -bounds[i, 6]/bounds[i, 0]
            invariant = invariant and mhi[i] <= limit and tail[i] <= limit
        if invariant:
            return True, 0, count, lo, hi, mlo, mhi
    return False, 3, count, lo, hi, mlo, mhi


def validate_inputs(state, previous, memory_lower, memory_upper, bounds, config=None):
    config = config or MergeConfig()
    values = [np.ascontiguousarray(x, dtype=float) for x in
              (state, previous, memory_lower, memory_upper, bounds)]
    for value, shape in zip(values, ((3, 2), (3,), (3,), (3,), (3, 7))):
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError('Invalid finite interval-certificate input')
    x, prev, mlo, mhi, b = values
    if (np.any(x[:, 1] < 0.) or np.any(x[:, 1] > config.vmax)
            or np.any(prev < -config.brake) or np.any(prev > config.accel)
            or np.any(mlo > mhi) or np.any(mlo < -config.brake-1e-10)
            or np.any(mhi > config.accel+1e-10)
            or np.any(b[:, 0] <= 0.) or np.any(b[:, 0] > b[:, 1])
            or np.any(b[:, 2] < 0.) or np.any(b[:, 2] > b[:, 3])
            or np.any(b[:, 4] < 0.) or np.any(b[:, 4] > b[:, 5])
            or np.any(b[:, 5] >= 1.) or np.any(b[:, 6] < 0.)):
        raise ValueError('Interval or actuator domain violation')
    return values


def certificate(state, previous, command, memory_lower, memory_upper, bounds):
    c = MergeConfig()
    x, prev, mlo, mhi, b = validate_inputs(state, previous, memory_lower, memory_upper, bounds, c)
    u = np.ascontiguousarray(command, dtype=float)
    if u.shape != (3,) or not np.isfinite(u).all():
        raise ValueError('Invalid proposed command')
    result = stopping_certificate(x, prev, u, mlo, mhi, b, parameters(c))
    return dict(certified=bool(result[0]),reason_code=int(result[1]),substeps=int(result[2]),
                terminal_lower=result[3].tolist(),terminal_upper=result[4].tolist(),
                terminal_memory_lower=result[5].tolist(),terminal_memory_upper=result[6].tolist())


class ActuatorRecoveryBaseline:
    """Original strong baseline proposal with an actuator-aware stopping check.

    If the nominal planner rejects its own model state, or the proposed command
    fails the new check, execute the jerk-limited stopping backup when certified.
    No performance improvement or eventual arrival is implied by safe stopping.
    """
    def __init__(self, profile):
        self.c = MergeConfig()
        self.packed = parameters(self.c)
        self.bounds = class_bounds(profile)
        self.profile = profile
        self.memory_lower, self.memory_upper = np.zeros(3), np.zeros(3)
        self.step = 0

    def select(self, state, previous, unused=None):
        started = time.perf_counter()
        x, prev, mlo, mhi, b = validate_inputs(state, previous, self.memory_lower,
                                               self.memory_upper, self.bounds, self.c)
        if self.step >= round(self.c.horizon/self.c.decision_dt):
            raise ValueError('Finite horizon exhausted')
        if self.step == 0 and np.any(prev):
            raise ValueError('Memory reset requires the declared zero initial command')
        if self.step > 0:
            for _ in range(round(self.c.decision_dt/self.c.dt)):
                mlo, mhi = memory_step(mlo, mhi, prev, b)
        self.memory_lower, self.memory_upper = mlo.copy(), mhi.copy()
        proposal_error = None
        proposal = None
        try:
            proposal, _, _, _ = deduplicated_joint_rollout(x, prev, self.step, self.packed)
        except RuntimeError as exc:
            proposal_error = str(exc)
        selected = None
        proposal_check = None
        backup_check = None
        if proposal is not None:
            proposal_check = certificate(x, prev, proposal, mlo, mhi, b)
            if proposal_check['certified']:
                selected = proposal
        recovery = selected is None
        if recovery:
            backup = np.maximum(prev-self.c.jerk*self.c.decision_dt, -self.c.brake)
            backup_check = certificate(x, prev, backup, mlo, mhi, b)
            if not backup_check['certified']:
                raise RuntimeError('No actuator-aware stopping certificate from current state')
            selected = backup
        self.step += 1
        return selected, dict(in_set=True,certified=True,intervention=float(recovery),
            gate_ms=(time.perf_counter()-started)*1000.,
            certificate_kind='bounded_actuator_memory_stopping',
            actuator_class=self.profile,actuator_bounds=b.tolist(),
            actuator_memory_lower=mlo.tolist(),actuator_memory_upper=mhi.tolist(),
            nominal_proposal=None if proposal is None else proposal.tolist(),
            nominal_proposal_error=proposal_error,proposal_certificate=proposal_check,
            recovery_certificate=backup_check,recovery_applied=recovery,
            safety_scope='Declared actuator class, bounded innovation, exact observed state and fixed-route contract',
            performance_risk_certified=False)
