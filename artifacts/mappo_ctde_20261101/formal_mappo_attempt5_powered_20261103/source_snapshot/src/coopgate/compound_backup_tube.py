"""Development-only stopping tubes for the public compound plant family.

Not installed in any frozen experiment. The mathematical result is conditional
on containing boxes and declared bounds; float64 is not verified interval
arithmetic. A failed search proves neither collision nor infeasibility.
"""
from copy import deepcopy
from dataclasses import asdict

import numpy as np

from .aged_observation_envelope import (
    PublicEnvelope, delayed_interval, memory_interval, state_interval,
)


def _array(value, shape):
    out = np.asarray(value, dtype=float)
    if out.shape != shape or not np.isfinite(out).all():
        raise ValueError('Finite array with shape '+str(shape)+' required')
    return out.copy()


def _occupancy(pl, pu, ql, qu, start, end):
    if qu < start or pl > end:
        return None
    enter = 0. if pu >= start else (start-pu)/max(qu-pu, 1e-300)
    leave = 1. if ql <= end else (end-pl)/max(ql-pl, 1e-300)
    return max(0., enter), min(1., leave)


def route_box_safe(lo, hi, nxt_lo, nxt_hi, separation=6.8,
                   zone_start=-25., zone_end=6.8, guard=1e-7):
    """Sufficient affine-substep geometry for fixed routes [0, 1, 0].

    All position endpoints must bound a nondecreasing route trajectory. The
    outer occupancy intervals and post-entry gap reproduce the benchmark's
    contract, with stricter separation and an expanded closed conflict zone.
    This is not arbitrary two-dimensional collision checking.
    """
    sep = separation+guard
    start, end = zone_start-guard, zone_end+guard
    if min(lo[0, 0]-hi[2, 0], nxt_lo[0, 0]-nxt_hi[2, 0]) < sep:
        return False
    for i, j in ((0, 1), (1, 2)):
        a = _occupancy(lo[i, 0], hi[i, 0], nxt_lo[i, 0], nxt_hi[i, 0], start, end)
        b = _occupancy(lo[j, 0], hi[j, 0], nxt_lo[j, 0], nxt_hi[j, 0], start, end)
        if a is not None and b is not None and max(a[0], b[0]) <= min(a[1], b[1]):
            return False
        if nxt_hi[i, 0] >= start and nxt_hi[j, 0] >= start:
            ti = 0. if hi[i, 0] >= start else (start-hi[i, 0])/max(nxt_hi[i, 0]-hi[i, 0], 1e-300)
            tj = 0. if hi[j, 0] >= start else (start-hi[j, 0])/max(nxt_hi[j, 0]-hi[j, 0], 1e-300)
            t = max(ti, tj)
            low_gap = lo[i, 0]+t*(nxt_lo[i, 0]-lo[i, 0])-hi[j, 0]-t*(nxt_hi[j, 0]-hi[j, 0])
            high_gap = hi[i, 0]+t*(nxt_hi[i, 0]-hi[i, 0])-lo[j, 0]-t*(nxt_lo[j, 0]-lo[j, 0])
            ahead = min(low_gap, nxt_lo[i, 0]-nxt_hi[j, 0]) >= sep
            behind = max(high_gap, nxt_hi[i, 0]-nxt_lo[j, 0]) <= -sep
            if not (ahead or behind):
                return False
    return True


def _substep(lo, hi, ml, mh, qlo, qhi, bounds):
    ml, mh = memory_interval(ml, mh, qlo, qhi, bounds)
    following_lo, following_hi = np.empty_like(lo), np.empty_like(hi)
    for i in range(3):
        following_lo[i], following_hi[i] = state_interval(lo[i], hi[i], ml[i], mh[i], bounds)
        # The older observer pads v=0 upward. A strictly negative pre-clipping
        # upper velocity proves saturation at exactly zero in real arithmetic;
        # retain position padding, rather than pretending a small speed is zero.
        amax = (bounds.gain_upper if mh[i] >= 0 else bounds.gain_lower)*mh[i]
        amax += -bounds.drag_lower*lo[i, 1]**2+bounds.innovation_radius
        if hi[i, 1]+bounds.dt*amax < -max(bounds.state_padding, 1e-12):
            following_lo[i, 1] = following_hi[i, 1] = 0.
            following_lo[i, 0] = lo[i, 0]-bounds.state_padding
            following_hi[i, 0] = hi[i, 0]+bounds.state_padding
    return following_lo, following_hi, ml, mh


def certify_braking_tube(lower, upper, memory_lower, memory_upper,
                        executed_commands, first_command, *, bounds=None,
                        jerk=5., max_decisions=64, separation=6.8,
                        zone_start=-25., zone_end=6.8, geometry_guard=1e-7):
    """Check one candidate followed by a jerk-limited all-vehicle brake tail.

    Only current public boxes and past executed commands enter. Unknown fixed
    delay is conservatively relaxed to a fresh choice every decision. Returned
    commands and endpoint tubes include the checked candidate, even on failure.
    On acceptance, hold the final command forever after following the complete
    checked tape. A deployment must retain the suffix across replanning failure.
    """
    b = bounds or PublicEnvelope()
    lo, hi = _array(lower, (3, 2)), _array(upper, (3, 2))
    ml, mh = _array(memory_lower, (3,)), _array(memory_upper, (3,))
    hist = np.asarray(executed_commands, dtype=float)
    if hist.size == 0 and hist.shape in ((0,), (0, 3)):
        hist = np.empty((0, 3))
    if hist.ndim != 2 or hist.shape[1:] != (3,) or not np.isfinite(hist).all():
        raise ValueError('Executed command prefix must have shape (decisions, 3)')
    history = [u.copy() for u in hist]
    command = _array(first_command, (3,))
    constants = [jerk, separation, zone_start, zone_end, geometry_guard]
    if (not np.isfinite(constants).all() or jerk <= 0 or separation <= 0
            or zone_start >= zone_end or geometry_guard < 0
            or type(max_decisions) is not int or max_decisions < 1
            or b.command_lower >= -b.innovation_radius/b.gain_lower):
        raise ValueError('Invalid stopping horizon, geometry or braking authority')
    if (np.any(lo > hi) or np.any(lo[:, 1] < 0) or np.any(hi[:, 1] > b.vmax)
            or np.any(ml > mh) or np.any(ml < b.command_lower)
            or np.any(mh > b.command_upper) or np.any(hist < b.command_lower)
            or np.any(hist > b.command_upper)):
        raise ValueError('State, memory or command prefix outside the public domain')
    previous = history[-1] if history else np.zeros(3)
    slew = jerk*b.dt*b.substeps
    commands, tube = [], []
    checks = dict(separation=separation, zone_start=zone_start, zone_end=zone_end,
                  guard=geometry_guard)
    reason = 'horizon_exhausted'
    for stage in range(max_decisions):
        if stage:
            command = np.maximum(command-slew, b.command_lower)
        if (np.any(command < np.maximum(b.command_lower, previous-slew))
                or np.any(command > np.minimum(b.command_upper, previous+slew))):
            reason = 'candidate_command_or_slew'
            break
        commands.append(command.tolist())
        history.append(command.copy())
        qlo, qhi = delayed_interval(history, b.maximum_delay)
        for _ in range(b.substeps):
            nlo, nhi, nml, nmh = _substep(lo, hi, ml, mh, qlo, qhi, b)
            valid = route_box_safe(lo, hi, nlo, nhi, **checks)
            lo, hi, ml, mh = nlo, nhi, nml, nmh
            tube.append(dict(lower=lo.tolist(), upper=hi.tolist(),
                             memory_lower=ml.tolist(), memory_upper=mh.tolist()))
            if not valid:
                reason = 'route_box_not_certified'
                break
        if not valid:
            break
        # All commands still able to emerge from the FIFO, and the command to
        # be held forever, must keep memory in the stopped invariant set.
        limit = -b.innovation_radius/b.gain_lower
        pending = history[-b.maximum_delay:]
        if len(pending) < b.maximum_delay:
            pending = pending+[np.zeros(3)]
        invariant = (np.all(hi[:, 1] == 0.) and np.all(mh < limit-1e-10)
                     and np.all(np.asarray(pending) < limit-1e-10)
                     and np.all(command < limit-1e-10))
        if invariant:
            reason = 'stopped_invariant_set'
            break
        previous = command.copy()
    return dict(certified=reason == 'stopped_invariant_set', reason=reason,
                commands=commands, substeps=len(tube), tube=tube,
                terminal_hold=commands[-1] if commands else None,
                pending_command_limit=-b.innovation_radius/b.gain_lower,
                public_bounds=asdict(b), geometry=checks,
                numerically_verified_interval_arithmetic=False,
                performance_risk_certified=False, task_completion_certified=False,
                frozen_controller_modified=False)


class RetainedBrakePlan:
    """Transactional tape manager, independent of simulator and learned policy.

    Acknowledge only after EXACTLY one decision period was physically executed.
    The caller must enforce that timing/actuation contract. Rejection of a new
    proposal leaves the previous verified suffix intact. This is not yet wired
    to SUMO; acknowledgment is not independent proof of physical execution.
    """
    def __init__(self, executed_commands=(), **options):
        hist = np.asarray(executed_commands, dtype=float)
        if hist.size == 0 and hist.shape in ((0,), (0, 3)):
            hist = np.empty((0, 3))
        if hist.ndim != 2 or hist.shape[1:] != (3,) or not np.isfinite(hist).all():
            raise ValueError('Finite executed prefix required')
        self._history = [u.copy() for u in hist]
        self._options = deepcopy(options)
        self._suffix = []
        self._hold = None
        self._pending = None
        self._invalid = False
        self._generation = 0
        self._evidence = None

    def _require_valid(self):
        if self._invalid:
            raise RuntimeError('Execution acknowledgment violated the retained plan')

    def offer(self, lower, upper, memory_lower, memory_upper, candidate):
        self._require_valid()
        if self._pending is not None:
            raise RuntimeError('Acknowledge the pending decision before replanning')
        result = certify_braking_tube(lower, upper, memory_lower, memory_upper,
            self._history, candidate, **self._options)
        if result['certified']:
            self._suffix = [np.array(command) for command in result['commands']]
            self._hold = np.array(result['terminal_hold'])
            self._evidence = deepcopy(result)
            self._generation += 1
        return dict(accepted=result['certified'], reason=result['reason'],
                    retained_plan_available=self._hold is not None,
                    generation=self._generation, check=deepcopy(result))

    def next_command(self):
        self._require_valid()
        if self._pending is None:
            if self._hold is None:
                raise RuntimeError('No initial certified tape; initialization not authorized')
            self._pending = (self._suffix[0] if self._suffix else self._hold).copy()
        # Returning an independent copy prevents caller mutation of the tape.
        return self._pending.copy()

    def acknowledge(self, executed_command):
        self._require_valid()
        if self._pending is None:
            raise RuntimeError('No issued decision to acknowledge')
        try:
            actual = _array(executed_command, (3,))
        except ValueError:
            self._invalid = True
            raise
        if not np.array_equal(actual, self._pending):
            self._invalid = True
            raise RuntimeError('Executed command differs from the certified tape')
        self._history.append(actual)
        if self._suffix:
            self._suffix.pop(0)
        self._pending = None

    def snapshot(self):
        return dict(generation=self._generation, invalidated=self._invalid,
                    acknowledged_decisions=len(self._history),
                    remaining_commands=[u.tolist() for u in self._suffix],
                    terminal_hold=None if self._hold is None else self._hold.tolist(),
                    pending=None if self._pending is None else self._pending.tolist(),
                    certificate=deepcopy(self._evidence))
