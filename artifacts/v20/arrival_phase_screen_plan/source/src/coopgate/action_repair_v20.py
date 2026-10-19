"""Finite attenuation of a learned joint command, followed by joint projection.

These are candidates, not certified performance improvements. Each candidate
still needs the complete all-vehicle, all-path baseline-relative value check.
"""
import numpy as np

from .marl_v12 import project_joint
from .robust_v9 import robust_certificate


REPAIR_WEIGHTS = (1., .75, .5, .25)


def attenuated_candidates(state, previous, learned_command, baseline_command, packed):
    learned = np.asarray(learned_command, dtype=float)
    baseline = np.asarray(baseline_command, dtype=float)
    if learned.shape != (3,) or baseline.shape != (3,) or not np.isfinite(learned).all() or not np.isfinite(baseline).all():
        raise ValueError('Finite joint endpoint commands required')
    if not robust_certificate(state, previous, learned, packed, .5) or not robust_certificate(state, previous, baseline, packed, .5):
        raise ValueError('Both endpoints must satisfy the common physical contract')
    results, seen = [], [baseline]
    for weight in REPAIR_WEIGHTS:
        if weight == 1.:
            command = learned.copy()
        else:
            proposal = baseline + weight * (learned - baseline)
            command, _, valid = project_joint(state, previous, proposal, packed, .5)
            if not valid:
                raise RuntimeError('No physical projection for attenuated proposal')
        if not robust_certificate(state, previous, command, packed, .5):
            raise RuntimeError('Projected joint repair violates the physical contract')
        if any(np.array_equal(command, other) for other in seen):
            continue
        seen.append(command.copy())
        results.append((weight, command.copy()))
    return results
