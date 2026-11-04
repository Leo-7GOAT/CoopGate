"""Observed-velocity projection for a separate OOD engineering probe.

This does not restore exact-state or actuator-model assumptions. It is not used
by any frozen formal study. Nonfinite observations still fail explicitly.
"""
import numpy as np

from .recovery_joint_gate_v35 import RecoveryJointGate


def project_velocity(observed, vmax=20.):
    observed = np.asarray(observed, dtype=float)
    if observed.shape != (3, 2) or not np.isfinite(observed).all():
        raise ValueError('A finite three-vehicle position/velocity observation is required')
    if not np.isfinite(vmax) or vmax <= 0.:
        raise ValueError('A finite positive physical velocity bound is required')
    projected = observed.copy()
    projected[:, 1] = np.clip(observed[:, 1], 0., vmax)
    return projected


class ProjectedObservationGate:
    """Give every baseline/learned rule the same bounded observed velocities."""
    def __init__(self, model, profile, rule='bank_performance'):
        self.inner = RecoveryJointGate(model, profile, rule)
        self.c = self.inner.c

    def select(self, observed, previous, unused=None):
        raw = np.asarray(observed, dtype=float)
        projected = project_velocity(raw, self.c.vmax)
        command, original_info = self.inner.select(projected, previous, unused)
        info = dict(original_info)
        info['velocity_projection'] = dict(applied=bool(np.any(projected[:, 1] != raw[:, 1])),
            observed_velocity=raw[:, 1].tolist(), projected_velocity=projected[:, 1].tolist(),
            physical_velocity_bounds=[0., float(self.c.vmax)],
            restores_physical_certificate=False)
        return command, info
