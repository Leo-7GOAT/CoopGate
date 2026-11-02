"""Longitudinal constrained MPC and a transparent rule-based backup planner.

Lateral motion is executed by SUMO's continuous lane-change model, not by MPC.
The backup is a heuristic candidate; it is never assumed universally safe.
"""
from dataclasses import dataclass

import numpy as np
import osqp
from scipy import sparse

from .config import ACTION_SPEEDS, Config


@dataclass(frozen=True)
class Intent:
    speed: float
    lane: int


def decode_action(action, lane, x, cfg):
    target_lane = int(np.clip(lane + (int(action) // 3 - 1), 0, 1))
    if x >= cfg.merge_end:
        target_lane = 1  # logical lane 1 continues downstream
    return Intent(ACTION_SPEEDS[int(action) % 3], target_lane)


def backup_intents(states, cfg):
    """Yield to existing main-lane traffic; merge only into a clear gap."""
    result = []
    for i, s in enumerate(states[: cfg.n_agents]):
        if not s[4]:
            result.append(Intent(0.0, 1))
            continue
        x, y, v = s[:3]
        lane = int(y > cfg.lane_width / 2)
        target = cfg.desired_speed
        ahead = [o for j, o in enumerate(states) if j != i and o[4]
                 and abs(o[1] - y) < 2.2 and o[0] > x]
        if ahead:
            lead = min(ahead, key=lambda o: o[0])
            gap = lead[0] - x - cfg.vehicle_length
            target = min(target, max(0.0, lead[2] + 0.55 * (gap - 5.0 - 0.9 * v)))
        if lane == 0 and x < cfg.merge_end:
            clear = all(abs(o[0] - x) > cfg.vehicle_length + 5.0 + 0.65 * max(v, o[2])
                        for o in states if o[4] and o[1] > 1.6)
            if clear and x > 45:
                lane = 1
            else:
                # Stop before the taper if no acceptable main-lane gap exists.
                target = min(target, np.sqrt(max(0, 2 * 2.5 * (cfg.merge_end - x - 15))))
                target = min(target, 12.0)
        result.append(Intent(float(target), lane))
    return result


class SpeedMPC:
    """Condensed QP over acceleration: speed tracking + effort + jerk.

    Hard acceleration, speed and slew constraints; emergency braking explicitly
    overrides comfort slew bounds when selected by the supervisor.
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        n, dt = cfg.mpc_horizon, cfg.decision_dt
        self.B = np.tril(np.ones((n, n))) * dt
        self.D = np.eye(n) - np.eye(n, k=-1)
        p = 2 * (self.B.T @ self.B + 0.1 * np.eye(n) + 0.3 * self.D.T @ self.D)
        self.A = sparse.vstack([sparse.eye(n), sparse.csc_matrix(self.B), sparse.csc_matrix(self.D)]).tocsc()
        self.solver = osqp.OSQP()
        self.solver.setup(P=sparse.csc_matrix(np.triu(p)), q=np.zeros(n), A=self.A,
                          l=np.full(3*n, -np.inf), u=np.full(3*n, np.inf),
                          verbose=False, eps_abs=1e-5, eps_rel=1e-5, polishing=False)
        self.failures = 0

    def control(self, speed, target, previous=0.0, emergency=False):
        c = self.cfg
        if emergency:
            return -c.max_brake if speed > 0.01 else 0.0
        n = c.mpc_horizon
        e = np.zeros(n)
        e[0] = previous
        q = 2 * (self.B.T @ np.full(n, speed-target) - 0.3 * self.D.T @ e)
        slew = c.max_jerk * c.decision_dt
        low = np.r_[np.full(n, -c.max_brake), np.full(n, -speed), e-slew]
        high = np.r_[np.full(n, c.max_accel), np.full(n, c.max_speed-speed), e+slew]
        self.solver.update(q=q, l=low, u=high)
        out = self.solver.solve(raise_error=False)
        if out.info.status_val not in (1, 2) or out.x is None or not np.isfinite(out.x[0]):
            self.failures += 1
            return float(np.clip((target-speed) / c.decision_dt, -c.max_brake, c.max_accel))
        return float(np.clip(out.x[0], -c.max_brake, c.max_accel))
