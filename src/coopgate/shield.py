"""Finite-horizon model checks; scores are NOT calibrated safety probabilities.

MAMPS-style is an adaptation of the monotone mode-switching heuristic. This
implementation lacks its invariant terminal-set proof and must not inherit it.
"""
from dataclasses import dataclass
from itertools import product
import time

import numpy as np

from .controllers import Intent, backup_intents


@dataclass
class GateResult:
    intents: list
    modes: np.ndarray
    score: float
    margin: float
    checks: int
    emergency: bool
    elapsed_ms: float


def rollout_margin(states, intents, cfg, background_accel=0.0):
    """Approximate candidate commitment followed by rule-backup rollout.

    Return per-agent signed geometric clearance. Include current geometry,
    lane-change swept occupancy (expanded lateral footprint), and taper limit.
    Uncontrolled cars use a fixed acceleration hypothesis, not oracle futures.
    """
    x = np.array(states, dtype=float, copy=True)
    n, dt = cfg.n_agents, cfg.decision_dt
    margins = np.full(n, 100.0)
    remaining = np.zeros(n)
    destinations = x[:n, 1].copy()
    for t in range(cfg.gate_horizon + 1):
        for i in range(n):
            if not x[i, 4]:
                continue
            if x[i, 0] > cfg.merge_end-3 and x[i, 1] < cfg.lane_width-1.0:
                margins[i] = min(margins[i], cfg.merge_end-3-x[i, 0])
            for j in range(len(x)):
                if i == j or not x[j, 4]:
                    continue
                if abs(x[i, 1]-x[j, 1]) < cfg.vehicle_width + 0.6:
                    clearance = abs(x[i, 0]-x[j, 0])-cfg.vehicle_length-cfg.safety_gap
                    margins[i] = min(margins[i], clearance-cfg.prediction_error*t*dt)
        if t == cfg.gate_horizon:
            break
        controls = intents if t == 0 else backup_intents(x, cfg)
        for i in range(len(x)):
            if not x[i, 4]:
                continue
            if i < n:
                # Once begun, maintain a lane change across planning cycles.
                if remaining[i] <= 0:
                    destinations[i] = controls[i].lane * cfg.lane_width
                    if abs(destinations[i]-x[i, 1]) > 0.1:
                        remaining[i] = cfg.lane_change_seconds
                if remaining[i] > 0:
                    x[i, 1] += (destinations[i]-x[i, 1]) * min(1, dt/remaining[i])
                    remaining[i] = max(0, remaining[i]-dt)
                a = np.clip((controls[i].speed-x[i, 2])/max(dt, 0.8), -cfg.max_brake, cfg.max_accel)
            else:
                a = background_accel
            v_next = np.clip(x[i, 2]+a*dt, 0, cfg.max_speed)
            x[i, 0] += 0.5*(x[i, 2]+v_next)*dt
            x[i, 2] = v_next
    return margins


class Supervisor:
    methods = ('traditional', 'mappo', 'independent', 'mamps_style', 'joint_nominal', 'joint_robust')

    def __init__(self, cfg, method='joint_robust'):
        if method not in self.methods:
            raise ValueError(method)
        self.cfg, self.method = cfg, method
        self.previous = np.zeros(cfg.n_agents, dtype=bool)

    def select(self, states, learned):
        start = time.perf_counter()
        c, n = self.cfg, self.cfg.n_agents
        backup = backup_intents(states, c)
        checks = 0

        def evaluate(mask, robust=False):
            nonlocal checks
            candidate = [learned[i] if mask[i] else backup[i] for i in range(n)]
            hypotheses = (-c.background_brake, 0, 1.5) if robust else (0,)
            margins = np.full(n, np.inf)
            for a in hypotheses:
                checks += 1
                margins = np.minimum(margins, rollout_margin(states, candidate, c, a))
            score = float(1/(1+np.exp(-np.clip(np.min(margins)/2, -50, 50))))
            return candidate, margins, score

        emergency = False
        if self.method == 'traditional':
            mask = np.zeros(n, bool)
            candidate, margins, score = backup, np.zeros(n), 0.0
        elif self.method == 'mappo':
            mask = np.ones(n, bool)
            candidate, margins, score = learned, np.zeros(n), 0.0
        elif self.method == 'independent':
            mask = np.ones(n, bool)
            _, local_margins, _ = evaluate(mask)
            # Deliberately independent baseline: no recheck after others switch.
            thresholds = np.where(self.previous, c.gate_threshold, c.gate_return_threshold)
            mask = local_margins >= 2*np.log(thresholds/(1-thresholds))
            candidate = [learned[i] if mask[i] else backup[i] for i in range(n)]
            margins, score = local_margins, float(np.min(1/(1+np.exp(-np.clip(local_margins/2,-50,50)))))
        elif self.method == 'mamps_style':
            mask = np.ones(n, bool)
            for _ in range(n+2):
                candidate, margins, score = evaluate(mask)
                threshold = c.gate_return_threshold if np.any(mask & ~self.previous) else c.gate_threshold
                required = 2*np.log(threshold/(1-threshold)) if mask.any() else 0
                if np.min(margins) >= required:
                    break
                new = mask & (margins >= required)
                if np.array_equal(new, mask):
                    mask = np.zeros(n, bool)
                    candidate, margins, score = evaluate(mask)
                    break
                mask = new
            emergency = np.min(margins) < 0
        else:
            robust = self.method == 'joint_robust'
            best = None
            best_value = -np.inf
            for bits in product((False, True), repeat=n):
                bits = np.array(bits)
                candidate_i, margins_i, score_i = evaluate(bits, robust)
                threshold = c.gate_return_threshold if np.any(bits & ~self.previous) else c.gate_threshold
                # All-backup remains a candidate but must pass the geometry check.
                accepted = np.min(margins_i) >= 0 and (not bits.any() or score_i >= threshold)
                objective = bits.sum() - c.switch_penalty*np.count_nonzero(bits != self.previous)
                objective += 0.001*sum(v.speed for v in candidate_i)
                if accepted and objective > best_value:
                    best_value = objective
                    best = (bits, candidate_i, margins_i, score_i)
            if best is None:
                mask = np.zeros(n, bool)
                candidate, margins, score = evaluate(mask, robust)
                emergency = True
            else:
                mask, candidate, margins, score = best
        if emergency:
            # Minimum-risk attempt, explicitly not reported as a safe certificate.
            candidate = [Intent(0.0, int(s[1] > c.lane_width/2)) for s in states[:n]]
        self.previous = mask.copy()
        return GateResult(candidate, mask, score, float(np.min(margins)), checks,
                          bool(emergency), (time.perf_counter()-start)*1000)
