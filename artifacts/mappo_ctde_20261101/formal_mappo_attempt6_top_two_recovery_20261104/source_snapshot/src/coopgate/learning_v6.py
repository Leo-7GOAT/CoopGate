"""Centralized high-level contextual-bandit learning with a performance gate.

The action selects one V5 feedback schedule for a three-car episode. This is
full-information offline bandit optimization, not decentralized MARL or PPO.
Safety is still supplied by the independent joint braking certificate.
"""
import math
import time

import numpy as np
import torch
from torch import nn

from .merge_v4 import admissible, backup, certify
from .planning_v5 import BITS, portfolio


def features(states):
    states = np.asarray(states)
    p, v = states[..., 0], states[..., 1]
    gaps = np.stack((p[..., 0]-p[..., 1], p[..., 0]-p[..., 2], p[..., 1]-p[..., 2]), axis=-1)
    eta = np.maximum(-25.-p, 0)/np.maximum(v, 1.)
    return np.concatenate((p/150., v/20., gaps/50., eta/10.), axis=-1).astype(np.float32)


class Selector(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(12, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh())
        self.actor = nn.Linear(128, 37)
        self.cost = nn.Linear(128, 37)

    def forward(self, x):
        h = self.encoder(x)
        return self.actor(h), self.cost(h)


def predict(models, states):
    """Ensemble selection; cost predictions concern excess arrival time vs base."""
    x = torch.as_tensor(features(states))
    with torch.no_grad():
        outputs = [model(x) for model in models]
        probabilities = torch.stack([a.softmax(-1) for a, _ in outputs]).mean(0)
        costs = torch.stack([v for _, v in outputs]).mean(0)
        action = probabilities.argmax(-1)
        # Anchor baseline excess at exactly zero, including inference roundoff.
        delta = costs-costs[..., :1]
        selected = delta.gather(-1, action[..., None]).squeeze(-1)
    return action.numpy(), selected.numpy()


def load_models(directory, seeds, checkpoint='policy.pt'):
    from pathlib import Path
    torch.set_num_threads(1)
    models = []
    for seed in seeds:
        data = torch.load(Path(directory)/f'seed{seed}'/checkpoint, map_location='cpu', weights_only=True)
        if data.get('benchmark') != 'merge_v6_bandit':
            raise ValueError('Not a V6 high-level checkpoint')
        model = Selector(); model.load_state_dict(data['model']); model.eval(); models.append(model)
    return models


def upper_quantile(residuals, alpha):
    """Split-conformal upper rank, including the required +infinity endpoint."""
    residuals = np.asarray(residuals, dtype=float)
    if residuals.ndim != 1 or not len(residuals) or not np.isfinite(residuals).all():
        raise ValueError('Finite, nonempty one-dimensional calibration scores required')
    if not 0 < alpha < 1:
        raise ValueError('alpha must be in (0,1)')
    rank = math.ceil((len(residuals)+1)*(1-alpha))
    return float(np.sort(residuals)[rank-1]) if rank <= len(residuals) else float('inf')


def decide(models, state, quantile=None):
    action, delta = predict(models, state[None])
    proposal, estimate = int(action[0]), float(delta[0])
    upper = estimate+quantile if quantile is not None else None
    accepted = quantile is None or upper <= 0.
    return (proposal if accepted else 0), {'proposal': proposal, 'estimated_excess_s': estimate,
                                           'upper_excess_s': upper, 'accepted': bool(accepted)}


class ProfileGate:
    """Execute a selected schedule in feedback; no full-horizon online search."""
    def __init__(self, config, index):
        if not 0 <= int(index) < 37:
            raise ValueError(index)
        self.c, self.profile = config, portfolio()[int(index)]
        self.step = 0

    def select(self, x, previous, ignored_proposal):
        start = time.perf_counter(); c = self.c
        raw = np.full(3, c.accel)
        if self.step < self.profile['coast_steps']:
            raw[np.arange(3) != self.profile['first']] = 0.
        raw[x[:, 0] >= c.goal] = c.accel
        self.step += 1
        proposal = admissible(raw, previous, c); b = backup(previous, c)
        candidates = np.where(BITS, proposal, b)
        valid = certify(x, previous, candidates, c)
        if not valid.any():
            raise RuntimeError('Measured state has no certified candidate; outside supported contract')
        cost = np.sum(np.array(self.profile['weights'])*(candidates-proposal)**2, axis=1)
        cost -= 1e-8*candidates.sum(axis=1)
        command = candidates[int(np.argmin(np.where(valid, cost, np.inf)))]
        return command, {'in_set': bool(certify(x, previous, b, c)[0]), 'certified': True,
                         'intervention': float(np.mean(np.abs(command-proposal) > 1e-5)),
                         'gate_ms': (time.perf_counter()-start)*1000}
