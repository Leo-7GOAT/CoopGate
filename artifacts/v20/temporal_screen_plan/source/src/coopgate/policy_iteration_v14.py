"""Training-only rollout policy iteration; deployment is a local neural actor.

This is model-based approximate policy iteration, not a new model-free RL
algorithm. The behavior policy generates visited states. Complete action
returns improve that policy, and a neural apprentice learns cost-sensitive
action preferences. Rollouts are never called by the deployed actor.
"""
import numpy as np
import torch
from torch import nn
from numba import njit

from .ego_v13 import ego_tail
from .marl_v12 import LEVELS, project_joint
from .single_v12 import constant_proposal
from .fast_v8 import advance_one
from .learned_rollout_v13 import actor_action, learned_tail


def kinematic_features(obs):
    p, v = obs[..., 0] * 150., obs[..., 1] * 20.
    own_eta = (-25. - p).clamp(min=0) / v.clamp(min=1.)
    extra = []
    for start in (6, 12):
        present = obs[..., start]
        distance = obs[..., start + 1] * 150.
        nv = (v + obs[..., start + 2] * 20.).clamp(min=1.)
        eta = (-25. - p - distance).clamp(min=0) / nv
        extra.extend((present * (eta - own_eta).clamp(-10, 10) / 10,
                      present * distance / (v.clamp(min=1.) * 10),
                      present * (distance.abs() < 35).float(),
                      present * (eta < own_eta).float()))
    return torch.cat((obs, torch.stack(extra, dim=-1)), dim=-1)


class IteratedPolicy(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(26, width), nn.ReLU(),
                                     nn.Linear(width, width), nn.ReLU())
        self.advantage = nn.Linear(width, len(LEVELS))

    def forward(self, obs):
        return self.advantage(self.encoder(kinematic_features(obs)))

    @torch.no_grad()
    def proposal(self, obs, mask):
        logits = self(torch.as_tensor(obs, dtype=torch.float32))
        return logits.masked_fill(~torch.as_tensor(mask, dtype=torch.bool), -1e9).argmax(-1).numpy()

    def compiled_weights(self):
        keys = ('encoder.0.weight', 'encoder.0.bias', 'encoder.2.weight',
                'encoder.2.bias', 'advantage.weight', 'advantage.bias')
        return tuple(self.state_dict()[k].detach().cpu().numpy().astype(np.float32) for k in keys)


def load_policy(path):
    meta = torch.load(path, map_location='cpu', weights_only=True)
    if meta['algorithm'] != 'rollout_policy_iteration_ego_v14':
        raise ValueError('Not a V14 approximate policy iteration checkpoint')
    policy = IteratedPolicy(meta['width'])
    policy.load_state_dict(meta['model'])
    policy.eval()
    return policy, meta


@njit(cache=True, fastmath=False)
def action_returns(x, prev, step, packed, weights, kind, use_learned=True):
    """Exact nominal complete-horizon costs; all model branches are counted.

    A rule tail is an explicitly declared initialization/stabilization policy.
    The learned tail is the preceding iteration's network, fixed for the
    entire data collection. No labels depend on a held-out trajectory.
    """
    costs = np.full(len(LEVELS), np.inf)
    transitions = 0
    for a in range(len(LEVELS)):
        if abs(LEVELS[a] - prev[1]) > 2.5 + 1e-8:
            continue
        proposal = constant_proposal(prev, packed)
        proposal[1] = LEVELS[a]
        command, _, valid = project_joint(x, prev, proposal, packed, .5)
        if not valid:
            raise RuntimeError('Training action outside shared safety contract')
        nxt = x.copy()
        if not advance_one(nxt, command, packed):
            raise RuntimeError('Unsafe training branch')
        transitions += 1
        rule, rule_time, _ = ego_tail(nxt, command, step + 1, packed, .5, .05)
        transitions += round(rule_time / .5)
        tail = rule
        if use_learned:
            # Count the evaluated learned trajectory without early pruning.
            follow = nxt.copy()
            last = command.copy()
            learned = 0.
            for k in range(step + 1, 50):
                if follow[1, 0] >= packed[11]:
                    break
                p = constant_proposal(last, packed)
                p[1] = actor_action(follow, last, k, weights, True, kind)
                u, _, ok = project_joint(follow, last, p, packed, .5)
                if not ok:
                    raise RuntimeError('Unsafe learned-tail branch')
                learned += .5 + .05 * abs(u[1] - last[1])
                if not advance_one(follow, u, packed):
                    raise RuntimeError('Unsafe learned-tail transition')
                transitions += 1
                last = u
            tail = min(tail, learned)
        costs[a] = .5 + .05 * abs(command[1] - prev[1]) + tail
    return costs, transitions


@njit(cache=True, fastmath=False)
def batch_returns(states, previous, steps, packed, weights, kind, use_learned=True):
    result = np.empty((len(states), len(LEVELS)))
    transitions = 0
    for i in range(len(states)):
        values, count = action_returns(states[i], previous[i], steps[i], packed,
                                       weights, kind, use_learned)
        result[i] = values
        transitions += count
    return result, transitions
