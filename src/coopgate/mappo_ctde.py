"""On-policy feed-forward MAPPO, separate from the frozen PI experiments.

The proposal policy is decentralized. The physical filter is not. All PPO
likelihoods are for sampled categorical proposals, before physical filtering.
"""
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .cooperative_v15 import CooperativeVector
from .marl_v12 import LEVELS, OBS_DIM, STATE_DIM


@dataclass(frozen=True)
class PPOConfig:
    width: int = 128
    n_envs: int = 64
    rollout: int = 64
    epochs: int = 5
    minibatch: int = 1024
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 1.0
    gae_lambda: float = .95
    clip: float = .2
    value_scale: float = 10.
    entropy: float = .01
    max_grad_norm: float = .5
    kinematic_features: bool = False
    accelerated_features: bool = False
    phase_actor: bool = False
    initial_actor_lr: float = 3e-4
    startup_width: int = 0
    startup_relu: bool = False


def local_kinematic_features(obs):
    """Eight deterministic functions of the SAME local observation.

    Relative ETA and time headway improve representation, not information:
    no planner, optimal-action labels, hidden state or rollout is evaluated.
    """
    position, speed = obs[..., 0]*150., obs[..., 1]*20.
    own_eta = (-25.-position).clamp(min=0.) / speed.clamp(min=1.)
    extra=[]
    for start in (6, 12):
        visible=obs[..., start]
        distance=obs[..., start+1]*150.
        neighbor_speed=(speed+obs[..., start+2]*20.).clamp(min=1.)
        eta=(-25.-position-distance).clamp(min=0.)/neighbor_speed
        extra.extend((visible*(eta-own_eta).clamp(-10., 10.)/10.,
                      visible*distance/(speed.clamp(min=1.)*10.),
                      visible*(distance.abs()<35.).float(),
                      visible*(eta<own_eta).float()))
    return torch.cat((obs, torch.stack(extra, dim=-1)), dim=-1)


def accelerating_arrival(distance, speed):
    """Analytic constant-a=2.5 arrival with speed cap 20; no policy rollout."""
    distance=distance.clamp(min=0.)
    speed=speed.clamp(min=0.,max=20.)
    accelerate_time=(20.-speed)/2.5
    accelerate_distance=(400.-speed.square())/5.
    hit=(torch.sqrt(speed.square()+5.*distance)-speed)/2.5
    cruise=accelerate_time+(distance-accelerate_distance)/20.
    return torch.where(distance<=accelerate_distance,hit,cruise)


def accelerated_interval_features(obs):
    position,speed=obs[...,0]*150.,obs[...,1]*20.
    entry=accelerating_arrival(-25.-position,speed)
    leave=accelerating_arrival(6.8-position,speed)
    goal=accelerating_arrival(60.-position,speed)
    extra=[entry/10.,leave/10.]
    for start in (6,12):
        visible=obs[...,start]
        neighbor_position=position+obs[...,start+1]*150.
        neighbor_speed=speed+obs[...,start+2]*20.
        other_entry=accelerating_arrival(-25.-neighbor_position,neighbor_speed)
        other_leave=accelerating_arrival(6.8-neighbor_position,neighbor_speed)
        other_goal=accelerating_arrival(60.-neighbor_position,neighbor_speed)
        extra.extend([visible*(other_entry-entry)/10.,visible*(other_entry-leave)/10.,
                      visible*(other_leave-entry)/10.,visible*(other_goal-goal)/10.])
    return torch.stack(extra,dim=-1)


def mlp(inputs, width, outputs, output_gain, relu=False):
    activation=nn.ReLU if relu else nn.Tanh
    net = nn.Sequential(nn.Linear(inputs, width), activation(),
                        nn.Linear(width, width), activation(), nn.Linear(width, outputs))
    for layer in net:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, np.sqrt(2.))
            nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(net[-1].weight, output_gain)
    return net


class PhaseActor(nn.Module):
    """Two local heads, routed by the public decision clock, not global state."""
    def __init__(self,inputs,width,startup_width=0,startup_relu=False):
        super().__init__()
        self.initial=mlp(inputs,startup_width or width,len(LEVELS),.01,relu=startup_relu)
        self.flow=mlp(inputs,width,len(LEVELS),.01)

    def forward(self,features):
        initial=features[...,5].abs()<1e-8
        return torch.where(initial[...,None],self.initial(features),self.flow(features))


class MAPPOCTDE(nn.Module):
    """One local actor shared across roles; one scalar common-payoff critic.

    IPPO is an information ablation with the same network count and width.
    Its shared local critic predicts three values from three local inputs.
    """
    def __init__(self, config=None, critic_kind='central'):
        super().__init__()
        self.config = config or PPOConfig()
        if critic_kind not in ('central', 'local'):
            raise ValueError('critic_kind must be central or local')
        if self.config.accelerated_features and not self.config.kinematic_features:
            raise ValueError('Accelerated representation extends kinematic representation')
        self.critic_kind = critic_kind
        actor_dim=OBS_DIM+8*int(self.config.kinematic_features)+10*int(self.config.accelerated_features)
        critic_dim=(STATE_DIM+3*(actor_dim-OBS_DIM)) if critic_kind=='central' else actor_dim
        self.actor = PhaseActor(actor_dim,self.config.width,self.config.startup_width,self.config.startup_relu) if self.config.phase_actor else mlp(actor_dim,self.config.width,len(LEVELS),.01)
        self.critic = mlp(critic_dim,
                          self.config.width, 1, 1.)

    def features(self,observations):
        features=local_kinematic_features(observations) if self.config.kinematic_features else observations
        if self.config.accelerated_features:
            features=torch.cat((features,accelerated_interval_features(observations)),dim=-1)
        return features

    def distribution(self, observations, mask):
        if observations.shape[-1] != OBS_DIM or mask.shape != observations.shape[:-1] + (len(LEVELS),):
            raise ValueError('Local observations/action masks have incompatible shapes')
        if not mask.any(-1).all():
            raise ValueError('Empty categorical action support')
        features=self.features(observations)
        logits = self.actor(features)
        if not torch.isfinite(logits).all():
            raise ValueError('Nonfinite actor logits')
        return Categorical(logits=logits.masked_fill(~mask, -1e9))

    def value(self, observations, state):
        if self.critic_kind == 'central':
            if self.config.kinematic_features:
                extra=self.features(observations)[..., 18:].flatten(start_dim=-2)
                state=torch.cat((state, extra), dim=-1)
            return self.config.value_scale * self.critic(state).expand(*state.shape[:-1], 3)
        features=self.features(observations)
        return self.config.value_scale * self.critic(features).squeeze(-1)

    @torch.no_grad()
    def proposal(self, observations, mask):
        device = next(self.parameters()).device
        obs = torch.as_tensor(observations, dtype=torch.float32, device=device)
        mask = torch.as_tensor(mask, dtype=torch.bool, device=device)
        return self.distribution(obs, mask).logits.argmax(-1).cpu().numpy()


class PPOVector(CooperativeVector):
    """Existing physical simulator; single stage freezes the two other roles.

    Reuses the common original shield and semi-implicit transition exactly.
    It does not claim to simulate actuator mismatch in this development stage.
    """
    def __init__(self, n=64, seed=1, stage='joint'):
        if stage not in ('single', 'joint'):
            raise ValueError('Unknown training stage')
        self.stage = stage
        super().__init__(n, seed)

    def training_mask(self, active):
        selected = np.array(active, dtype=bool, copy=True)
        if self.stage == 'single':
            selected[:, (0, 2)] = False
        return selected

    def execute(self, sampled, mask):
        actions = np.array(sampled, copy=True)
        if self.stage == 'single':
            fixed = np.where(mask, np.arange(len(LEVELS)), -1).max(-1)
            actions[:, (0, 2)] = fixed[:, (0, 2)]
        return actions

    def step(self, actions):
        # Single stage optimizes the ramp cost; joint stage the common team cost.
        before = self.x.copy()
        previous = self.previous.copy()
        following, reward, boundary, active, finished = super().step(actions)
        if self.stage == 'single':
            # following inputs are captured BEFORE parent reset, unlike self.x.
            after_position = following[0][:, 1, 0] * 150.
            after_command = following[0][:, 1, 2] * 5.
            stage = active[:, 1] * (.5 + .05 * np.abs(after_command - previous[:, 1]))
            potential = -np.maximum(self.c.goal-before[:, 1, 0], 0.) / self.c.vmax
            future = -np.maximum(self.c.goal-after_position, 0.) / self.c.vmax
            future[boundary] = 0.
            reward = (-stage + future - potential).astype(np.float32)
        return following, reward, boundary, self.training_mask(active), finished


def gae(rewards, values, next_values, terminals, boundaries, gamma=1., lam=.95):
    """Terminals mask bootstrap; boundaries separately stop inter-episode GAE.

    A rollout cutoff is not a terminal or reset. Time-limit truncation may
    bootstrap in continuing tasks; this experiment's finite horizon terminates.
    """
    rewards, values, next_values = map(np.asarray, (rewards, values, next_values))
    if values.shape != next_values.shape or rewards.shape != values.shape[:2]:
        raise ValueError('GAE expects rewards[T,E] and values[T,E,A]')
    adv = np.zeros_like(values, dtype=np.float32)
    carry = np.zeros_like(values[0], dtype=np.float32)
    for t in range(len(rewards)-1, -1, -1):
        delta = rewards[t, :, None] + gamma * (~terminals[t, :, None]) * next_values[t] - values[t]
        carry = delta + gamma * lam * (~boundaries[t, :, None]) * carry
        adv[t] = carry
    return adv, adv + values


def losses(model, batch, config):
    active = batch['active']
    if not active.any():
        raise ValueError('No active actor samples')
    dist = model.distribution(batch['obs'], batch['mask'])
    log_ratio = dist.log_prob(batch['actions']) - batch['old_logp']
    ratio = log_ratio.exp()
    adv = batch['advantage']
    objective = torch.minimum(ratio * adv, ratio.clamp(1-config.clip, 1+config.clip) * adv)
    actor = -objective[active].mean()
    value = model.value(batch['obs'], batch['state']) / config.value_scale
    old = batch['old_value'] / config.value_scale
    target = batch['returns'] / config.value_scale
    clipped = old + (value-old).clamp(-config.clip, config.clip)
    # Huber regression limits the initial cost-scale transient; PPO value clipping.
    raw_error = nn.functional.smooth_l1_loss(value, target, reduction='none')
    clip_error = nn.functional.smooth_l1_loss(clipped, target, reduction='none')
    critic = torch.maximum(raw_error, clip_error).mean()
    entropy = dist.entropy()[active].mean()
    total = actor + .5 * critic - config.entropy * entropy
    if not torch.isfinite(total):
        raise FloatingPointError('Nonfinite PPO loss')
    with torch.no_grad():
        kl = ((ratio-1)-log_ratio)[active].mean()
        clip_fraction = ((ratio-1).abs() > config.clip)[active].float().mean()
    return total, dict(actor=actor, critic=critic, entropy=entropy,
                      approximate_kl=kl, clip_fraction=clip_fraction)


def load_mappo(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['algorithm'] not in ('mappo_ctde_20261101', 'ippo_local_20261101'):
        raise ValueError('Not a new on-policy PPO checkpoint')
    model = MAPPOCTDE(PPOConfig(**checkpoint['ppo_config']), checkpoint['critic_kind'])
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, checkpoint
