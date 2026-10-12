"""Parameter-shared categorical actor; centralized team-value critic.

PPO clips per-agent likelihood ratios; a team GAE is broadcast to agents.
Bootstrapping distinguishes time-limit truncation from true terminal states.
Training is on the unshielded environment; deployment compares fixed weights.
"""
from pathlib import Path
import csv
import json
import random

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .config import Config, OBS_DIM, N_ACTIONS
from .sim import SumoEnv


def network(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs,64), nn.Tanh(), nn.Linear(64,64), nn.Tanh(), nn.Linear(64,outputs))


class MAPPO(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = network(OBS_DIM, N_ACTIONS)
        self.critic = network(3*OBS_DIM, 1)

    def distribution(self, observations):
        return Categorical(logits=self.actor(observations))

    def value(self, observations):
        return self.critic(observations.flatten(start_dim=-2)).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        t = torch.as_tensor(obs, dtype=torch.float32)
        dist = self.distribution(t)
        a = dist.probs.argmax(-1) if deterministic else dist.sample()
        return a.numpy(), dist.log_prob(a).numpy(), float(self.value(t))


def compute_gae(rewards, values, next_values, terminals, boundaries, gamma=0.99, lam=0.95):
    adv = np.zeros(len(rewards), np.float32)
    carry = 0.0
    for i in reversed(range(len(rewards))):
        delta = rewards[i]+gamma*(1-terminals[i])*next_values[i]-values[i]
        carry = delta+gamma*lam*(1-boundaries[i])*carry
        adv[i] = carry
    return adv, adv+values


def train(out, total_steps=20000, seed=7, rollout=256, epochs=5, cfg=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    cfg = cfg or Config()
    model = MAPPO()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    episodes, rows, returns = 0, [], []
    total, episode_return = 0, 0.0
    with SumoEnv(cfg, method='mappo') as env:
        obs = env.reset(seed)
        while total < total_steps:
            batch = {k:[] for k in ('obs','actions','logp','rewards','values','next_values','terminals','boundaries')}
            for _ in range(min(rollout,total_steps-total)):
                a, lp, val = model.act(obs)
                next_obs, r, terminal, truncated, info = env.step(a)
                with torch.no_grad():
                    nv = float(model.value(torch.as_tensor(next_obs)))
                for k,v in zip(batch, (obs,a,lp,r,val,nv,terminal,terminal or truncated)):
                    batch[k].append(v)
                episode_return += r
                obs = next_obs
                total += 1
                if terminal or truncated:
                    episodes += 1
                    returns.append(episode_return)
                    episode_return = 0.0
                    scenario = ('nominal','dense','braking')[episodes % 3]
                    obs = env.reset(seed+episodes, scenario)
            b = {k:np.asarray(v) for k,v in batch.items()}
            adv, ret = compute_gae(b['rewards'], b['values'], b['next_values'], b['terminals'], b['boundaries'])
            adv = (adv-adv.mean())/(adv.std()+1e-8)
            tensors = {k:torch.as_tensor(b[k], dtype=torch.long if k=='actions' else torch.float32)
                       for k in ('obs','actions','logp')}
            adv_t, ret_t = torch.as_tensor(adv), torch.as_tensor(ret, dtype=torch.float32)
            for _ in range(epochs):
                permutation = torch.randperm(len(adv))
                for ids in permutation.split(64):
                    dist = model.distribution(tensors['obs'][ids])
                    ratios = torch.exp(dist.log_prob(tensors['actions'][ids])-tensors['logp'][ids])
                    a = adv_t[ids,None]
                    actor_loss = -torch.minimum(ratios*a, ratios.clamp(0.8,1.2)*a).mean()
                    value_loss = (model.value(tensors['obs'][ids])-ret_t[ids]).square().mean()
                    loss = actor_loss+0.5*value_loss-0.01*dist.entropy().mean()
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(),0.5)
                    optimizer.step()
            row = {'steps':total,'episodes':episodes,'mean_last20_return':float(np.mean(returns[-20:])) if returns else 0,
                   'actor_loss':float(actor_loss.detach()),'value_loss':float(value_loss.detach())}
            rows.append(row)
            if len(rows)%5 == 0 or total == total_steps:
                print(json.dumps(row), flush=True)
            torch.save({'model':model.state_dict(),'seed':seed,'steps':total,'config':cfg.dict()}, out/'policy.pt')
            with (out/'training.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(row)); w.writeheader(); w.writerows(rows)
    (out/'config.json').write_text(json.dumps({'training_seed':seed,'steps':total,'episodes':episodes,'config':cfg.dict()},indent=2))
    return model


def load_policy(path):
    torch.set_num_threads(2)
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    model=MAPPO()
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, checkpoint
