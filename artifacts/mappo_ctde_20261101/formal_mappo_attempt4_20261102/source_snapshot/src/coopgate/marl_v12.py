"""Sequential CTDE MAPPO with ego-centric actors and a joint backup shield.

The actor consumes one vehicle's local observation, never the critic state.
The shield is centralized; only the learned proposal policy is decentralized.
Actions are proposed commands. PPO likelihoods must NOT use shielded commands.
"""
import time
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from numba import njit
from .merge_v4 import MergeConfig, BRANCH
from .fast_v8 import parameters, transition_values
from .robust_v9 import robust_certificate, validate

LEVELS = np.arange(-8, 5, dtype=np.float64)*.625
OBS_DIM = 18
STATE_DIM = 13
SENSOR_RANGE = 70.


def observations(x, previous, step, config=None, sensor_range=SENSOR_RANGE):
    """Own 6 features + up to 2 visible neighbors, sorted by relative distance.

    Neighbor fields: visibility, signed distance, relative speed, same route,
    relative distance to conflict entry, whether already beyond the goal.
    No neighbor command or hidden future disturbance is exposed.
    """
    c = config or MergeConfig()
    x, previous = np.asarray(x), np.asarray(previous)
    single = x.ndim == 2
    if single:
        x, previous = x[None], previous[None]
    steps = np.broadcast_to(np.asarray(step), (len(x),))
    result = np.zeros((len(x), 3, OBS_DIM), np.float32)
    for i in range(3):
        result[:,i,:6] = np.column_stack((x[:,i,0]/150., x[:,i,1]/c.vmax,
            previous[:,i]/5., np.full(len(x),BRANCH[i]),
            (x[:,i,0] >= c.goal).astype(float), steps/(c.horizon/c.decision_dt)))
        neighbors = [j for j in range(3) if j != i]
        rel = x[:,neighbors,0]-x[:,i,None,0]
        order = np.argsort(np.abs(rel), axis=1, kind='stable')
        for rank in range(2):
            j = np.asarray(neighbors)[order[:,rank]]
            rows = np.arange(len(x)); distance = x[rows,j,0]-x[:,i,0]
            visible = np.abs(distance) <= sensor_range
            feature = np.column_stack((np.ones(len(x)), distance/sensor_range,
                (x[rows,j,1]-x[:,i,1])/c.vmax, (BRANCH[j] == BRANCH[i]).astype(float),
                (x[rows,j,0]-c.zone_start)/150., (x[rows,j,0] >= c.goal).astype(float)))
            result[:,i,6+6*rank:12+6*rank] = feature*visible[:,None]
    return result[0] if single else result


def central_state(x, previous, step, config=None):
    c = config or MergeConfig()
    x, previous = np.asarray(x), np.asarray(previous)
    single = x.ndim == 2
    if single: x, previous = x[None], previous[None]
    states = np.concatenate((x[:,:,0]/150., x[:,:,1]/c.vmax, previous/5.,
        np.broadcast_to(BRANCH,(len(x),3)),
        np.broadcast_to(np.asarray(step),(len(x),))[:,None]/(c.horizon/c.decision_dt)),axis=1)
    return states.astype(np.float32)[0] if single else states.astype(np.float32)


def action_mask(previous, config=None):
    c = config or MergeConfig()
    return np.abs(LEVELS-np.asarray(previous)[...,None]) <= c.jerk*c.decision_dt+1e-8


def mlp(inputs, outputs):
    layers = nn.Sequential(nn.Linear(inputs,64), nn.Tanh(), nn.Linear(64,64), nn.Tanh(), nn.Linear(64,outputs))
    for layer in layers:
        if isinstance(layer,nn.Linear):
            nn.init.orthogonal_(layer.weight,np.sqrt(2)); nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(layers[-1].weight,.01)
    return layers


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = mlp(OBS_DIM,len(LEVELS))
        self.critic = mlp(STATE_DIM,1)

    def distribution(self, obs, mask):
        return Categorical(logits=self.actor(obs).masked_fill(~mask, -1e9))

    def value(self, state):
        return self.critic(state).squeeze(-1)

    @torch.no_grad()
    def proposal(self, local_obs, mask):
        # This path deliberately has no critic or global-state argument.
        logits = self.actor(torch.as_tensor(local_obs,dtype=torch.float32))
        return logits.masked_fill(~torch.as_tensor(mask,dtype=torch.bool),-1e9).argmax(-1).numpy()


@njit(cache=True, fastmath=False)
def project_joint(x, previous, proposed, c, radius):
    """Nearest proposed/hold/backup joint vertex; no rollout optimizer."""
    commands = np.empty((27,3)); scores = np.empty(27)
    for mode in range(27):
        code = mode; score = 0.
        for i in range(2,-1,-1):
            option = code % 3; code //= 3
            backup = max(previous[i]-c[6]*c[1],-c[4])
            v = backup if option == 0 else (previous[i] if option == 1 else proposed[i])
            commands[mode,i] = v
            score += (v-proposed[i])**2
        scores[mode] = score
    order = np.argsort(scores,kind='mergesort')
    for m in order:
        if robust_certificate(x,previous,commands[m],c,radius):
            return commands[m].copy(), int(m), True
    return commands[0].copy(), 0, False


@njit(cache=True, fastmath=False)
def batch_step(states, previous, proposals, errors, c, radius):
    n = len(states)
    following = states.copy(); commands = np.empty((n,3)); valid = np.ones(n,np.bool_)
    for e in range(n):
        proposed = proposals[e].copy()
        for i in range(3):
            if states[e,i,0] >= c[11]:
                proposed[i] = min(c[5],previous[e,i]+c[6]*c[1])
        command, _, ok = project_joint(states[e],previous[e],proposed,c,radius)
        valid[e] = ok; commands[e] = command
        for t in range(round(c[1]/c[0])):
            p0,p1,p2 = following[e,0,0],following[e,1,0],following[e,2,0]
            for i in range(3):
                following[e,i,1] = min(c[3],max(0.,following[e,i,1]+(command[i]+errors[e,t,i])*c[0]))
                following[e,i,0] += following[e,i,1]*c[0]
            valid[e] &= transition_values(p0,p1,p2,following[e,0,0],following[e,1,0],following[e,2,0],c)
    return following,commands,valid


class MARLGate:
    def __init__(self, policy, config=None, radius=.5, mode='learned', sensor_range=SENSOR_RANGE):
        self.policy = policy; self.c = config or MergeConfig(); self.radius = radius
        self.packed = parameters(self.c); self.step = 0; self.mode = mode; self.sensor_range = sensor_range

    def select(self, state, previous, unused=None):
        start = time.perf_counter()
        x,prev,_ = validate(state,previous,self.c,self.radius)
        obs = observations(x,prev,self.step,self.c,self.sensor_range)
        mask = action_mask(prev,self.c)
        if self.mode == 'constant':
            indices = np.where(mask,np.arange(len(LEVELS)),-1).max(-1)
        elif self.mode == 'single':
            indices = np.where(mask,np.arange(len(LEVELS)),-1).max(-1)
            indices[1] = self.policy.proposal(obs[1:2],mask[1:2])[0]
        else:
            indices = self.policy.proposal(obs,mask)
        proposed = LEVELS[indices].copy()
        cleared = x[:,0] >= self.c.goal
        proposed[cleared] = np.minimum(self.c.accel,prev[cleared]+self.c.jerk*self.c.decision_dt)
        backup = np.maximum(prev-self.c.jerk*self.c.decision_dt,-self.c.brake)
        in_set = bool(robust_certificate(x,prev,backup,self.packed,self.radius))
        if not in_set: raise RuntimeError('Measured state outside robust backup contract')
        command,vertex,valid = project_joint(x,prev,proposed,self.packed,self.radius)
        if not valid: raise RuntimeError('No certified joint candidate')
        if self.mode == 'raw': command = proposed
        certified = bool(robust_certificate(x,prev,command,self.packed,self.radius))
        self.step += 1
        return command, {'in_set':in_set,'certified':certified,
            'intervention':float(np.mean(np.abs(command-proposed)>1e-8)),
            'gate_ms':(time.perf_counter()-start)*1000.,'proposal':proposed.tolist(),
            'agent_actions':indices.tolist(),'shield_vertex':vertex,
            'policy_step':self.step,'joint_proposal_accepted':bool(np.allclose(command,proposed,atol=1e-9,rtol=0))}


def load_policy(path):
    torch.set_num_threads(1)
    checkpoint = torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint['algorithm'] not in ('sequential_mappo_v12','single_ego_ppo_v12'): raise ValueError('Unexpected checkpoint')
    model = ActorCritic(); model.load_state_dict(checkpoint['model']); model.eval()
    return model,checkpoint
