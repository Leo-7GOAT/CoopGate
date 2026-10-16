"""Three separately trainable actors, with centralized or local value learning.

This module is experimental. Training is guarded by the V14 single-ego audit.
The physical shield remains centralized; only action proposals are local.
"""
import copy
import time
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from .policy_iteration_v14 import IteratedPolicy, kinematic_features
from .ego_v13 import ego_observations
from .marl_v12 import LEVELS,action_mask,central_state,project_joint,batch_step
from .merge_v4 import MergeConfig
from .fast_v8 import parameters
from .robust_v9 import robust_certificate,validate
from .disturbance_v9 import disturbance_tape


def critic_network(inputs):
    return nn.Sequential(nn.Linear(inputs,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,1))


class JointLearner(nn.Module):
    def __init__(self,ego,critic_kind='central'):
        super().__init__()
        if critic_kind not in ('central','local'):raise ValueError('Unknown critic')
        self.width=ego.encoder[0].out_features;self.critic_kind=critic_kind
        self.actors=nn.ModuleList([IteratedPolicy(self.width),copy.deepcopy(ego),IteratedPolicy(self.width)])
        # Initialize the two new roles to the existing highest-admissible rule.
        # Only the proposal preference is initialized; all actor parameters train.
        for i in (0,2):
            with torch.no_grad():
                self.actors[i].advantage.weight.zero_()
                self.actors[i].advantage.bias.copy_(torch.tensor(3.*LEVELS,dtype=torch.float32))
        self.critics=nn.ModuleList([critic_network(13)] if critic_kind=='central'
                                   else [critic_network(26) for _ in range(3)])

    def logits(self,obs):
        return torch.stack([actor(obs[...,i,:]) for i,actor in enumerate(self.actors)],dim=-2)

    def distribution(self,obs,mask):
        return Categorical(logits=self.logits(obs).masked_fill(~mask,-1e9))

    def value(self,obs,state):
        if self.critic_kind=='central':return self.critics[0](state).expand(*state.shape[:-1],3)
        return torch.stack([critic(kinematic_features(obs[...,i,:])).squeeze(-1)
                            for i,critic in enumerate(self.critics)],dim=-1)

    @torch.no_grad()
    def proposal(self,obs,mask):
        # No centralized critic input is available through this interface.
        obs=torch.as_tensor(obs,dtype=torch.float32);mask=torch.as_tensor(mask,dtype=torch.bool)
        logits=self.logits(obs)
        if not torch.isfinite(logits).all() or not mask.any(-1).all():
            raise ValueError('Nonfinite actor or empty action set')
        return logits.masked_fill(~mask,-1e9).argmax(-1).cpu().numpy()


class JointPolicyGate:
    def __init__(self,policy,config=None):
        self.policy=policy;self.c=config or MergeConfig();self.packed=parameters(self.c);self.step=0

    def select(self,state,previous,unused=None):
        start=time.perf_counter();x,prev,_=validate(state,previous,self.c,.5)
        indices=self.policy.proposal(ego_observations(x,prev,self.step,self.c),action_mask(prev,self.c))
        proposed=LEVELS[indices].copy();cleared=x[:,0]>=self.c.goal
        proposed[cleared]=np.minimum(self.c.accel,prev[cleared]+self.c.jerk*self.c.decision_dt)
        backup=np.maximum(prev-self.c.jerk*self.c.decision_dt,-self.c.brake)
        if not robust_certificate(x,prev,backup,self.packed,.5):raise RuntimeError('Joint policy outside backup set')
        command,vertex,ok=project_joint(x,prev,proposed,self.packed,.5)
        if not ok:raise RuntimeError('No certified joint command')
        self.step+=1
        return command,dict(in_set=True,certified=True,intervention=float(np.mean(np.abs(command-proposed)>1e-8)),
            gate_ms=(time.perf_counter()-start)*1000.,agent_actions=indices.tolist(),proposal=proposed.tolist(),
            shield_vertex=vertex,performance_rollouts=0)


def load_joint(path):
    data=torch.load(path,map_location='cpu',weights_only=True)
    if data['algorithm'] not in ('mappo_three_actor_v15','ippo_three_actor_v15'):raise ValueError('Unexpected checkpoint')
    model=JointLearner(IteratedPolicy(data['actor_width']),data['critic_kind'])
    model.load_state_dict(data['model']);model.eval();return model,data


class CooperativeVector:
    """Exact model training, with per-car costs stopped at that car's arrival.

    All actors share a team payoff. This is a cooperative stochastic game,
    not evidence of a general-sum Nash equilibrium or real-vehicle validation.
    """
    def __init__(self,n=32,seed=1):
        self.c=MergeConfig();self.packed=parameters(self.c);self.n=n;self.rng=np.random.default_rng(seed)
        self.x=np.zeros((n,3,2));self.previous=np.zeros((n,3));self.steps=np.zeros(n,int)
        self.time=np.zeros((n,3));self.variation=np.zeros((n,3));self.errors=np.zeros((n,250,3))
        self.episodes=0;self.reset(np.arange(n))

    def reset(self,ids):
        for i in ids:
            dense=bool(self.rng.integers(2))
            v=self.rng.uniform(12,17,3) if dense else self.rng.uniform(8,14,3)
            p=-self.rng.uniform(70,85,3) if dense else -self.rng.uniform(70,110,3)
            p[2]=p[0]-self.rng.uniform(30,45)
            self.x[i]=np.column_stack((p,v));self.previous[i]=0.;self.steps[i]=0
            self.time[i]=0.;self.variation[i]=0.
            mode=('clean','uniform','bias','pulse')[self.rng.integers(4)]
            self.errors[i]=disturbance_tape(int(self.rng.integers(70000000,80000000)),mode,.5,250)
            if not robust_certificate(self.x[i],self.previous[i],np.full(3,-2.5),self.packed,.5):
                raise RuntimeError('Invalid initial; never silently reject')

    def inputs(self):
        return (ego_observations(self.x,self.previous,self.steps,self.c),
                central_state(self.x,self.previous,self.steps,self.c),action_mask(self.previous,self.c))

    def step(self,actions):
        actions=np.asarray(actions)
        if actions.shape!=(self.n,3) or not np.issubdtype(actions.dtype,np.integer):raise ValueError('Three integer proposals required')
        if np.any(actions<0) or np.any(actions>=len(LEVELS)):raise ValueError('Invalid action index')
        if not np.take_along_axis(action_mask(self.previous),actions[...,None],axis=-1).all():raise ValueError('Slew-inadmissible proposal')
        error=self.errors[np.arange(self.n)[:,None],self.steps[:,None]*5+np.arange(5)]
        following,command,ok=batch_step(self.x,self.previous,LEVELS[actions],error,self.packed,.5)
        if not ok.all():raise RuntimeError('Joint training safety contract failed')
        active=self.x[:,:,0]<self.c.goal
        dt=active*self.c.decision_dt;variation=active*np.abs(command-self.previous)
        individual_cost=dt+.05*variation
        potential=-np.maximum(self.c.goal-self.x[:,:,0],0).mean(-1)/self.c.vmax
        future=-np.maximum(self.c.goal-following[:,:,0],0).mean(-1)/self.c.vmax
        self.time+=dt;self.variation+=variation;self.steps+=1;self.x=following;self.previous=command
        terminal=np.all(self.x[:,:,0]>=self.c.goal,axis=-1);boundary=terminal|(self.steps>=50)
        future[boundary]=0.
        # gamma=1 and terminal potential=0 preserve the finite-horizon objective.
        reward=-individual_cost.mean(-1)+future-potential
        following_inputs=self.inputs()
        finished=[dict(time=self.time[i].tolist(),variation=self.variation[i].tolist(),
            cost=(self.time[i]+.05*self.variation[i]).tolist(),complete=bool(terminal[i])) for i in np.flatnonzero(boundary)]
        self.episodes+=int(boundary.sum());self.reset(np.flatnonzero(boundary))
        return following_inputs,reward.astype(np.float32),boundary,active,finished
