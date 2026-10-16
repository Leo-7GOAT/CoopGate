"""Vectorized exact longitudinal dynamics for on-policy multi-agent learning.

This is a model simulator, not a SUMO invocation. Independent SUMO validation
and trajectory parity are required before reporting deployment performance.
"""
import numpy as np
from .merge_v4 import MergeConfig
from .fast_v8 import parameters
from .marl_v12 import observations,central_state,action_mask,batch_step,LEVELS
from .robust_v9 import robust_certificate


class VectorMerge:
    def __init__(self,n=32,seed=1,variation_weight=.015):
        self.c = MergeConfig(); self.packed = parameters(self.c); self.radius = .5
        self.n = n; self.rng = np.random.default_rng(seed); self.variation_weight = variation_weight
        self.x = np.zeros((n,3,2)); self.previous = np.zeros((n,3)); self.steps = np.zeros(n,int)
        self.returns = np.zeros(n); self.time_cost = np.zeros(n); self.variation = np.zeros(n)
        self.episodes = 0; self.reset(np.arange(n))

    def reset(self,ids):
        for i in ids:
            dense = bool(self.rng.integers(2))
            v = self.rng.uniform(12,17,3) if dense else self.rng.uniform(8,14,3)
            p = -self.rng.uniform(70,85,3) if dense else -self.rng.uniform(70,110,3)
            p[2] = p[0]-self.rng.uniform(30,45)
            # SUMO insertion preserves departPos; checked against TraCI reset.
            self.x[i] = np.column_stack((p,v)); self.previous[i] = 0.
            self.steps[i] = 0; self.returns[i] = 0.; self.time_cost[i] = 0.; self.variation[i] = 0.
            if not robust_certificate(self.x[i],self.previous[i],np.full(3,-2.5),self.packed,self.radius):
                raise RuntimeError('Training reset outside contract; do not silently reject')

    def inputs(self):
        return (observations(self.x,self.previous,self.steps,self.c),
                central_state(self.x,self.previous,self.steps,self.c),action_mask(self.previous,self.c))

    def step(self,actions):
        errors = self.rng.uniform(-self.radius,self.radius,(self.n,5,3))
        following,commands,valid = batch_step(self.x,self.previous,LEVELS[actions],errors,self.packed,self.radius)
        if not valid.all(): raise RuntimeError('Training safety contract failed')
        active = self.x[:,:,0] < self.c.goal
        stage_time = active.sum(-1)*self.c.decision_dt/3.
        delta = np.abs(commands-self.previous).sum(-1)
        potential = -np.maximum(self.c.goal-self.x[:,:,0],0).sum(-1)/(3*self.c.vmax)
        next_potential = -np.maximum(self.c.goal-following[:,:,0],0).sum(-1)/(3*self.c.vmax)
        self.steps += 1; self.x = following; self.previous = commands
        terminal = np.all(self.x[:,:,0] >= self.c.goal,axis=1)
        truncated = (self.steps >= round(self.c.horizon/self.c.decision_dt)) & ~terminal
        boundary = terminal | truncated
        # The 25-second task horizon is part of the state and objective, so its
        # boundary is terminal (not an external infinite-horizon time limit).
        next_potential[boundary] = 0.
        rewards = -stage_time-self.variation_weight*delta+.995*next_potential-potential
        self.returns += rewards; self.time_cost += stage_time; self.variation += delta
        next_inputs = self.inputs()
        finished = [{'return':float(self.returns[i]),'cost':float(self.time_cost[i]),
                     'variation':float(self.variation[i]),'complete':bool(terminal[i])} for i in np.flatnonzero(boundary)]
        self.episodes += int(boundary.sum()); self.reset(np.flatnonzero(boundary))
        return next_inputs,rewards.astype(np.float32),terminal,boundary,finished


class SingleVectorMerge(VectorMerge):
    """Exactly one trainable vehicle (ramp ego1), two fixed proposal policies.

    All methods use the same joint safety filter; it can brake rule vehicles.
    This is single-agent RL in interactive traffic, not a single-car empty road.
    """
    def inputs(self):
        obs,state,mask=super().inputs()
        return obs[:,1:2],state,mask[:,1:2]

    def step(self,actions):
        mask=action_mask(self.previous,self.c)
        joint=np.where(mask,np.arange(len(LEVELS)),-1).max(-1)
        joint[:,1]=np.asarray(actions).reshape(self.n)
        return super().step(joint)
