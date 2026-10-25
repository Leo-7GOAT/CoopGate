"""Observable-only nominal dynamics for the pinned original DMPS planner.

No physical environment handle, actuator truth, realized disturbance, or future
tape is accepted. Forecast recovery is deterministic and nominal; mismatch may
invalidate it. Completed vehicles follow the common post-goal cruise lifecycle.
"""
from dataclasses import dataclass
from types import SimpleNamespace
import uuid

import numpy as np
import torch

from .merge_v4 import MergeConfig, admissible, backup, transition_safe


def command_for(action, state, previous, config):
    action=np.asarray(action,dtype=float)
    if action.shape!=(3,) or not np.isfinite(action).all() or np.any(np.abs(action)>1+1e-12):
        raise ValueError('Normalized finite joint action in [-1,1] required')
    command=admissible(-1.25+3.75*np.clip(action,-1,1),previous,config)
    cruise=admissible(np.full(3,config.accel),previous,config)
    return np.where(state[:,0]>=config.goal,cruise,command)


@dataclass
class ModelState:
    owner: str
    generation: int
    state: np.ndarray
    previous: np.ndarray
    step: int
    failed: bool


class NominalModelEnv:
    """Pure deterministic model state, compatible with the author four-tuple API."""
    def __init__(self,initial_state,previous=None,step=0):
        self.c=MergeConfig();self.num_steps=round(self.c.horizon/self.c.decision_dt)
        self.observation_space=SimpleNamespace(shape=(10,));self.action_space=SimpleNamespace(shape=(3,))
        self.owner=uuid.uuid4().hex;self.generation=0;self.model_decisions=0
        self.initial=np.asarray(initial_state,dtype=float).copy()
        self.initial_previous=np.zeros(3) if previous is None else np.asarray(previous,dtype=float).copy()
        self.initial_step=step
        self.reset()

    def sync(self,state,previous,step):
        x=np.asarray(state,dtype=float);u=np.asarray(previous,dtype=float)
        if (x.shape!=(3,2) or u.shape!=(3,) or not np.isfinite(x).all() or not np.isfinite(u).all()
                or np.any(x[:,1]<0) or np.any(x[:,1]>self.c.vmax)
                or np.any(u < -self.c.brake) or np.any(u > self.c.accel)
                or isinstance(step,(bool,np.bool_)) or not isinstance(step,(int,np.integer))
                or not 0<=step<self.num_steps):
            raise ValueError('Finite observable state, feasible previous command and actual decision index required')
        self.x=x.copy();self.previous=u.copy();self.index=int(step);self.generation+=1
        self.failed=not bool(transition_safe(x[:,0][None],x[:,0][None],self.c)[0])

    def reset(self):
        self.sync(self.initial,self.initial_previous,self.initial_step)
        return self.obs()

    def obs(self):
        return np.append(np.column_stack((self.x[:,0]/150,self.x[:,1]/20,self.previous/5)).ravel(),
                         self.index/self.num_steps).astype(np.float32)

    def safe(self):
        return bool(not self.failed and self.index<self.num_steps+20)

    def stable(self):
        active=self.x[:,0]<self.c.goal
        # Active cars stay stopped under braking. Completed cars at vmax keep
        # equal speed under cruise; they are downstream of every active car.
        return bool(self.safe() and np.all(self.x[active,1]<=1e-10)
                    and np.all(self.previous[active]<=0)
                    and np.all(self.x[~active,1]>=self.c.vmax-1e-10))

    def backup(self):
        return np.clip((backup(self.previous,self.c)+1.25)/3.75,-1,1)

    def step(self,action):
        if not self.safe():return self.obs(),0.,True,dict(nominal_safe=False)
        command=command_for(action,self.x,self.previous,self.c)
        active=self.x[:,0]<self.c.goal
        reward=-float(np.mean(active*(self.c.decision_dt+.05*np.abs(command-self.previous))))
        for _ in range(round(self.c.decision_dt/self.c.dt)):
            before=self.x[:,0].copy()
            self.x[:,1]=np.clip(self.x[:,1]+self.c.dt*command,0,self.c.vmax)
            self.x[:,0]+=self.c.dt*self.x[:,1]
            self.failed|=not bool(transition_safe(before[None],self.x[:,0][None],self.c)[0])
        self.previous=command;self.index+=1;self.model_decisions+=1
        done=self.failed or self.index>=self.num_steps or np.all(self.x[:,0]>=self.c.goal)
        return self.obs(),reward,bool(done),dict(nominal_safe=self.safe())

    def get_state(self):
        return ModelState(self.owner,self.generation,self.x.copy(),self.previous.copy(),self.index,self.failed)

    def set_state(self,state):
        if not isinstance(state,ModelState) or (state.owner,state.generation)!=(self.owner,self.generation):
            raise ValueError('Model snapshot belongs to another observable-state synchronization')
        self.x=state.state.copy();self.previous=state.previous.copy();self.index=state.step;self.failed=state.failed


class ObservableDMPS:
    def __init__(self,model,planner_class):
        self.model=model
        self.env=NominalModelEnv(np.array([[-140.,12.],[-220.,10.],[-190.,12.]]))
        self.planner=planner_class(self.env,.99,4,4,2,torch.device('cpu'),model.critic1,single_action=True)

    def choose(self,state,previous,step):
        self.env.sync(state,previous,step)
        if not self.env.safe():raise RuntimeError('Observed state outside nominal geometric domain')
        self.planner.obs=self.env.obs()
        with torch.no_grad():
            proposal=self.model.actor(torch.as_tensor(self.planner.obs,dtype=torch.float32)).cpu().numpy()
        before=self.env.model_decisions
        data,(actions,intervention)=self.planner.take_step(proposal)
        if len(data)!=1 or len(actions)!=1 or self.env.index!=step+1:
            raise RuntimeError('Original planner did not return exactly one nominal decision')
        if not self.env.safe():raise RuntimeError('Original planner returned a nominally unsafe command')
        command=command_for(actions[0],np.asarray(state),np.asarray(previous),self.env.c)
        np.testing.assert_array_equal(command,self.env.previous)
        return command,dict(proposal=np.asarray(proposal).tolist(),selected_action=np.asarray(actions[0]).tolist(),
            intervention=int(intervention),predicted_state=self.env.x.tolist(),
            nominal_model_decisions=self.env.model_decisions-before,
            prediction_contract='Deterministic nominal backup recovery; no unknown-plant or disturbance guarantee')
