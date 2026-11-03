"""Offline, coupled multi-agent approximate policy iteration.

One actor is improved while the other actors and the complete future joint
policy remain frozen. Branch returns include the three vehicles' common cost
and their closed-loop reactions. Deployment performs no performance rollout.
This is model-based MARL, not MAPPO and not a new policy-iteration theorem.
"""
import numpy as np
import torch
from .cooperative_v15 import JointLearner
from .policy_iteration_v14 import IteratedPolicy
from .ego_v13 import ego_observations
from .marl_v12 import LEVELS,action_mask,batch_step
from .merge_v4 import MergeConfig
from .fast_v8 import parameters


def nominal_returns(states,previous,steps,first_actions,model):
    """Complete finite-horizon common costs for a batch of first joint actions."""
    c=MergeConfig();packed=parameters(c)
    x=np.array(states,dtype=float,copy=True);prev=np.array(previous,dtype=float,copy=True)
    clock=np.array(steps,dtype=int,copy=True);actions=np.asarray(first_actions)
    if x.shape!=(len(x),3,2) or prev.shape!=(len(x),3) or actions.shape!=(len(x),3):
        raise ValueError('Joint state, command memory, and first actions required')
    if clock.shape!=(len(x),) or np.any(clock<0) or np.any(clock>=50):raise ValueError('Invalid finite-horizon clock')
    if not np.take_along_axis(action_mask(prev),actions[...,None],axis=-1).all():raise ValueError('Inadmissible first proposal')
    totals=np.zeros((len(x),3));done=np.all(x[:,:,0]>=c.goal,axis=1);transitions=0
    first=True
    while not done.all():
        ids=np.flatnonzero(~done);state=x[ids];last=prev[ids]
        if first:proposal=actions[ids]
        else:proposal=model.proposal(ego_observations(state,last,clock[ids]),action_mask(last))
        following,command,valid=batch_step(state,last,LEVELS[proposal],np.zeros((len(ids),5,3)),packed,.5)
        if not valid.all():raise RuntimeError('Counterfactual branch outside physical contract')
        active=state[:,:,0]<c.goal
        totals[ids]+=active*(.5+.05*np.abs(command-last))
        x[ids]=following;prev[ids]=command;clock[ids]+=1;transitions+=len(ids)
        done[ids]=np.all(following[:,:,0]>=c.goal,axis=1)|(clock[ids]>=50);first=False
    return totals,int(transitions)


def unilateral_returns(states,previous,steps,model,role):
    """Q_pi(s, a_i, pi_-i(s)); the future joint policy is the current policy.

    This is not an isolated single-car task: other vehicles react throughout
    every branch, the shared physical projection can change their commands,
    and all three costs enter each action's value.
    """
    if role not in (0,1,2):raise ValueError('Unknown vehicle')
    states=np.asarray(states);previous=np.asarray(previous);steps=np.asarray(steps)
    obs=ego_observations(states,previous,steps);mask=action_mask(previous)
    incumbent=model.proposal(obs,mask)
    rows,actions=np.nonzero(mask[:,role])
    joint=incumbent[rows].copy();joint[:,role]=actions
    costs,count=nominal_returns(states[rows],previous[rows],steps[rows],joint,model)
    table=np.full((len(states),len(LEVELS)),np.inf);table[rows,actions]=costs.mean(-1)
    return table,incumbent[:,role],count


def load_coupled(path):
    meta=torch.load(path,map_location='cpu',weights_only=True)
    if meta['algorithm']!='coupled_policy_iteration_v16':raise ValueError('Not a coupled policy-iteration model')
    model=JointLearner(IteratedPolicy(meta['actor_width']));model.load_state_dict(meta['model']);model.eval()
    return model,meta
