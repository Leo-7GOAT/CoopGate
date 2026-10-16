"""Independent checks for training-only planning and deployed neural actions."""
import numpy as np
import torch
from coopgate.ego_v13 import EgoVector, ego_observations, ego_rollout
from coopgate.marl_v12 import LEVELS, action_mask, batch_step
from coopgate.learned_rollout_v13 import actor_action
from coopgate.policy_iteration_v14 import IteratedPolicy, action_returns


def test_compiled_apprentice_matches_local_torch_on_visited_states():
    torch.set_num_threads(1)
    torch.manual_seed(417)
    model=IteratedPolicy()
    env=EgoVector(8,8133)
    weights=model.compiled_weights()
    for _ in range(8):
        obs,_,mask=env.inputs()
        indices=model.proposal(obs[:,0],mask[:,0])
        for i in range(env.n):
            proposal=actor_action(env.x[i],env.previous[i],env.steps[i],weights,True,2)
            assert proposal==LEVELS[indices[i]]
        env.step(indices[:,None])


def test_rule_only_action_returns_equal_strong_rollout_value():
    torch.manual_seed(11)
    model=IteratedPolicy()
    env=EgoVector(3,1833)
    for _ in range(6):
        for i in range(env.n):
            cost,count=action_returns(env.x[i],env.previous[i],env.steps[i],env.packed,
                                      model.compiled_weights(),2,False)
            _,_,minimum,_=ego_rollout(env.x[i],env.previous[i],env.steps[i],env.packed,.5,.05)
            np.testing.assert_allclose(np.min(cost),minimum,atol=1e-10,rtol=0)
            assert count>0
        _,_,mask=env.inputs()
        env.step(np.where(mask[:,0],np.arange(len(LEVELS)),-1).max(-1)[:,None])


def test_policy_returns_match_independent_torch_feedback_rollout():
    torch.set_num_threads(1)
    torch.manual_seed(134)
    model=IteratedPolicy()
    env=EgoVector(1,7398)
    x=env.x.copy();prev=env.previous.copy()
    costs,_=action_returns(x[0],prev[0],0,env.packed,model.compiled_weights(),2,True)
    for first in np.flatnonzero(action_mask(prev)[0,1]):
        alternatives=[]
        for learned in (False,True):
            state=x.copy();memory=prev.copy();value=0.
            for step in range(50):
                if state[0,1,0]>=env.c.goal:break
                mask=action_mask(memory)
                indices=np.where(mask,np.arange(len(LEVELS)),-1).max(-1)
                if step==0:indices[0,1]=first
                elif learned:indices[0,1]=model.proposal(ego_observations(state,memory,step)[:,1],mask[:,1])[0]
                following,commands,valid=batch_step(state,memory,LEVELS[indices],np.zeros((1,5,3)),env.packed,.5)
                assert valid.all()
                value+=.5+.05*abs(commands[0,1]-memory[0,1])
                state,memory=following,commands
            alternatives.append(value)
        np.testing.assert_allclose(costs[first],min(alternatives),atol=1e-10,rtol=0)
