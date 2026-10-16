import numpy as np
import torch
from coopgate.ego_v13 import EgoVector,ego_observations,EgoPolicyGate
from coopgate.marl_v12 import observations,ActorCritic,action_mask,LEVELS,batch_step
from coopgate.robust_v9 import robust_certificate


def test_neighbor_command_memory_removes_observation_alias():
    state=np.array([[-80.,12.],[-85.,13.],[-120.,11.]])
    a=np.zeros(3);b=np.array([2.5,0.,-2.5])
    # The old actor cannot distinguish different joint safety-filter memories.
    np.testing.assert_array_equal(observations(state,a,0)[1],observations(state,b,0)[1])
    new_a=ego_observations(state,a,0)[1];new_b=ego_observations(state,b,0)[1]
    assert new_a.shape==(18,) and np.max(np.abs(new_a-new_b))==.5
    np.testing.assert_array_equal(new_a[:6],new_b[:6])


def test_ego_reward_matches_executed_ego_command_with_terminal_potential():
    env=EgoVector(4,1382,.05);x=env.x.copy();previous=env.previous.copy()
    mask=action_mask(previous);actions=np.where(mask[:,1],np.arange(len(LEVELS)),-1).max(-1)
    next_inputs,reward,terminal,boundary,_=env.step(actions[:,None])
    # Central-state return is before any reset, preserving the terminal command.
    following=next_inputs[1][:,1]*150.;commands=next_inputs[1][:,7]*5.
    phi=-np.maximum(60-x[:,1,0],0)/20.
    next_phi=-np.maximum(60-following,0)/20.;next_phi[boundary]=0
    expected=-.5-.05*np.abs(commands-previous[:,1])+.995*next_phi-phi
    np.testing.assert_allclose(reward,expected,atol=2e-6,rtol=0)


def test_ego_inference_and_training_share_shield_and_fixed_neighbors():
    torch.manual_seed(22);torch.set_num_threads(1);model=ActorCritic();env=EgoVector(1,731,.05)
    x=env.x[0].copy();previous=env.previous[0].copy()
    gate=EgoPolicyGate(model);command,info=gate.select(x,previous)
    proposed=np.array(info['proposal']);assert proposed[0]==2.5 and proposed[2]==2.5
    nxt,checked,valid=batch_step(x[None],previous[None],proposed[None],np.zeros((1,5,3)),env.packed,.5)
    assert valid.all() and robust_certificate(x,previous,command,env.packed,.5)
    np.testing.assert_array_equal(command,checked[0])
