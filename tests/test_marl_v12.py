import numpy as np
import torch
from coopgate.marl_v12 import (ActorCritic,MARLGate,LEVELS,OBS_DIM,STATE_DIM,
    observations,central_state,action_mask,project_joint,batch_step)
from coopgate.fast_v8 import parameters
from coopgate.merge_v4 import MergeConfig
from coopgate.robust_v9 import robust_certificate


def initial():
    return np.array([[-80.,12.],[-85.,13.],[-120.,11.]])


def test_actor_information_boundary():
    x=initial();x[2,0]=-300;prev=np.zeros(3)
    before=observations(x,prev,0)[0].copy()
    x[2]=[-500.,19.];prev[2]=-5
    assert np.array_equal(before,observations(x,prev,0)[0])
    assert observations(x,prev,0).shape==(3,OBS_DIM)
    assert central_state(x,prev,0).shape==(STATE_DIM,)


def test_deployment_does_not_call_critic():
    model=ActorCritic()
    def forbidden(*args):raise AssertionError('Critic leaked into deployment')
    model.critic.forward=forbidden
    gate=MARLGate(model);command,info=gate.select(initial(),np.zeros(3))
    assert info['certified'] and command.shape==(3,)


def test_joint_projection_matches_exhaustive_reference():
    from itertools import product
    c=MergeConfig();packed=parameters(c);x=initial();prev=np.zeros(3);proposal=np.array([2.5,1.25,0.])
    candidates=np.array(list(product(*[(-2.5,prev[i],proposal[i]) for i in range(3)])))
    scores=((candidates-proposal)**2).sum(-1)
    valid=np.array([robust_certificate(x,prev,u,packed,.5) for u in candidates])
    expected=candidates[np.argmin(np.where(valid,scores,np.inf))]
    chosen,_,ok=project_joint(x,prev,proposal,packed,.5)
    assert ok and np.array_equal(expected,chosen)


def test_mask_and_disturbance_preserve_contract():
    c=MergeConfig();packed=parameters(c);x=initial()[None];prev=np.zeros((1,3));rng=np.random.default_rng(12)
    for step in range(40):
        mask=action_mask(prev)
        action=np.array([[rng.choice(np.flatnonzero(mask[0,i])) for i in range(3)]])
        nxt,cmd,valid=batch_step(x,prev,LEVELS[action],rng.uniform(-.5,.5,(1,5,3)),packed,.5)
        assert valid.all()
        assert np.max(np.abs(cmd-prev))<=2.5+1e-12
        backup=np.maximum(cmd[0]-2.5,-5)
        assert robust_certificate(nxt[0],cmd[0],backup,packed,.5)
        x,prev=nxt,cmd


def test_policy_gradient_is_for_sampled_proposals():
    torch.manual_seed(12);model=ActorCritic()
    obs=torch.tensor(observations(initial(),np.zeros(3),0));mask=torch.tensor(action_mask(np.zeros(3)))
    dist=model.distribution(obs,mask);actions=dist.sample()
    loss=-dist.log_prob(actions).sum()
    loss.backward()
    assert sum(float(p.grad.abs().sum()) for p in model.actor.parameters())>0
    assert all(p.grad is None for p in model.critic.parameters())
    assert all(mask[i,actions[i]] for i in range(3))
