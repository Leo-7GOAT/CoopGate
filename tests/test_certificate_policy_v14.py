import numpy as np
import torch
from coopgate.certificate_policy_v14 import CertificatePolicy,candidates,choose_compiled,returns,CertificatePolicyGate
from coopgate.ego_v13 import EgoVector,ego_rollout
from coopgate.marl_v12 import batch_step,LEVELS
from coopgate.robust_v9 import robust_certificate


def test_certificate_actor_uses_same_actions_and_safety_filter():
    torch.set_num_threads(1);torch.manual_seed(515)
    actor=CertificatePolicy();env=EgoVector(3,3433)
    for step in range(12):
        actions=[]
        for i in range(env.n):
            x,prev=env.x[i],env.previous[i]
            features,commands,mask=candidates(x,prev,int(env.steps[i]),env.packed)
            selected=int(actor.choose(features[None],mask[None])[0])
            compiled,index=choose_compiled(x,prev,int(env.steps[i]),env.packed,actor.compiled_weights())
            np.testing.assert_array_equal(compiled,commands[selected])
            # Different proposals can project to identical commands/features.
            # BLAS rounding may break their score tie differently by platform.
            if index!=selected:
                np.testing.assert_array_equal(features[index],features[selected])
            assert robust_certificate(x,prev,compiled,env.packed,.5)
            actions.append(selected)
        env.step(np.array(actions)[:,None])


def test_certificate_feature_rule_targets_preserve_strong_baseline():
    torch.manual_seed(15);actor=CertificatePolicy();env=EgoVector(3,574)
    for i in range(env.n):
        _,q,count=returns(env.x[i],env.previous[i],0,env.packed,actor.compiled_weights(),False)
        _,_,expected,_=ego_rollout(env.x[i],env.previous[i],0,env.packed,.5,.05)
        np.testing.assert_allclose(q.min(),expected,atol=1e-10,rtol=0)
        assert count>0


def test_arrived_ego_uses_shared_clearance_rule():
    actor=CertificatePolicy();env=EgoVector(1,572)
    x=np.array([[110.,15.],[80.,13.],[65.,12.]])
    previous=np.array([2.5,0.,2.5])
    f,commands,mask=candidates(x,previous,25,env.packed)
    assert mask.sum()==1 and LEVELS[np.flatnonzero(mask)[0]]==2.5
    gate=CertificatePolicyGate(actor);gate.step=25
    command,info=gate.select(x,previous)
    _,checked,valid=batch_step(x[None],previous[None],command[None],np.zeros((1,5,3)),env.packed,.5)
    assert valid.all() and info['performance_rollouts']==0
    np.testing.assert_array_equal(checked[0],command)
