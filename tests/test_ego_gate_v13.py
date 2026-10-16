import numpy as np
import torch
from coopgate.ego_models_v13 import QPolicy
from coopgate.ego_v13 import EgoVector,ego_observations
from coopgate.marl_v12 import action_mask,LEVELS
from coopgate.learned_rollout_v13 import actor_action,LearnedRolloutGate
from coopgate.ego_confidence_v13 import EgoConfidenceGate
from coopgate.confidence_v12 import conformal_upper


def test_compiled_q_actions_match_torch_on_visited_states():
    torch.set_num_threads(1);torch.manual_seed(173);model=QPolicy();params=model.state_dict()
    keys=['encoder.0.weight','encoder.0.bias','encoder.2.weight','encoder.2.bias','advantage.weight','advantage.bias']
    weights=tuple(params[k].numpy() for k in keys);env=EgoVector(16,382,.05)
    for _ in range(20):
        obs,_,mask=env.inputs();chosen=model.proposal(obs[:,0],mask[:,0])
        fast=np.array([actor_action(x,p,int(t),weights,True,1) for x,p,t in zip(env.x,env.previous,env.steps)])
        np.testing.assert_array_equal(LEVELS[chosen],fast)
        env.step(chosen[:,None])


def test_max_residual_quantile_bounds_both_components():
    residual=np.array([[-.3,.2],[.4,.1],[-.2,-.1],[0.,.25]])
    q=conformal_upper(residual.max(-1),.4)
    assert q==.25
    assert np.array_equal((residual<=q).all(-1),residual.max(-1)<=q)
    assert np.isinf(conformal_upper([.1,.2],.1))


def test_out_of_domain_admission_returns_baseline_without_loading_policy():
    initial=np.array([[-80.,7.9],[-85.,13.],[-120.,11.]])
    gate=EgoConfidenceGate('does-not-exist.pt',initial,{'groups':{}})
    assert not gate.accept and gate.reason=='outside_calibration_domain'
    command,info=gate.select(initial,np.zeros(3))
    assert np.isfinite(command).all() and info['certified']


def test_policy_fault_uses_current_state_baseline_and_keeps_takeover():
    class Broken:
        def select(self,*args):raise RuntimeError('injected network fault')
    initial=np.array([[-80.,7.9],[-85.,13.],[-120.,11.]])
    gate=EgoConfidenceGate('unused',initial,{'groups':{}});gate.accept=True;gate.learned=Broken()
    expected,_=gate.baseline.select(initial,np.zeros(3));gate.baseline.step=0
    command,info=gate.select(initial,np.zeros(3))
    np.testing.assert_array_equal(command,expected)
    assert gate.fault_fallback and info['fault_fallback'] and gate.step==1
