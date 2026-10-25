from pathlib import Path

import numpy as np
import pytest
import torch

from coopgate.dmps_effective_action import EffectiveCriticInputs,effective_action
from coopgate.dmps_nominal_model import command_for
from coopgate.merge_v4 import MergeConfig
import load_original_dmps as author


def observation(state,previous):
    return torch.tensor(np.append(np.column_stack((state[:,0]/150,state[:,1]/20,previous/5)).ravel(),0.),dtype=torch.float32)


def test_decoding_matches_physical_command_function_and_ignored_actions():
    state=np.array([[100.,20.],[-30.,0.],[-65.,0.]])
    previous=np.array([-5.,-5.,-5.]);obs=observation(state,previous)
    for a0 in (-1.,0.,1.):
        a=np.array([a0,1.,-.8]);effective=effective_action(obs,torch.tensor(a,dtype=torch.float32))
        expected=(command_for(a,state,previous,MergeConfig())+1.25)/3.75
        np.testing.assert_allclose(effective.numpy(),expected,atol=1e-7,rtol=0)


def test_ignored_and_slew_saturated_coordinates_have_zero_action_gradient():
    state=np.array([[100.,20.],[-30.,0.],[-65.,0.]])
    obs=observation(state,np.zeros(3));a=torch.tensor([-.9,-.8,0.],requires_grad=True)
    effective_action(obs,a).sum().backward()
    torch.testing.assert_close(a.grad,torch.tensor([0.,0.,1.]))


def test_just_before_goal_rounding_does_not_force_cruise():
    state=np.array([[60.-1e-7,12.],[-30.,0.],[-65.,0.]])
    obs=observation(state,np.zeros(3))
    assert obs[0]==torch.tensor(.4)
    command=-1.25+3.75*effective_action(obs,torch.full((3,),-1.))
    assert command[0]==-2.5


def test_hooks_preserve_parameters_optimizer_and_serialization_but_remove_spurious_values():
    source=Path(__file__).resolve().parents[2]/'reference_sources/dmps-f37df2d'
    if not source.exists():pytest.skip('Pinned external author source unavailable')
    module,_=author.load(source);torch.manual_seed(805200)
    model=module.Model(10,3,torch.device('cpu'),ret_low=-32.,ret_high=0.)
    original={k:v.clone() for k,v in model.state_dict().items()}
    identities=[id(p) for p in model.parameters()]
    hooks=EffectiveCriticInputs(model)
    assert identities==[id(p) for p in model.parameters()]
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in original.items())
    obs=observation(np.array([[100.,20.],[-30.,0.],[-65.,0.]]),np.zeros(3)).repeat(3,1)
    actions=torch.tensor([[-1.,-.9,-.8],[0.,-.9,-.8],[1.,-.9,-.8]])
    for name in hooks.names:
        values=getattr(model,name)(obs,actions)
        torch.testing.assert_close(values,values[0].expand_as(values),atol=0,rtol=0)
    loss=model.critic1(obs,actions).square().mean();optimizer=torch.optim.Adam(model.critic1.parameters())
    optimizer.zero_grad();loss.backward();optimizer.step()
    assert any(not torch.equal(v,model.state_dict()[k]) for k,v in original.items() if k.startswith('critic1.'))
    with pytest.raises(ValueError,match='already installed'):EffectiveCriticInputs(model)


def test_unbatched_and_batched_maps_match_and_reject_shape_aliasing():
    obs=torch.zeros(4,10);action=torch.tensor([[.2,-.9,1.]]).repeat(4,1)
    expected=torch.stack([effective_action(o,a) for o,a in zip(obs,action)])
    torch.testing.assert_close(effective_action(obs,action),expected,atol=0,rtol=0)
    with pytest.raises(ValueError):effective_action(obs,action[0])


def test_encoding_matches_all_500_sealed_native_curriculum_test_commands():
    import json
    root=Path(__file__).resolve().parents[1]/'artifacts/literature_reproduction/dmps_curriculum_training/evaluation_new'
    if not root.exists():pytest.skip('Sealed native fixture unavailable')
    errors=[]
    for i in range(10):
        record=json.loads((root/f'episode_{i:03d}'/'episode.json').read_text())
        for decision in record['decisions']:
            state=np.asarray(decision['observation_state']);previous=np.asarray(decision['previous_command'])
            obs=observation(state,previous);a=torch.tensor(decision['planner']['selected_action'],dtype=torch.float32)
            command=(-1.25+3.75*effective_action(obs,a)).numpy()
            actual=np.asarray(decision['trace']['command'])
            errors.append(float(np.max(np.abs(command-actual))))
    assert len(errors)==500 and max(errors)<2e-6
