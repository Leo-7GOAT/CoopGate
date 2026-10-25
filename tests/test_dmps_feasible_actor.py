import json
from pathlib import Path

import numpy as np
import pytest
import torch

from coopgate.dmps_effective_action import EffectiveCriticInputs,effective_action
from coopgate.dmps_feasible_actor import FeasibleActorOutputs,feasible_request
from coopgate.dmps_nominal_model import command_for
from coopgate.merge_v4 import MergeConfig
import load_original_dmps as author


def test_active_composite_gradient_is_not_flat_in_the_request_interior():
    generator=torch.Generator().manual_seed(905800)
    obs=torch.zeros(1024,10);obs[:,(0,3,6)]=-.5
    obs[:,(2,5,8)]=(-5+7.5*torch.rand(1024,3,generator=generator))/5
    z=(-.95+1.9*torch.rand(1024,3,generator=generator)).requires_grad_()
    effective_action(obs,feasible_request(obs,z)).sum().backward()
    previous=5*obs[:,(2,5,8)]
    expected=(torch.minimum(previous+2.5,torch.full_like(previous,2.5))
        -torch.maximum(previous-2.5,torch.full_like(previous,-5.)))/7.5
    torch.testing.assert_close(z.grad,expected,atol=2e-7,rtol=0)
    assert z.grad.min()>=1/3-2e-7 and z.grad.max()<=2/3+2e-7


def test_both_endpoints_cover_the_entire_existing_feasible_interval():
    obs=torch.zeros(301,10);previous=torch.linspace(-5.,2.5,301)[:,None].repeat(1,3)
    obs[:,(2,5,8)]=previous/5
    for endpoint,expected in [(-1.,torch.maximum(previous-2.5,torch.full_like(previous,-5.))),
                              (1.,torch.minimum(previous+2.5,torch.full_like(previous,2.5)))]:
        request=feasible_request(obs,torch.full((301,3),endpoint))
        assert torch.all(request.abs()<=1.)
        decoded=-1.25+3.75*request
        torch.testing.assert_close(decoded,expected,atol=1e-6,rtol=0)


def test_completed_coordinates_use_common_cruise_with_no_latent_gradient():
    obs=torch.zeros(10);obs[(0,3,6),]=1.;obs[(2,5,8),]=torch.tensor([-1.,0.,.5])
    z=torch.tensor([-.8,.1,.9],requires_grad=True)
    decoded=-1.25+3.75*feasible_request(obs,z)
    torch.testing.assert_close(decoded,torch.tensor([-2.5,2.5,2.5]),atol=0,rtol=0)
    decoded.sum().backward();torch.testing.assert_close(z.grad,torch.zeros(3),atol=0,rtol=0)


def test_both_actor_hooks_preserve_state_keys_parameters_and_gradient_updates():
    source=Path(__file__).resolve().parents[2]/'reference_sources/dmps-f37df2d'
    if not source.exists():pytest.skip('Pinned original author source unavailable')
    module,_=author.load(source);torch.manual_seed(805200)
    model=module.Model(10,3,torch.device('cpu'),ret_low=-32.,ret_high=0.)
    original={k:v.clone() for k,v in model.state_dict().items()};ids=[id(p) for p in model.parameters()]
    EffectiveCriticInputs(model);hooks=FeasibleActorOutputs(model)
    assert ids==[id(p) for p in model.parameters()]
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in original.items())
    obs=torch.zeros(4,10);a=model.actor(obs);target=model.actor_targ(obs)
    torch.testing.assert_close(a,target,atol=0,rtol=0)
    assert hooks.calls==dict(actor=1,actor_targ=1)
    opt=torch.optim.Adam(model.actor.parameters());opt.zero_grad();a.sum().backward();opt.step()
    assert any(not torch.equal(v,model.state_dict()[k]) for k,v in original.items() if k.startswith('actor.'))
    before=[len(getattr(model,n)._forward_hooks) for n in hooks.names]
    with pytest.raises(ValueError,match='already installed'):FeasibleActorOutputs(model)
    assert before==[len(getattr(model,n)._forward_hooks) for n in hooks.names]


def test_float32_boundary_does_not_force_cruise_just_before_goal():
    obs=torch.zeros(10);obs[0]=torch.tensor((60.-1e-7)/150)
    assert obs[0]==torch.tensor(.4)
    decoded=-1.25+3.75*feasible_request(obs,torch.full((3,),-1.))
    assert decoded[0]==-2.5


def test_invalid_shapes_nonfinite_or_infeasible_memory_are_rejected():
    with pytest.raises(ValueError):feasible_request(torch.zeros(3,10),torch.zeros(3))
    with pytest.raises(ValueError):feasible_request(torch.zeros(10),torch.tensor([float('nan'),0.,0.]))
    obs=torch.zeros(10);obs[2]=2.
    with pytest.raises(ValueError):feasible_request(obs,torch.zeros(3))


def test_new_requests_match_physical_decoding_on_all_sealed_native_states():
    root=Path(__file__).resolve().parents[1]/'artifacts/literature_reproduction/dmps_effective_action_training/evaluation_new'
    if not root.exists():pytest.skip('Sealed native states unavailable')
    count=0
    for path in root.glob('episode_*/episode.json'):
        record=json.loads(path.read_text())
        for entry in record['decisions']:
            state=np.asarray(entry['observation_state']);prev=np.asarray(entry['previous_command'])
            obs=np.append(np.column_stack((state[:,0]/150,state[:,1]/20,prev/5)).ravel(),entry['step']/50)
            requested=feasible_request(torch.tensor(obs,dtype=torch.float32),torch.tensor(entry['planner']['proposal'],dtype=torch.float32)).detach().numpy()
            physical=command_for(requested,state,prev,MergeConfig())
            np.testing.assert_allclose(physical,-1.25+3.75*requested,atol=2e-6,rtol=0)
            count+=1
    assert count==500
