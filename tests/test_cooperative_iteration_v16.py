import numpy as np
import torch
from coopgate.cooperative_v15 import JointLearner,JointPolicyGate,CooperativeVector
from coopgate.policy_iteration_v14 import IteratedPolicy
from coopgate.cooperative_iteration_v16 import nominal_returns,unilateral_returns
from coopgate.marl_v12 import LEVELS,action_mask,batch_step
from coopgate.ego_v13 import ego_observations


def test_complete_counterfactual_returns_match_scalar_closed_loop():
    torch.set_num_threads(1);torch.manual_seed(161)
    model=JointLearner(IteratedPolicy(16));env=CooperativeVector(2,161)
    table,incumbent,count=unilateral_returns(env.x,env.previous,env.steps,model,0)
    assert count>len(env.x)
    for k in range(2):
        state=env.x[k].copy();previous=env.previous[k].copy();cost=np.zeros(3)
        controller=JointPolicyGate(model)
        for step in range(50):
            active=state[:,0]<60
            if not active.any():break
            command,_=controller.select(state,previous)
            cost+=active*(.5+.05*np.abs(command-previous))
            nxt,actual,valid=batch_step(state[None],previous[None],command[None],np.zeros((1,5,3)),env.packed,.5)
            assert valid.all();np.testing.assert_array_equal(actual[0],command)
            state=nxt[0];previous=command
        np.testing.assert_allclose(table[k,incumbent[k]],cost.mean(),atol=1e-9,rtol=0)
        assert np.min(table[k])<=table[k,incumbent[k]]


def test_counterfactual_cost_stops_at_each_individual_arrival():
    model=JointLearner(IteratedPolicy(16))
    state=np.array([[[250.,2.],[59.9,2.],[200.,2.]]]);previous=np.array([[-2.5,0.,-2.5]])
    first=np.array([[8,12,8]])
    values,count=nominal_returns(state,previous,np.array([0]),first,model)
    np.testing.assert_allclose(values,[[0.,.625,0.]],atol=1e-10);assert count==1


def test_values_depend_on_the_other_learned_actor_and_not_only_the_updated_car():
    torch.manual_seed(162);model=JointLearner(IteratedPolicy(16));env=CooperativeVector(3,162)
    before,_,_=unilateral_returns(env.x,env.previous,env.steps,model,1)
    with torch.no_grad():model.actors[0].advantage.bias.copy_(torch.tensor(-3.*LEVELS,dtype=torch.float32))
    after,_,_=unilateral_returns(env.x,env.previous,env.steps,model,1)
    mask=np.isfinite(before)&np.isfinite(after)
    assert np.max(np.abs(before[mask]-after[mask]))>.1
