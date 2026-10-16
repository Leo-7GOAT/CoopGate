"""Interface/physical accounting checks only, not MARL training experiments."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from coopgate.policy_iteration_v14 import IteratedPolicy
from coopgate.cooperative_v15 import JointLearner,JointPolicyGate,CooperativeVector
from coopgate.ego_v13 import EgoPolicyGate
from coopgate.marl_v12 import LEVELS,batch_step


def test_joint_initialization_exactly_preserves_single_ego_greedy_behavior():
    torch.set_num_threads(1);torch.manual_seed(151)
    ego=IteratedPolicy(16);joint=JointLearner(ego);env=CooperativeVector(2,151)
    for i in range(2):
        single=EgoPolicyGate(ego);multi=JointPolicyGate(joint)
        for step in range(8):
            x,prev=env.x[i],env.previous[i]
            a,_=single.select(x,prev);b,_=multi.select(x,prev)
            np.testing.assert_array_equal(a,b)
            # Follow the same certified commands, without any gradient update.
            nxt,command,valid=batch_step(x[None],prev[None],a[None],np.zeros((1,5,3)),env.packed,.5)
            assert valid.all();np.testing.assert_array_equal(command[0],a);env.x[i]=nxt[0]
            env.previous[i]=a


def test_critic_changes_cannot_change_deployed_action_proposals():
    torch.manual_seed(152);model=JointLearner(IteratedPolicy(16));env=CooperativeVector(2,152)
    obs,state,mask=env.inputs();before=model.proposal(obs,mask)
    with torch.no_grad():
        for parameter in model.critics.parameters():parameter.add_(100.)
    np.testing.assert_array_equal(model.proposal(obs,mask),before)
    assert model.value(torch.tensor(obs),torch.tensor(state)).shape==(2,3)
    local=JointLearner(IteratedPolicy(16),'local')
    assert local.value(torch.tensor(obs),torch.tensor(state)).shape==(2,3)


def test_costs_stop_at_individual_arrival_and_terminal_potential_is_zero():
    env=CooperativeVector(1,153);env.x[0]=[[250.,2.],[59.9,2.],[200.,2.]]
    env.previous[0]=[-2.5,0.,-2.5];env.errors[:]=0.
    potential=-(60.-59.9)/(3*20.)
    indices=np.array([[int(np.flatnonzero(LEVELS==0.)[0]),12,int(np.flatnonzero(LEVELS==0.)[0])]])
    _,reward,boundary,active,records=env.step(indices)
    assert boundary.tolist()==[True] and active.tolist()==[[False,True,False]]
    np.testing.assert_allclose(records[0]['cost'],[0.,.625,0.])
    np.testing.assert_allclose(records[0]['time'],[0.,.5,0.])
    np.testing.assert_allclose(reward,[-.625/3-potential],atol=1e-7)


def test_marl_training_guard_rejects_missing_single_ego_evidence(tmp_path):
    path=Path(__file__).resolve().parents[1]/'tools/prerequisite_v15.py'
    spec=importlib.util.spec_from_file_location('prerequisite_v15',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with pytest.raises(RuntimeError,match='acceptance/replay evidence'):module.require_single(tmp_path)


def test_nonfinite_actor_signals_a_fault_instead_of_silently_selecting_braking():
    model=JointLearner(IteratedPolicy(16));env=CooperativeVector(1,154)
    obs,_,mask=env.inputs()
    with torch.no_grad():model.actors[1].advantage.bias.fill_(float('nan'))
    with pytest.raises(ValueError,match='Nonfinite actor'):model.proposal(obs,mask)
