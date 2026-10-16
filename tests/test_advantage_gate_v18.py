import numpy as np
import torch
from coopgate.advantage_gate_v18 import baseline_suffix,action_then_baseline,JointAdvantageGate
from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.joint_planner_v15 import JointRolloutGate
from coopgate.profile_prediction_v16 import predict_profile
from coopgate.fast_v8 import parameters
from coopgate.merge_v4 import MergeConfig


def test_full_planner_potential_matches_bellman_identity_and_scalar_accounting():
    x=np.array([[-80.,12.],[-85.,13.],[-120.,11.]])
    previous=np.zeros(3);packed=parameters(MergeConfig())
    values,first,steps,internal=baseline_suffix(x,previous,0,packed)
    expected,details=predict_profile(x,lambda x:JointRolloutGate())
    np.testing.assert_allclose(values,expected,atol=1e-12,rtol=0)
    branch,_,_=action_then_baseline(x,previous,0,first,packed)
    np.testing.assert_allclose(branch,values,atol=1e-12,rtol=0)
    assert steps==details['forecast_transitions'] and internal>steps


def test_repeated_joint_admission_preserves_each_nominal_cost_and_arrival():
    torch.set_num_threads(1)
    model,_=load_coupled('artifacts/v17/development_models/coupled_1701/policy.pt')
    for x in (np.array([[-77.,13.],[-78.,14.],[-112.,12.]]),
              np.array([[-100.,10.],[-80.,11.],[-140.,9.]])):
        baseline,_=predict_profile(x,lambda x:JointRolloutGate())
        gated,details=predict_profile(x,lambda x:JointAdvantageGate(model))
        assert np.all(gated<=baseline+1e-10)
        assert details['completed']
