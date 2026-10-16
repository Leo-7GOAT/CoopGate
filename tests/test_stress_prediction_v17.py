import numpy as np
import torch
from numba import set_num_threads
from coopgate.stress_prediction_v17 import fixed_stress_bank,predict_stress
from coopgate.profile_prediction_v16 import predict_profile
from coopgate.joint_planner_v15 import JointRolloutGate
from coopgate.cooperative_v15 import JointLearner,JointPolicyGate
from coopgate.policy_iteration_v14 import IteratedPolicy


def test_bank_contains_all_constant_bias_corners_but_claims_no_exhaustive_pulses():
    names,tapes=fixed_stress_bank()
    assert len(names)==len(set(names))==17 and tapes.shape==(17,250,3)
    np.testing.assert_array_equal(tapes[0],0.)
    assert len({tuple(t[0]) for t in tapes[1:9]})==8
    assert np.max(np.abs(tapes))==.5
    for tape in tapes[9:]:
        np.testing.assert_array_equal(tape[:10],0.)
        np.testing.assert_array_equal(tape[10:20],tape[30:40])


def test_parallel_stress_forecasts_match_scalar_nominal_controllers():
    torch.set_num_threads(1);set_num_threads(2);torch.manual_seed(171)
    model=JointLearner(IteratedPolicy(16));state=np.array([[[-80.,12.],[-85.,13.],[-120.,11.]]])
    upper,details=predict_stress(state,model)
    base,_=predict_profile(state[0],lambda x:JointRolloutGate())
    learned,_=predict_profile(state[0],lambda x:JointPolicyGate(model))
    np.testing.assert_allclose(details['baseline'][0,0],base,atol=1e-9,rtol=0)
    np.testing.assert_allclose(details['learned'][0,0],learned,atol=1e-9,rtol=0)
    np.testing.assert_allclose(upper,(details['learned']-details['baseline']).max(1),atol=1e-10)
