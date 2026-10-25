import numpy as np

from coopgate.dmps_nominal_model_v2 import NominalModelEnvV2
from coopgate.merge_v4 import MergeConfig
from diagnose_dmps_curriculum_value import discounted_returns,equivalent_actions,replay_equivalent


def test_finished_vehicle_action_is_physically_ignored_with_command_memory():
    state=np.array([[100.,20.],[-30.,0.],[-65.,0.]])
    previous=np.array([-5.,-5.,-5.]);action=np.array([-.9,-1.,-1.])
    alternatives=equivalent_actions(action,state,previous,MergeConfig())
    assert sum(dim==0 for dim,_ in alternatives)==3
    env=NominalModelEnvV2(state,previous=previous)
    assert replay_equivalent(env,state,previous,8,action,alternatives)==len(alternatives)


def test_slew_plateau_probes_exclude_actions_with_different_commands():
    state=np.array([[-100.,12.],[-170.,10.],[-220.,12.]])
    action=np.array([-1.,-1.,-1.]);previous=np.zeros(3)
    alternatives=equivalent_actions(action,state,previous,MergeConfig())
    assert len(alternatives)==3
    assert all(a[dim]==-1. for dim,a in alternatives)


def test_finite_horizon_return_has_no_post_terminal_bootstrap():
    assert discounted_returns([-1.,-2.,-3.],gamma=.5)==[-2.75,-3.5,-3.]
