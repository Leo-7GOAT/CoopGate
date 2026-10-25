import numpy as np
import pytest
import torch

from coopgate.dmps_nominal_model import NominalModelEnv,ObservableDMPS,command_for
from coopgate.merge_v4 import MergeConfig


def initial():return np.array([[-140.,12.],[-220.,10.],[-190.,12.]])


def test_nested_model_branches_restore_all_dynamical_state():
    env=NominalModelEnv(initial());root=env.get_state()
    env.step([1,-1,0]);child=env.get_state()
    result=env.step([-1,0,1]);leaf=env.get_state()
    env.set_state(child);second=env.step([-1,0,1])
    np.testing.assert_array_equal(result[0],second[0]);assert result[1:]==second[1:]
    np.testing.assert_array_equal(env.x,leaf.state)
    env.set_state(root);np.testing.assert_array_equal(env.x,initial());assert env.index==0
    assert env.model_decisions==3  # Work count is deliberately not rolled back.


def test_snapshots_cannot_cross_observation_updates_or_environments():
    env=NominalModelEnv(initial());snapshot=env.get_state();env.sync(initial(),np.zeros(3),1)
    with pytest.raises(ValueError):env.set_state(snapshot)
    with pytest.raises(ValueError):NominalModelEnv(initial()).set_state(snapshot)


def test_model_does_not_consume_or_restore_planner_search_rng():
    np.random.seed(805000);before=np.random.get_state()
    env=NominalModelEnv(initial());snapshot=env.get_state();env.step([1,0,-1]);env.set_state(snapshot)
    after=np.random.get_state()
    assert before[0]==after[0] and before[2:]==after[2:]
    np.testing.assert_array_equal(before[1],after[1])


def test_post_goal_cruise_is_not_replaced_by_braking():
    state=initial();state[0]=[70.,20.]
    command=command_for([-1,-1,-1],state,np.zeros(3),MergeConfig())
    np.testing.assert_array_equal(command,[2.5,-2.5,-2.5])


def test_nominal_stable_set_is_invariant_under_backup_with_cleared_car():
    state=np.array([[70.,20.],[-100.,0.],[-140.,0.]])
    env=NominalModelEnv(state,np.array([2.5,-5.,-5.]))
    assert env.stable()
    for _ in range(10):
        env.step(env.backup());assert env.stable()
    np.testing.assert_array_equal(env.x[1:],state[1:])
    assert env.x[0,0]>state[0,0]


def test_cleared_slow_vehicle_is_not_declared_invariant_too_early():
    state=np.array([[70.,10.],[-100.,0.],[-140.,0.]])
    assert not NominalModelEnv(state,np.array([2.5,-5.,-5.])).stable()


@pytest.mark.parametrize('bad_step',[-1,50,True,1.5])
def test_invalid_actual_clock_rejected(bad_step):
    with pytest.raises(ValueError):NominalModelEnv(initial(),step=bad_step)


def test_geometrically_unsafe_state_rejected_before_planning():
    state=initial();state[2,0]=state[0,0]-1
    assert not NominalModelEnv(state).safe()


def test_actual_action_matches_requested_model_action_and_observation_sync():
    class Model:
        critic1=None
        @staticmethod
        def actor(obs):return torch.zeros(3)
    class Planner:
        def __init__(self,env,*args,**kwargs):self.env=env
        def take_step(self,proposal):
            np.testing.assert_array_equal(self.obs,self.env.obs())
            return [self.env.step(proposal)],([proposal],0)
    gate=ObservableDMPS(Model(),Planner)
    state=initial();before=state.copy()
    command,info=gate.choose(state,np.zeros(3),3)
    np.testing.assert_array_equal(command,[-1.25]*3)
    np.testing.assert_array_equal(state,before)
    assert info['nominal_model_decisions']==1
    assert gate.env.index==4
