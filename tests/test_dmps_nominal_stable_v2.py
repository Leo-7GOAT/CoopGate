import numpy as np
import pytest

from coopgate.dmps_nominal_model import NominalModelEnv
from coopgate.dmps_nominal_model_v2 import NominalModelEnvV2


def close_cleared_pair():
    return np.array([[100.,20.],[80.,20.],[93.2,20.]])


def test_preserved_counterexample_old_stable_set_can_leave_geometric_safety():
    old=NominalModelEnv(close_cleared_pair(),np.array([-5.,2.5,2.5]))
    corrected=NominalModelEnvV2(close_cleared_pair(),np.array([-5.,2.5,2.5]))
    assert old.safe() and old.stable() and not corrected.stable()
    old.step(old.backup())
    assert not old.safe()  # Front car decelerates while the following car cruises.


@pytest.mark.parametrize('previous_front',[-2.5,-1.,0.,2.5])
def test_boundary_allowed_memory_preserves_speed_and_minimum_separation(previous_front):
    env=NominalModelEnvV2(close_cleared_pair(),np.array([previous_front,2.5,2.5]))
    assert env.stable()
    for _ in range(20):
        env.step(env.backup());assert env.stable()
        np.testing.assert_array_equal(env.x[:,1],[20.]*3)
        assert env.x[0,0]-env.x[2,0]>=6.8-1e-8


def test_small_positive_active_speed_is_not_an_infinite_horizon_stop():
    state=np.array([[-100.,1e-11],[-120.,0.],[-140.,0.]])
    assert NominalModelEnv(state,np.zeros(3)).stable()
    assert not NominalModelEnvV2(state,np.zeros(3)).stable()


def test_small_cleared_speed_difference_is_not_an_equal_speed_certificate():
    state=close_cleared_pair();state[0,1]-=1e-11
    assert NominalModelEnv(state,np.zeros(3)).stable()
    assert not NominalModelEnvV2(state,np.zeros(3)).stable()


def test_clock_cap_is_not_mislabeled_an_invariant_of_augmented_state():
    env=NominalModelEnvV2(close_cleared_pair(),np.zeros(3),step=49)
    for _ in range(21):env.step(env.backup())
    assert env.index==70 and not env.safe() and not env.stable()
    np.testing.assert_array_equal(env.x[:,1],[20.]*3)
    assert not env.failed  # Only the administrative search clock expired.
