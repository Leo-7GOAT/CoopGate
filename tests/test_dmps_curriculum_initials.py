import numpy as np
import pytest

from coopgate.dmps_curriculum_initials import phase_at,target_state
from coopgate.dmps_nominal_model_v2 import NominalModelEnvV2
from coopgate.merge_v4 import MergeConfig,backup,certify


@pytest.mark.parametrize('count,phase',[(0,0),(8191,0),(8192,1),(16384,2),(24576,3),(32768,3)])
def test_phase_changes_only_at_predeclared_actual_work_boundaries(count,phase):
    assert phase_at(count)==phase


def test_every_near_goal_corner_finishes_even_under_maximum_admissible_braking():
    for shift in (-1.,1.):
        for speed in (19.5,20.):
            state=np.column_stack((np.array([39.,57.,21.])+shift,np.full(3,speed)))
            env=NominalModelEnvV2(state)
            for _ in range(50):
                env.step(env.backup());assert env.safe()
                if np.all(env.x[:,0]>=60):break
            assert np.all(env.x[:,0]>=60)


def test_curriculum_targets_are_reproducible_recoverable_and_all_initially_active():
    c=MergeConfig()
    for phase in range(3):
        for seed in range(159900001,159900017):
            state=target_state(seed,phase)
            np.testing.assert_array_equal(state,target_state(seed,phase))
            assert np.all(state[:,0]<60)
            assert certify(state,np.zeros(3),backup(np.zeros(3),c),c)[0]
    assert target_state(159900001,3) is None
