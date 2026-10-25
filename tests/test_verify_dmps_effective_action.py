import numpy as np
import pytest

from verify_dmps_effective_action import reconstruct_replay,verify_hook_counts


def test_rejected_proposal_penalty_is_distinct_from_actual_finite_horizon_transition():
    call=dict(intervention=1,observation=[1.]*10,proposed=[1.]*3,executed=[-1.]*3,
        next_observation=[2.]*10,reward=-.5,done=True)
    replay=reconstruct_replay([call])
    np.testing.assert_array_equal(replay['rew_buf'],[-32.,-.5])
    np.testing.assert_array_equal(replay['act_buf'],[[1.]*3,[-1.]*3])
    np.testing.assert_array_equal(replay['next_obs_buf'],[[0.]*10,[2.]*10])
    np.testing.assert_array_equal(replay['done_buf'],[True,True])


def test_missing_target_or_actor_update_hook_is_rejected():
    counts=dict(critic1=46056,critic2=30704,critic1_targ=30704,critic2_targ=30704)
    proof=dict(critic_calls=counts,optimizer_parameter_identity_unchanged=True,all_four_critics_hooked=True)
    verify_hook_counts(proof)
    for role in counts:
        changed=dict(proof,critic_calls={**counts,role:counts[role]-1})
        with pytest.raises(AssertionError):verify_hook_counts(changed)
