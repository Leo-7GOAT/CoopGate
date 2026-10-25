import copy

import pytest

from verify_dmps_feasible_actor import check_proposal_feasible,verify_network_hooks


def test_target_actor_mapping_is_required_in_addition_to_both_target_critics():
    proof=dict(critic_calls=dict(critic1=46056,critic2=30704,critic1_targ=30704,critic2_targ=30704),
        actor_calls=dict(actor=46072,actor_targ=30704),optimizer_parameter_identity_unchanged=True,
        all_four_critics_hooked=True,both_actor_outputs_hooked=True)
    verify_network_hooks(proof)
    for field,value in [('actor_calls',dict(actor=46072,actor_targ=0)),('both_actor_outputs_hooked',False)]:
        bad=copy.deepcopy(proof);bad[field]=value
        with pytest.raises(AssertionError):verify_network_hooks(bad)


def test_noise_free_raw_request_must_fit_true_recorded_command_interval():
    decision=dict(previous_command=[0.,-5.,2.5])
    assert check_proposal_feasible(decision,dict(proposed=[1.,-1.,1.]))==0.
    with pytest.raises(AssertionError):check_proposal_feasible(decision,dict(proposed=[-1.,1.,-1.]))
