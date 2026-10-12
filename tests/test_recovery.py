import numpy as np
import pytest
from coopgate.recovery import RecoveryConfig,backup_acceleration,witnesses,integrate,RecoverySupervisor
from coopgate.platoon import initial_state,PlatoonEnv
from coopgate.chain_recovery import solve_chain,build_chain,chain_modes,brute_modes


def test_joint_backup_set_is_preserved_under_bounded_lead_actions():
    c=RecoveryConfig();rng=np.random.default_rng(71)
    for seed in range(20):
        x=initial_state(seed,c);previous=np.zeros(3)
        for k in range(12):
            reserve,end=witnesses(x,previous,c)
            assert end[0] and reserve[0]>=c.witness_buffer-1e-7
            action=backup_acceleration(previous,c)
            integrate(x[None,:,:],action[None,:],rng.uniform(-9,2),c)
            previous=action


def test_joint_filter_stays_recoverable_for_arbitrary_proposals():
    c=RecoveryConfig();rng=np.random.default_rng(177)
    for seed in range(8):
        x=initial_state(seed,c);previous=np.zeros(3);gate=RecoverySupervisor(c,'joint_rl')
        for k in range(30):
            action,info=gate.select(x,previous,rng.uniform(-5,2.5,3))
            assert info['in_recoverable_set'] and info['chosen_certified']
            assert np.max(np.abs(action-previous))<=c.max_jerk*c.decision_dt+1e-5
            integrate(x[None,:,:],action[None,:],rng.uniform(-9,2),c)
            previous=action


def test_jerk_memory_changes_recovery_margin():
    c=RecoveryConfig();x=initial_state(81,c)
    actual,_=witnesses(x,np.ones(3)*2.5,c,np.ones((1,3))*2.5)
    forgotten,_=witnesses(x,np.ones(3)*2.5,c,np.ones((1,3))*2.5,forget_memory=True)
    assert forgotten[0]>actual[0]+1


def test_independent_switching_can_invalidate_follower_certificate():
    c=RecoveryConfig();x=initial_state(0,c);previous=np.zeros(3);proposal=np.full(3,2.5)
    _,naive=RecoverySupervisor(c,'individual').select(x,previous,proposal)
    _,joint=RecoverySupervisor(c,'joint_rl').select(x,previous,proposal)
    assert naive['in_recoverable_set']
    assert naive['chosen_margin']<0 and not naive['chosen_certified']
    assert joint['chosen_certified'] and joint['chosen_margin']>0


def test_sumoclosure_matches_joint_witness_plant():
    with PlatoonEnv(method='joint_rl') as env:
        env.reset(731,'hard_brake')
        for _ in range(50):
            _,_,t,u,m=env.step([4,4,4])
            assert m['max_prediction_error']<1e-4
            assert m['uncertified_action_steps']==0 and m['outside_set_steps']==0
            assert m['gap_violation_substeps']==0 and not m['collision']
            if t or u:break
        assert m['completed_horizon']


def test_chain_solution_matches_full_rollout_enumeration():
    c=RecoveryConfig();rng=np.random.default_rng(337)
    for seed in range(30):
        x=initial_state(seed,c);previous=np.zeros(3);gate=RecoverySupervisor(c,'joint_rl')
        for _ in range(5):
            proposal=rng.uniform(-5,2.5,3)
            selected,info=gate.select(x,previous,proposal)
            dynamic,certificate=solve_chain(x,previous,np.array(info['proposal']),c)
            np.testing.assert_allclose(dynamic,selected,atol=1e-7)
            assert certificate['certified']==info['chosen_certified']
            assert certificate['margin']==pytest.approx(info['chosen_margin'],abs=1e-7)
            integrate(x[None,:,:],selected[None,:],rng.uniform(-9,2),c);previous=selected


def test_dynamic_program_matches_exhaustive_binary_chains():
    rng=np.random.default_rng(831)
    for n in (1,2,3,8):
        for _ in range(30):
            feasible=rng.random((n,2,2))>.25;cost=rng.uniform(0,4,(n,2))
            exact=brute_modes(feasible,cost);dynamic=chain_modes(feasible,cost)
            if exact is None:assert dynamic is None
            else:np.testing.assert_array_equal(dynamic,exact)
