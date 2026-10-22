import numpy as np
import pytest

from coopgate.fast_v8 import parameters
from coopgate.merge_v4 import MergeConfig
from coopgate.physical_switch_v27 import (PhysicalSwitchGate, certificate_flags,
    interval_transition_flags, recovery_search)
from coopgate.robust_v9 import interval_transition, robust_certificate


def test_pair_flags_equal_global_transition_across_intervals():
    rng = np.random.default_rng(2701)
    packed = parameters(MergeConfig())
    for _ in range(10000):
        lo = np.column_stack((rng.uniform(-40,40,3),rng.uniform(0,20,3)))
        hi = lo + rng.uniform(0,3,(3,2))
        qlo, qhi = lo.copy(), hi.copy()
        qlo[:,0] += rng.uniform(0,2,3)
        qhi[:,0] += rng.uniform(2,4,3)
        flags = interval_transition_flags(lo,hi,qlo,qhi,packed)
        assert bool(flags.all()) == bool(interval_transition(lo,hi,qlo,qhi,packed))


def test_local_certificate_conjunction_equals_global_with_short_and_full_horizons():
    rng = np.random.default_rng(2702)
    accepted = rejected = 0
    for tail_steps in (1,24):
        packed = parameters(MergeConfig(tail_steps=tail_steps))
        for _ in range(3000):
            state = np.column_stack((rng.uniform(-150,70,3),rng.uniform(0,20,3)))
            previous = rng.uniform(-5,2.5,3)
            command = rng.uniform(np.maximum(-5,previous-2.5),np.minimum(2.5,previous+2.5))
            radius = rng.choice([0.,.1,.5])
            flags = certificate_flags(state,previous,command,packed,radius)
            expected = bool(robust_certificate(state,previous,command,packed,radius))
            assert bool(flags.all()) == expected
            accepted += expected
            rejected += not expected
    assert accepted > 100 and rejected > 100


def test_slew_and_amplitude_violations_are_not_certified():
    state = np.array([[-100.,0.],[-110.,0.],[-140.,0.]])
    for command in (np.full(3,3.),np.full(3,-5.)):
        assert not certificate_flags(state,np.zeros(3),command,parameters(MergeConfig()),.5).all()


@pytest.mark.parametrize("positions,expected", [([-100.,-110.,-103.],[False,True,False]),
                                                ([0.,0.,-100.],[False,False,True])])
def test_pair_violation_marks_both_incident_agents_only(positions,expected):
    lo = np.column_stack((positions,np.ones(3)))
    following = lo.copy()
    following[:,0] += .1
    np.testing.assert_array_equal(interval_transition_flags(lo,lo,following,following,parameters(MergeConfig())),expected)


def test_mamps_rechecks_cascading_switches_until_jointly_accepted():
    outcomes = iter(([True,False,True],[True,True,False],[True,True,True]))
    seen = []
    def checker(command):
        seen.append(command.copy())
        return np.array(next(outcomes),bool)
    command,mask,masks,flags = recovery_search(np.ones(3),np.zeros(3),checker,"mamps_shared")
    assert masks == [[True,True,True],[True,False,True],[True,False,False]]
    np.testing.assert_array_equal(command,[1.,0.,0.])
    assert len(seen)==3 and flags[-1]==[True]*3


def test_stalled_mask_forces_all_baseline_and_rechecks_it():
    outcomes = iter(([True,False,True],[True,False,True],[True,True,True]))
    command,mask,masks,_ = recovery_search(np.ones(3),np.zeros(3),lambda _:np.array(next(outcomes),bool),"mamps_shared")
    assert masks == [[True,True,True],[True,False,True],[False,False,False]]
    assert not mask.any() and not command.any()


def test_mps_switches_the_whole_team():
    outcomes = iter(([True,False,True],[True,True,True]))
    _,mask,masks,_ = recovery_search(np.ones(3),np.zeros(3),lambda _:np.array(next(outcomes),bool),"mps_shared")
    assert masks == [[True]*3,[False]*3] and not mask.any()


@pytest.mark.parametrize("rule",["mamps_shared","mps_shared"])
def test_uncertified_all_baseline_raises(rule):
    with pytest.raises(RuntimeError,match="baseline failed"):
        recovery_search(np.ones(3),np.zeros(3),lambda _:np.zeros(3,bool),rule)


@pytest.mark.parametrize("bad",[np.ones(3),np.array([True,False]),np.array([True,True,True,None])])
def test_non_boolean_or_malformed_oracle_is_refused(bad):
    with pytest.raises(ValueError,match="Boolean"):
        recovery_search(np.ones(3),np.zeros(3),lambda _:bad,"mamps_shared")


class Aggressive:
    def proposal(self, obs, mask):
        return np.argmax(mask*np.arange(mask.shape[-1]),axis=-1)


@pytest.mark.parametrize("rule",["mamps_shared","mps_shared"])
def test_actual_control_uses_common_baseline_and_preserves_physical_contract(rule):
    gate = PhysicalSwitchGate(Aggressive(),rule)
    state,previous = np.array([[-50.,12.],[-51.,12.],[-90.,11.]]),np.zeros(3)
    command,info = gate.select(state,previous)
    assert robust_certificate(state,previous,command,gate.packed,.5)
    assert info['mode_search_flags'][-1] == [True]*3
    assert info['individual_performance_checked'] is False
    assert info['conditional_performance_risk_certified'] is False
    for a,b in zip(info['mode_search_masks'],info['mode_search_masks'][1:]):
        assert np.all(np.asarray(b)<=np.asarray(a))
    np.testing.assert_array_equal(state,[[-50.,12.],[-51.,12.],[-90.,11.]])
    np.testing.assert_array_equal(previous,np.zeros(3))
