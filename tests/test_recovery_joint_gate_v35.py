import numpy as np
import pytest

from coopgate.actuator_recovery_v34 import ActuatorRecoveryBaseline, class_bounds, stopping_certificate
from coopgate.fast_v8 import parameters
from coopgate.marl_v12 import LEVELS
from coopgate.merge_v4 import MergeConfig, transition_safe
from coopgate.recovery_joint_gate_v35 import (
    RULES, RecoveryJointGate, actuator_certificate_flags, compositions,
)
from coopgate.shifted_sumo_v19 import apply_actuator, plant_parameters


PROFILES = ('nominal','gain_drag','lag_mild','lag_strong')
STATE = np.array([[-140.,12.],[-220.,10.],[-190.,12.]])


class Cautious:
    def proposal(self,observations,mask):
        score = np.broadcast_to(-np.abs(LEVELS),mask.shape).copy()
        score[~mask] = -np.inf
        return score.argmax(-1)


@pytest.mark.parametrize('profile',PROFILES)
def test_local_flags_conjunction_matches_authoritative_joint_certificate(profile):
    rng = np.random.default_rng(35101)
    packed,bounds = parameters(MergeConfig()),class_bounds(profile)
    accepted = rejected = 0
    for _ in range(1000):
        x = np.column_stack((rng.uniform(-150.,70.,3),rng.uniform(0.,20.,3)))
        previous = rng.uniform(-5.,2.5,3)
        command = rng.uniform(np.maximum(-5.,previous-2.5),np.minimum(2.5,previous+2.5))
        mlo = rng.uniform(-5.,0.,3)
        mhi = mlo+rng.uniform(0.,2.5,3)
        flags = actuator_certificate_flags(x,previous,command,mlo,mhi,bounds,packed)
        expected = bool(stopping_certificate(x,previous,command,mlo,mhi,bounds,packed)[0])
        assert bool(flags.all()) == expected
        accepted += expected;rejected += not expected
    assert accepted > 10 and rejected > 100


@pytest.mark.parametrize('profile',PROFILES)
def test_baseline_mode_preserves_original_recovery_commands_through_actual_history(profile):
    original,integrated = ActuatorRecoveryBaseline(profile),RecoveryJointGate(None,profile,'baseline')
    x,prev,memory = STATE.copy(),np.zeros(3),np.zeros(3)
    plant = plant_parameters(35102,profile)
    for _ in range(25):
        a,ia = original.select(x,prev)
        b,ib = integrated.select(x,prev)
        np.testing.assert_array_equal(a,b)
        assert ib['baseline_evidence'] == {k:v for k,v in ia.items() if k != 'gate_ms'}
        assert not ib['actual_learning_used'] and ib['forecast_calls'] == 0
        for _ in range(5):
            applied,memory = apply_actuator(a,x[:,1],memory,np.array([.1,-.1,.1]),plant)
            x[:,1] = np.clip(x[:,1]+applied*.1,0.,20.);x[:,0] += x[:,1]*.1
        prev = a
        if np.all(x[:,0] >= 60.): break


@pytest.mark.parametrize('profile',PROFILES)
@pytest.mark.parametrize('rule',('mps_shared','mamps_shared'))
def test_physical_switching_uses_same_observer_and_stopping_contract(profile,rule):
    gate = RecoveryJointGate(Cautious(),profile,rule)
    x,prev,memory = STATE.copy(),np.zeros(3),np.zeros(3)
    plant = plant_parameters(35103,profile)
    for _ in range(12):
        command,info = gate.select(x,prev)
        assert info['mode_search_flags'][-1] == [True]*3
        for before,after in zip(info['mode_search_masks'],info['mode_search_masks'][1:]):
            assert np.all(np.asarray(after) <= np.asarray(before))
        assert stopping_certificate(x,prev,command,np.asarray(info['actuator_memory_lower']),
            np.asarray(info['actuator_memory_upper']),gate.bounds,gate.packed)[0]
        assert info['forecast_calls'] == 0 and not info['conditional_performance_risk_certified']
        for _ in range(5):
            before = x[:,0].copy()
            applied,memory = apply_actuator(command,x[:,1],memory,np.array([.1,-.1,.1]),plant)
            x[:,1] = np.clip(x[:,1]+applied*.1,0.,20.);x[:,0] += x[:,1]*.1
            assert bool(transition_safe(before[None],x[:,0][None],gate.c)[0])
        prev = command


def artificial_prediction(command,path,*,bad_path=None,incomplete=None):
    # Explicit fixture for selection logic only; it is not an efficacy result.
    base = command is None
    output = [None]*22
    output[0] = np.full((3,2),10. if base else (11. if path == bad_path else 9.))
    output[1] = np.full(3,10. if base else 9.)
    output[2] = not (incomplete == 'baseline' and base or incomplete == 'candidate' and not base)
    output[3],output[4],output[5],output[6],output[21] = 0,1,0,0,True
    return tuple(output)


def test_nominal_and_bank_share_candidates_but_bank_rejects_adverse_path(monkeypatch):
    nominal = RecoveryJointGate(Cautious(),'nominal','nominal_performance')
    bank = RecoveryJointGate(Cautious(),'nominal','bank_performance')
    for gate in (nominal,bank):
        monkeypatch.setattr(gate,'predict',lambda x,p,lo,hi,u,path:artificial_prediction(u,path,bad_path=1))
    a,ia = nominal.select(STATE,np.zeros(3));b,ib = bank.select(STATE,np.zeros(3))
    assert ia['actual_learning_used'] and not ib['actual_learning_used']
    assert ia['candidate_count'] == ib['candidate_count'] > 0
    assert ia['nominal_evidence'] == ib['nominal_evidence']
    np.testing.assert_array_equal(ib['baseline_command'],b)
    assert not np.array_equal(a,b)


@pytest.mark.parametrize('incomplete',('baseline','candidate'))
@pytest.mark.parametrize('rule',('nominal_performance','bank_performance'))
def test_incomplete_forecasts_never_authorize_learning(monkeypatch,incomplete,rule):
    gate = RecoveryJointGate(Cautious(),'nominal',rule)
    monkeypatch.setattr(gate,'predict',lambda x,p,lo,hi,u,path:artificial_prediction(u,path,incomplete=incomplete))
    command,info = gate.select(STATE,np.zeros(3))
    assert not info['actual_learning_used'] and info['finite_bank_performance_margin'] is None
    np.testing.assert_array_equal(command,info['baseline_command'])
    assert info['nominal_evidence'] and not any(r['passed'] for r in info['nominal_evidence'])


def test_bank_admission_requires_every_fixed_path(monkeypatch):
    gate = RecoveryJointGate(Cautious(),'nominal','bank_performance')
    seen = []
    def predict(x,p,lo,hi,u,path):
        seen.append(path)
        return artificial_prediction(u,path)
    monkeypatch.setattr(gate,'predict',predict)
    _,info = gate.select(STATE,np.zeros(3))
    assert info['actual_learning_used']
    assert len(info['selected_bank_differences']) == 33
    assert set(seen) == set(range(33))
    assert info['margin_is_probability'] is False and not info['conditional_performance_risk_certified']


@pytest.mark.parametrize('profile',PROFILES)
def test_all_compositions_keep_cleared_car_at_common_baseline(profile):
    x = np.array([[80.,12.],[-120.,10.],[-170.,12.]])
    gate = ActuatorRecoveryBaseline(profile)
    base,info = gate.select(x,np.zeros(3))
    proposal = np.zeros(3)
    proposal[0] = base[0]
    candidates,_ = compositions(x,np.zeros(3),proposal,base,np.zeros(3),np.zeros(3),gate.bounds,gate.packed)
    for _,_,command,_ in candidates:
        assert command[0] == base[0]
        assert stopping_certificate(x,np.zeros(3),command,np.zeros(3),np.zeros(3),gate.bounds,gate.packed)[0]


@pytest.mark.parametrize('rule',RULES)
def test_reference_point_tracks_actual_executed_command(rule):
    gate = RecoveryJointGate(None if rule == 'baseline' else Cautious(),'lag_strong',rule)
    gate.select(STATE,np.zeros(3))
    _,info = gate.select(STATE,np.full(3,-2.5))
    expected = np.zeros(3)
    retention = np.exp(-.1/.4)
    for _ in range(5): expected = retention*expected+(1.-retention)*-2.5
    np.testing.assert_array_equal(info['reconstructed_actuator'],expected)
    assert not info['future_realization_used'] and not info['true_randomized_plant_used']


def test_bad_actor_output_is_not_silently_reported_as_learning():
    class Bad:
        def proposal(self,obs,mask): return np.array([-1,2,2])
    with pytest.raises(ValueError,match='action mask'):
        RecoveryJointGate(Bad(),'nominal','mps_shared').select(STATE,np.zeros(3))
