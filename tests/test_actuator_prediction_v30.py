from copy import deepcopy

import numpy as np
import pytest

from coopgate.actuator_prediction_v30 import advance_actuator, paired_suffix, plant_arrays, predict_suffix
from coopgate.arrival_phase_v20 import arrival_phase_suffix
from coopgate.disturbance_v9 import disturbance_tape
from coopgate.fast_v8 import parameters
from coopgate.matched_shift_v21 import matched_plant_parameters
from coopgate.merge_v4 import MergeConfig
from coopgate.shifted_sumo_v19 import PLANTS, apply_actuator

STATE = np.array([[-80.,12.],[-85.,13.],[-120.,11.]])


def test_constant_input_matches_closed_form_lag_and_semi_implicit_motion():
    plant = dict(gain=[1.]*3, drag=[0.]*3, tau=[.4]*3)
    dt, n = .1, 20
    state = np.array([[0.,2.],[20.,2.],[-20.,2.]])
    memory, command = np.zeros(3), np.ones(3)
    initial = state.copy()
    arrays = plant_arrays(plant, dt)
    for _ in range(n):
        state, memory, _ = advance_actuator(state, memory, command, np.zeros(3), arrays, dt, 20.)
    r = np.exp(-dt/.4)
    v = 2.+dt*(n-r*(1.-r**n)/(1.-r))
    s = initial[:,0]+2.*n*dt+dt**2*(n*(n+1)/2-r/(1-r)*(n-r*(1-r**n)/(1-r)))
    np.testing.assert_allclose(memory, 1-r**n, atol=1e-12, rtol=0)
    np.testing.assert_allclose(state[:,1], v, atol=1e-12, rtol=0)
    np.testing.assert_allclose(state[:,0], s, atol=1e-12, rtol=0)


@pytest.mark.parametrize('plant_name', PLANTS)
def test_independent_prediction_recurrence_matches_simulator_plant_with_clipping(plant_name):
    plant = matched_plant_parameters(63400001, plant_name)
    arrays = plant_arrays(plant, .1)
    x = STATE.copy()
    memory = np.zeros(3)
    rng = np.random.default_rng(30013)
    for _ in range(50):
        command = rng.uniform(-5., 2.5, 3)
        innovation = rng.uniform(-.1, .1, 3)
        expected, am = apply_actuator(command, x[:,1], memory, innovation, plant)
        following, actual_memory, error = advance_actuator(x, memory, command, innovation, arrays, .1, 20.)
        np.testing.assert_allclose(actual_memory, am, atol=1e-12, rtol=0)
        np.testing.assert_allclose(error, expected-command, atol=1e-12, rtol=0)
        np.testing.assert_allclose(following[:,1], np.clip(x[:,1]+.1*expected, 0.,20.), atol=1e-12, rtol=0)
        x, memory = following, actual_memory


@pytest.mark.parametrize('mode', ['clean','uniform','bias','pulse'])
def test_nominal_augmented_predictor_retains_original_complete_cost_time_and_phase(mode):
    c = MergeConfig()
    tape = disturbance_tape(30017, mode, .1, 250)
    plant = matched_plant_parameters(30017, 'nominal')
    state, previous, actuator = STATE.copy(), np.zeros(3), np.zeros(3)
    command = np.zeros(3)
    for first in (False, True):
        expected, phase, complete = arrival_phase_suffix(state, previous, 0, tape, parameters(c), command, first)
        actual = predict_suffix(state, previous, actuator, 0, tape, plant, command if first else None)
        assert actual['complete'] == complete and complete
        np.testing.assert_allclose(actual['values'], expected, atol=1e-9, rtol=0)
        np.testing.assert_allclose(actual['crossing'], phase, atol=1e-9, rtol=0)
    np.testing.assert_array_equal(state, STATE)
    np.testing.assert_array_equal(previous, np.zeros(3))
    np.testing.assert_array_equal(actuator, np.zeros(3))


def test_counterfactuals_have_different_command_errors_under_identical_external_noise():
    plant = matched_plant_parameters(30017, 'lag_strong')
    arrays = plant_arrays(plant, .1)
    memory = np.array([1., -1., .5])
    x, a, error1 = advance_actuator(STATE, memory, np.zeros(3), np.zeros(3), arrays, .1, 20.)
    y, b, error2 = advance_actuator(STATE, memory, np.full(3,2.5), np.zeros(3), arrays, .1, 20.)
    assert not np.allclose(error1, error2)
    assert not np.allclose(a,b) and not np.allclose(x,y)


def test_horizon_and_infeasible_branches_remain_explicit_and_cannot_authorize_comparison():
    plant = matched_plant_parameters(30017, 'lag_strong')
    pair = paired_suffix(STATE, np.zeros(3), np.zeros(3), 50, np.zeros((250,3)), plant, np.zeros(3))
    assert not pair['both_complete'] and pair['difference'] is None
    assert pair['baseline']['value_status'] == 'capped_incomplete'
    assert pair['baseline']['crossing'] == [None]*3
    unsafe = np.array([[0.,10.],[0.,10.],[-1.,10.]])
    result = predict_suffix(unsafe, np.zeros(3), np.zeros(3), 0, np.zeros((250,3)), plant)
    assert result['error'] is not None and not result['complete']
    assert result['steps'] == 0 and result['values'] == [[25.,25.]]*3


@pytest.mark.parametrize('change', ['tau','gain','drag','memory','clock','tape','command'])
def test_reject_invalid_plant_memory_clock_path_or_command(change):
    plant = matched_plant_parameters(30017, 'nominal')
    args = dict(state=STATE, previous=np.zeros(3), actuator=np.zeros(3), step=0,
                innovations=np.zeros((250,3)), plant=deepcopy(plant))
    if change in ('tau','gain','drag'):
        args['plant'][change][0] = -1.
    elif change == 'memory': args['actuator'] = np.full(3,np.nan)
    elif change == 'clock': args['step'] = True
    elif change == 'tape': args['innovations'] = np.zeros((5,3))
    else: args['first_command'] = np.full(3,-5.)
    with pytest.raises(ValueError): predict_suffix(**args)
