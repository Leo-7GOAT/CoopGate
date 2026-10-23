from copy import deepcopy

import numpy as np
import pytest

from coopgate.actuator_intervention_v31 import (
    RecordedPrefixBranch, compare_branch, intervention_state, pair_effect, prefix_values)
from coopgate.actuator_prediction_v30 import predict_suffix
from coopgate.disturbance_v9 import disturbance_tape
from coopgate.joint_planner_v15 import JointRolloutGate
from coopgate.matched_shift_v21 import matched_plant_parameters
from coopgate.shifted_sumo_v19 import PLANTS

STATE = np.array([[-80.,12.],[-85.,13.],[-120.,11.]])


def fixture(plant_name='lag_mild', target=1):
    plant = matched_plant_parameters(31031,plant_name)
    tape = disturbance_tape(31031,'uniform',.1,250)
    whole = predict_suffix(STATE,np.zeros(3),np.zeros(3),0,tape,plant,np.zeros(3),record_trace=True)
    assert whole['complete'] and len(whole['trace']) > target
    trace = deepcopy(whole['trace'])
    bank = np.zeros((33,3,2))
    bank[0,:,0] = -.1
    state = STATE if target == 0 else np.asarray(trace[target-1]['x'])
    previous = np.zeros(3) if target == 0 else np.asarray(trace[target-1]['command'])
    baseline = JointRolloutGate()
    baseline.step = target
    command,_ = baseline.select(state,previous)
    trace[target].update(nominal_joint_action_admitted=True,baseline_command=command.tolist(),
        selected_bank_differences=bank.tolist(),selected_phase_differences=np.zeros((33,3)).tolist())
    row = dict(initial=STATE.tolist(),per_vehicle_cost=np.asarray(whole['values'])[:,0].tolist(),
        per_vehicle_finish=np.asarray(whole['values'])[:,1].tolist(),completed=whole['complete'],
        aborted=whole['error'] is not None,abort_reason=None if whole['error'] is None else whole['error']['message'])
    return row,trace,whole,tape,plant


def test_prefix_costs_stop_each_vehicle_clock_and_keep_variation_at_arrival():
    initial = [[59.,2.],[61.,1.],[-20.,2.]]
    prefix = [dict(t=.5,x=[[61.,2.],[62.,1.],[-19.,2.]],command=[2.,2.,2.]),
              dict(t=1.,x=[[62.,2.],[63.,1.],[-18.,2.]],command=[0.,0.,0.])]
    np.testing.assert_allclose(prefix_values(initial,prefix),[[.6,.5],[0.,0.],[1.2,1.]],atol=1e-12,rtol=0)
    prefix[1]['t'] = 2.
    with pytest.raises(ValueError,match='prefix clock'):
        prefix_values(initial,prefix)


def test_prior_learning_prefix_is_executed_and_global_baseline_clock_continues():
    row,trace,_,_,_ = fixture()
    gate = RecordedPrefixBranch(row,trace,1,'baseline_suffix')
    base,_ = JointRolloutGate().select(STATE,np.zeros(3))
    first,info = gate.select(STATE,np.zeros(3))
    np.testing.assert_array_equal(first,[0.]*3)
    assert not np.array_equal(first,base) and info['diagnostic_archived_prefix']
    second,info = gate.select(np.asarray(trace[0]['x']),first)
    np.testing.assert_array_equal(second,trace[1]['baseline_command'])
    assert gate.step == gate.baseline.step == 2 and not info['diagnostic_intervention']


def test_candidate_and_baseline_roles_execute_different_commands_at_same_boundary():
    row,trace,_,_,_ = fixture(target=0)
    candidate = RecordedPrefixBranch(row,trace,0,'candidate_then_baseline')
    baseline = RecordedPrefixBranch(row,trace,0,'baseline_suffix')
    a,info = candidate.select(STATE,np.zeros(3))
    b,_ = baseline.select(STATE,np.zeros(3))
    np.testing.assert_array_equal(a,trace[0]['command'])
    np.testing.assert_array_equal(b,trace[0]['baseline_command'])
    assert not np.array_equal(a,b) and info['diagnostic_intervention']


def test_prefix_cannot_be_replaced_with_a_nearby_state_or_a_different_baseline():
    row,trace,_,_,_ = fixture(target=0)
    gate = RecordedPrefixBranch(row,trace,0,'candidate_then_baseline')
    shifted = STATE.copy()
    shifted[0,0] += .001
    with pytest.raises(AssertionError):
        gate.select(shifted,np.zeros(3))
    trace[0]['baseline_command'][0] += .001
    gate = RecordedPrefixBranch(row,trace,0,'candidate_then_baseline')
    with pytest.raises(AssertionError):
        gate.select(STATE,np.zeros(3))
    with pytest.raises(ValueError,match='Unknown'):
        RecordedPrefixBranch(row,trace,0,'different_controller')


@pytest.mark.parametrize('plant_name',PLANTS)
def test_boundary_forecast_restores_original_full_cost_state_memory_and_errors(plant_name):
    row,trace,_,tape,plant = fixture(plant_name)
    state,previous,memory = intervention_state(row,trace,1)
    prediction = predict_suffix(state,previous,memory,1,tape,plant,record_trace=True)
    review = compare_branch(prediction,row,trace,row,trace,1)
    assert review['all_match'] and review['suffix_commands'] == len(trace)-1
    assert review['metric_max_abs_error'] < 1e-9


@pytest.mark.parametrize('change',['prefix','memory','cost','clock_error_count'])
def test_independent_review_detects_corrupt_prefix_memory_cost_or_error_accounting(change):
    row,trace,_,tape,plant = fixture()
    state,previous,memory = intervention_state(row,trace,1)
    prediction = predict_suffix(state,previous,memory,1,tape,plant,record_trace=True)
    actual_row,actual_trace = deepcopy(row),deepcopy(trace)
    if change == 'prefix': actual_trace[0]['actuator_states'][0][0] += .01
    elif change == 'memory': prediction['trace'][0]['actuator_states'][0][0] += .01
    elif change == 'cost': actual_row['per_vehicle_cost'][0] += .05
    else: prediction['error_contract_exceeded_substeps'] += 1
    assert not compare_branch(prediction,actual_row,actual_trace,row,trace,1)['all_match']


def test_incomplete_low_capped_cost_is_not_admissible_or_a_reported_improvement():
    base = dict(complete=True,values=[[10.,10.]]*3)
    candidate = dict(complete=False,values=[[0.,0.]]*3)
    result = pair_effect(dict(baseline_suffix=base,candidate_then_baseline=candidate))
    assert not result['admissible_performance_evidence'] and result['difference'] is None
    assert result['local_regression'] is None
    candidate['complete'] = True
    candidate['values'] = [[9.,10.],[9.,10.],[10.1,10.]]
    result = pair_effect(dict(baseline_suffix=base,candidate_then_baseline=candidate))
    assert result['mean_cost_difference'] < 0 and result['local_regression']
    assert not result['admissible_performance_evidence']
