from copy import deepcopy

import numpy as np
import pytest

from coopgate.regression_diagnostic_v28 import (
    SingleInterventionBaseline, check_admission, decompose_outcome,
    observed_crossings, realized_noise_branches)


def outcome(finish, variation):
    return dict(per_vehicle_finish=finish, per_vehicle_variation=variation,
                per_vehicle_cost=(np.asarray(finish)+.05*np.asarray(variation)).tolist())


def test_small_crossing_delay_can_change_discrete_cost_sign_without_redefining_it():
    base = outcome([9.5, 10., 10.], [2.5, 0., 0.])
    row = outcome([10., 10., 10.], [1.25, 0., 0.])
    original = deepcopy(row)
    result = decompose_outcome(row, [9.500168, 9.9, 9.9], base, [9.487334, 9.9, 9.9])
    assert result['original_cost_delta'][0] == .4375
    assert result['interpolated_cost_diagnostic_delta'][0] < 0
    assert result['decision_rounding_delta'][0] > .48
    assert result['interpolated_crossing_delta'][0] < .013
    assert row == original and not result['original_outcome_definition_changed']


def test_variation_loss_remains_when_logged_arrival_is_identical():
    base, row = outcome([10.]*3, [0.]*3), outcome([10.]*3, [0., 5., 0.])
    result = decompose_outcome(row, [9.9, 9.95, 9.9], base, [9.9]*3)
    assert result['logged_finish_delta'] == [0., 0., 0.]
    assert result['original_cost_delta'] == result['variation_cost_delta'] == [0., .25, 0.]


def test_incomplete_crossing_does_not_get_fabricated_as_a_success():
    data = outcome([25.]*3, [0.]*3)
    with pytest.raises(ValueError, match='Complete finite'):
        decompose_outcome(data, [1., np.nan, 2.], data, [1., 2., 3.])


def test_substep_crossing_preserves_already_arrived_and_unfinished_vehicles():
    initial = [[59.49, 1.], [60.1, 0.], [55., 0.]]
    trace = [dict(t=.5, x=[[59.99, 1.], [60.1, 0.], [55., 0.]], command=[0.]*3, disturbance=[[0.]*3]*5),
             dict(t=1., x=[[60.49, 1.], [60.1, 0.], [55., 0.]], command=[0.]*3, disturbance=[[0.]*3]*5)]
    result = observed_crossings(initial, trace)
    assert result[0] == pytest.approx(.51) and result[1] == 0 and np.isnan(result[2])
    broken = deepcopy(trace)
    broken[1]['x'][0][0] += .01
    with pytest.raises(AssertionError):
        observed_crossings(initial, broken)
    broken = deepcopy(trace)
    broken[0]['disturbance'].pop()
    with pytest.raises(ValueError, match='Complete finite observed'):
        observed_crossings(initial, broken)


@pytest.mark.parametrize('field,value', [('kind', 'lag_mild'), ('gain', [1., .9, 1.]),
    ('drag', [.001, 0., 0.]), ('tau', [.1, 0., 0.])])
def test_effective_noise_is_never_transplanted_to_mismatched_actuator(field, value):
    plant = dict(kind='nominal', gain=[1.]*3, drag=[0.]*3, tau=[0.]*3)
    plant[field] = value
    with pytest.raises(ValueError, match='nominal actuator'):
        realized_noise_branches(dict(plant_name='nominal', plant=plant), [], 0)


def test_bank_and_phase_evidence_cannot_drop_a_path_or_allow_a_violation():
    item = dict(nominal_joint_action_admitted=True, selected_bank_differences=np.zeros((33, 3, 2)).tolist(),
                selected_phase_differences=np.zeros((33, 3)).tolist())
    item['selected_bank_differences'][0][0][0] = -.1
    check_admission(item)
    broken = deepcopy(item)
    broken['selected_bank_differences'].pop()
    with pytest.raises(ValueError, match='complete original'):
        check_admission(broken)
    broken = deepcopy(item)
    broken['selected_phase_differences'][0][0] = .051
    with pytest.raises(ValueError, match='complete original'):
        check_admission(broken)


def test_single_intervention_applies_once_then_uses_feedback_baseline():
    state = np.array([[-80., 12.], [-85., 13.], [-120., 11.]])
    gate = SingleInterventionBaseline(0, state, np.zeros(3), np.zeros(3))
    command, info = gate.select(state, np.zeros(3))
    np.testing.assert_array_equal(command, np.zeros(3))
    assert info['diagnostic_frozen_action'] and gate.step == gate.baseline.step == 1
    _, info = gate.select(state, command)
    assert not info['diagnostic_frozen_action'] and gate.step == gate.baseline.step == 2
    with pytest.raises(ValueError):
        SingleInterventionBaseline(True, state, np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError):
        SingleInterventionBaseline(0, state, np.zeros(3), [float('nan'), 0., 0.])
