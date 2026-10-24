import numpy as np
import pytest

from coopgate.recovery_risk_v39 import (ADOPTION_MIN, ATTEMPT, ATTEMPT_ALPHA,
    CALIBRATION_N, CONFIRMATION_N, GAIN_MIN_PERCENT, MODELS,
    adoption_certificate, joint_deployment_certificate, paired_gain_certificate)


def observations():
    calibration_adopted = np.ones((CALIBRATION_N, len(MODELS)), dtype=bool)
    calibration_harmful = np.zeros_like(calibration_adopted)
    calibration_failed = np.zeros_like(calibration_adopted)
    calibration_plants = np.resize(
        np.array(['nominal', 'gain_drag', 'lag_mild', 'lag_strong']), CALIBRATION_N)
    confirmation_adopted = np.ones((CONFIRMATION_N, len(MODELS)), dtype=bool)
    variation = np.linspace(-.1, .1, CONFIRMATION_N)
    paired_gain = np.column_stack((1.0 + variation, 1.1 + variation, 1.2 + variation))
    confirmation_failed = np.zeros_like(confirmation_adopted)
    calibration_baseline_failed = np.zeros(CALIBRATION_N, dtype=bool)
    confirmation_baseline_failed = np.zeros(CONFIRMATION_N, dtype=bool)
    return dict(calibration_adopted=calibration_adopted,
        calibration_harmful=calibration_harmful,
        calibration_failed=calibration_failed,
        calibration_plants=calibration_plants,
        confirmation_adopted=confirmation_adopted,
        paired_gain_percent=paired_gain,
        confirmation_failed=confirmation_failed,
        calibration_baseline_failed=calibration_baseline_failed,
        confirmation_baseline_failed=confirmation_baseline_failed)


def test_attempt_two_alpha_and_complete_fixed_family_pass_jointly():
    result = joint_deployment_certificate(**observations())
    assert ATTEMPT == 2 and ATTEMPT_ALPHA == pytest.approx(.05 / 6)
    assert result['fixed_models'] == list(MODELS)
    assert len(result['fixed_inferential_members']) == 12
    assert result['joint_family_p_value'] <= ATTEMPT_ALPHA
    assert result['all_statistical_components_certified']
    assert result['all_baseline_and_candidate_executions_complete']
    assert result['joint_deployment_certified']
    assert result['intersection_union_test']
    assert not result['constituent_bonferroni_used']


def test_low_risk_cannot_hide_zero_adoption():
    values = observations()
    values['confirmation_adopted'][:] = False
    result = joint_deployment_certificate(**values)
    assert result['primary_risk']['all_fixed_models_both_targets_certified']
    assert result['adoption']['minimum_adoption_lower'] == 0
    assert not result['adoption']['all_fixed_models_adoption_certified']
    assert not result['joint_deployment_certified']


def test_observed_twenty_percent_is_not_a_twenty_percent_lower_bound():
    adopted = np.ones((CONFIRMATION_N, len(MODELS)), dtype=bool)
    adopted[:, 0] = False
    adopted[:int(np.ceil(ADOPTION_MIN * CONFIRMATION_N)), 0] = True
    result = adoption_certificate(adopted)
    member = result['members'][str(MODELS[0])]
    assert member['observed_rate'] >= ADOPTION_MIN
    assert member['adoption_lower'] < ADOPTION_MIN
    assert not result['all_fixed_models_adoption_certified']


def test_mean_gain_above_threshold_without_a_lower_bound_is_rejected():
    gain = np.ones((CONFIRMATION_N, len(MODELS)))
    spread = np.resize(np.array([-10., 11.2]), CONFIRMATION_N)
    gain[:, 0] = spread
    result = paired_gain_certificate(gain)
    member = result['members'][str(MODELS[0])]
    assert member['mean_gain_percent'] > GAIN_MIN_PERCENT
    assert member['gain_lower_percent'] < GAIN_MIN_PERCENT
    assert not result['all_fixed_models_gain_certified']


def test_zero_variance_gain_is_un_testable_not_infinite_evidence():
    result = paired_gain_certificate(np.ones((CONFIRMATION_N, len(MODELS))))
    assert all(not item['testable'] and not item['certified']
               for item in result['members'].values())


@pytest.mark.parametrize('sample,location', [('calibration', (0, 0)),
                                              ('confirmation', (0, 1)),
                                              ('calibration_baseline', 0),
                                              ('confirmation_baseline', 0)])
def test_any_observed_execution_failure_blocks_the_joint_claim(sample, location):
    values = observations()
    if sample == 'calibration':
        values['calibration_failed'][location] = True
        values['calibration_harmful'][location] = True
    elif sample == 'confirmation':
        values['confirmation_failed'][location] = True
    elif sample == 'calibration_baseline':
        values['calibration_baseline_failed'][location] = True
    else:
        values['confirmation_baseline_failed'][location] = True
    result = joint_deployment_certificate(**values)
    assert result['observed_execution_failures'] == 1
    assert not result['all_baseline_and_candidate_executions_complete']
    assert not result['joint_deployment_certified']


def test_one_bad_risk_member_blocks_the_indivisible_family():
    values = observations()
    strong = values['calibration_plants'] == 'lag_strong'
    indices = np.flatnonzero(strong)[:80]
    values['calibration_harmful'][indices, -1] = True
    result = joint_deployment_certificate(**values)
    assert not result['primary_risk']['all_fixed_models_both_targets_certified']
    assert not result['all_statistical_components_certified']
    assert not result['joint_deployment_certified']


def test_model_reordering_and_changed_plant_mixture_are_rejected():
    values = observations()
    with pytest.raises(ValueError, match='ordered three-model'):
        joint_deployment_certificate(**values, model_ids=MODELS[::-1])
    values = observations()
    values['calibration_plants'][0] = 'lag_strong'
    with pytest.raises(ValueError, match='balanced plant'):
        joint_deployment_certificate(**values)


def test_failure_must_remain_harm_in_calibration():
    values = observations()
    values['calibration_failed'][0, 0] = True
    with pytest.raises(ValueError, match='failure'):
        joint_deployment_certificate(**values)
