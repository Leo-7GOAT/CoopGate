import copy

import numpy as np
import pytest

from coopgate.prospective_validation_v21 import (
    CALIBRATION_N, FAMILIES, MODELS, attempt_error, calibrate_observations, profile_names,
    select_development_family, target_counts, validate_attempt_history, validate_observations,
)


def development():
    return {f'{family}_{seed}': dict(actual_adopted=40, actual_harmful_adopted=0,
        bundle_adopted=60, bundle_harmful_adopted=0, actual_adoption_rate=40 / 128,
        actual_mean_gain_percent=.8, actual_mean_gain_pointwise_bootstrap95=[.01, .1], failed_runs=0)
        for family in FAMILIES for seed in MODELS}


def observations(n=3):
    names = profile_names('phase05')
    a = np.zeros((n, 4, len(names)), bool)
    return names, a, a.copy(), np.zeros((*a.shape, 3, 2)), a.copy()


def test_selection_retains_all_models_and_uses_development_only():
    data = development()
    for seed in MODELS:
        data[f'phase10_{seed}'].update(actual_adopted=50, actual_adoption_rate=50 / 128)
    result = select_development_family(data, 128)
    assert result['selected_family'] == 'phase10'
    assert not result['risk_certified'] and not result['calibration_outcomes_used']
    assert not result['cross_model_family_splicing']
    # A single member below the effectiveness target excludes its entire family.
    data['phase10_1822']['actual_mean_gain_percent'] = .49
    assert select_development_family(data, 128)['selected_family'] == 'phase05'


def test_best_model_per_family_cannot_be_spliced_into_a_pass():
    data = development()
    for family, good_seed in zip(FAMILIES, MODELS):
        for seed in MODELS:
            if seed != good_seed: data[f'{family}_{seed}']['actual_mean_gain_percent'] = .1
    result = select_development_family(data, 128)
    assert result['selected_family'] is None
    assert not any(row['eligible'] for row in result['candidates'].values())


@pytest.mark.parametrize('field,value', [('failed_runs', 1), ('actual_harmful_adopted', 3),
    ('actual_mean_gain_pointwise_bootstrap95', [-.01, .1])])
def test_development_failure_excess_harm_and_uncertain_gain_exclude_a_family(field, value):
    data = development()
    data['phase05_1802'][field] = value
    if field == 'actual_harmful_adopted': data['phase05_1802']['bundle_harmful_adopted'] = value
    assert not select_development_family(data, 128)['candidates']['phase05']['eligible']


def test_missing_models_false_denominators_and_nonfinite_gain_are_rejected():
    data = development()
    del data['repair_1812']
    with pytest.raises(ValueError, match='Every candidate'): select_development_family(data, 128)
    for field, value in (('actual_adoption_rate', .99), ('actual_mean_gain_percent', np.nan),
                         ('actual_adopted', True), ('bundle_adopted', 20)):
        data = development()
        data['phase05_1802'][field] = value
        with pytest.raises(ValueError): select_development_family(data, 128)


def test_bundle_counts_initials_and_keeps_harm_in_another_mode():
    names, a, h, _, _ = observations(2)
    j = names.index('phase05_1802')
    a[0, 0, j] = a[1, 1, j] = True
    h[0, 3, j] = h[1, 1, j] = True
    result = target_counts(names, a, h, np.array([0, 1]), 'phase05')
    assert result['1802:actual'] == (1, 2)
    assert result['1802:bundle'] == (2, 2)
    assert result['1812:actual'] == (0, 0)
    with pytest.raises(ValueError, match='one preassigned'):
        target_counts(names, a, h, np.array([[0, 1], [1, 2]]), 'phase05')


def test_observation_checks_cannot_hide_failure_or_a_single_car_regression():
    names, a, h, d, f = observations()
    f[0, 0, 0] = True
    with pytest.raises(ValueError, match='Harm must'): validate_observations(names, a, h, d, f)
    h[0, 0, 0] = True
    validate_observations(names, a, h, d, f)
    d[1, 0, 0, 2, 1] = .5
    with pytest.raises(ValueError, match='Harm must'): validate_observations(names, a, h, d, f)


def test_partial_calibration_and_altered_common_profile_order_are_rejected():
    names, a, h, d, f = observations()
    with pytest.raises(ValueError, match='sample size'):
        calibrate_observations(names, a, h, d, f, family='phase05', seed_base=63500001, attempt=1)
    with pytest.raises(ValueError, match='common candidate'):
        calibrate_observations(names[::-1], a, h, d, f, family='phase05', seed_base=63500001, attempt=1)


def test_complete_synthetic_calibration_exercises_every_adapter_without_reusing_development():
    names, a, h, d, f = observations(CALIBRATION_N)
    active = np.arange(0, CALIBRATION_N, 5)
    for seed in MODELS:
        j = names.index(f'phase05_{seed}')
        a[active, :, j] = True
        d[active, :, j, :, 0] = -.02
    result = calibrate_observations(names, a, h, d, f, family='phase05', seed_base=63500001,
                                    attempt=1, mean_simulations=1000)
    assert result['primary_certificate']['all_fixed_models_both_targets_certified']
    assert result['attempt_error'] == .025
    assert result['primary_certificate']['members']['1802:actual']['adoptions'] == len(active)
    assert result['ltt_selection'] == {str(s): f'phase05_{s}' for s in MODELS}
    assert result['mean_spi_selection'] == result['ltt_selection']
    assert result['conformal']['q'] == 0
    assert result['common_calibration_budget'] and result['final_confirmation_required']
    assert not result['primary_can_be_replaced_by_comparator']


def test_error_budget_is_summable_and_retries_cannot_reset_or_omit_history():
    assert attempt_error(1) == .025
    assert attempt_error(2) == pytest.approx(.05 / 6)
    assert sum(attempt_error(i) for i in range(1, 10001)) == pytest.approx(.05 * 10000 / 10001)
    entry = dict(attempt=1, allocated_error=.025, status='rejected', frozen_protocol_sha256='a', terminal_evidence_sha256='b')
    assert validate_attempt_history([entry], 2)['allocated_error'] == pytest.approx(.05 / 6)
    for bad in ([], [dict(entry, allocated_error=.05)], [dict(entry, status='running')],
                [dict(entry, terminal_evidence_sha256=None)]):
        with pytest.raises(ValueError): validate_attempt_history(bad, 2)
    for attempt in (0, -1, True, 1.5):
        with pytest.raises(ValueError): attempt_error(attempt)
