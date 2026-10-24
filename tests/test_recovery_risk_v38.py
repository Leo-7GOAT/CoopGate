import numpy as np
import pytest
from scipy.stats import binom

from coopgate.recovery_risk_v38 import (CALIBRATION_N, MEMBERS, MODELS,
    attempt_error, certify_complete_observations, recovery_family_certificate,
    target_counts)


def counts(k=0, n=200):
    return {name: (k, n) for name in MEMBERS}


def test_second_attempt_uses_remaining_summable_error_budget():
    assert attempt_error(1) == .025
    assert attempt_error(2) == pytest.approx(.05 / 6)
    assert sum(attempt_error(i) for i in range(1, 10000)) < .05


def test_one_bad_model_or_stratum_prevents_the_indivisible_claim():
    data = counts(0, 300)
    data['1822:strong_lag'] = (25, 100)
    result = recovery_family_certificate(data)
    assert not result['all_fixed_models_both_targets_certified']
    assert result['family_p_value'] == result['members']['1822:strong_lag']['pointwise_p_value']
    assert result['intersection_union_test']
    assert not result['constituent_bonferroni_used']


def test_boundary_null_false_family_claim_is_controlled_by_allocated_level():
    level, n = attempt_error(2), 300
    rejected = []
    for k in range(n + 1):
        data = counts(0, n)
        data[MEMBERS[0]] = (k, n)
        if recovery_family_certificate(data)['all_fixed_models_both_targets_certified']:
            rejected.append(k)
    assert float(binom.pmf(rejected, n, .05).sum()) <= level + 1e-14


def test_zero_adoption_and_missing_member_cannot_be_certified():
    data = counts(0, 200)
    data[MEMBERS[0]] = (0, 0)
    assert not recovery_family_certificate(data)['all_fixed_models_both_targets_certified']
    del data[MEMBERS[0]]
    with pytest.raises(ValueError, match='All three'):
        recovery_family_certificate(data)


def test_counts_use_all_clusters_for_mixture_and_only_strong_lag_for_stress_target():
    adopted = np.zeros((CALIBRATION_N, len(MODELS)), dtype=bool)
    harmful = np.zeros_like(adopted)
    plants = np.resize(np.array(['nominal', 'gain_drag', 'lag_mild', 'lag_strong']), CALIBRATION_N)
    adopted[:, 0] = True
    harmful[3, 0] = True
    adopted[plants == 'lag_strong', 1] = True
    result = target_counts(adopted, harmful, plants)
    assert result['1802:mixture'] == (1, CALIBRATION_N)
    assert result['1802:strong_lag'] == (1, CALIBRATION_N // 4)
    assert result['1812:mixture'] == (0, CALIBRATION_N // 4)
    assert result['1812:strong_lag'] == (0, CALIBRATION_N // 4)
    assert result['1822:mixture'] == (0, 0)


def test_failures_are_never_dropped_and_nonadopted_failures_stay_separate():
    adopted = np.ones((CALIBRATION_N, len(MODELS)), dtype=bool)
    harmful = np.zeros_like(adopted)
    failed = np.zeros_like(adopted)
    plants = np.resize(np.array(['nominal', 'gain_drag', 'lag_mild', 'lag_strong']), CALIBRATION_N)
    adopted[0, 0] = False
    harmful[0, 0] = failed[0, 0] = True
    result = certify_complete_observations(adopted, harmful, failed, plants)
    assert result['nonadopted_controller_failures'] == 1
    assert not result['all_controller_executions_complete']
    harmful[0, 0] = False
    with pytest.raises(ValueError, match='failure'):
        certify_complete_observations(adopted, harmful, failed, plants)


@pytest.mark.parametrize('bad', [(True, 3), (4, 3), (-1, 2), (.5, 10)])
def test_invalid_count_pairs_are_rejected(bad):
    data = counts()
    data[MEMBERS[0]] = bad
    with pytest.raises(ValueError):
        recovery_family_certificate(data)
