import numpy as np
import pytest
from scipy.stats import t

import analyze_joint_risk_power as power
from coopgate.recovery_risk_v39 import _risk_certificate


def test_vectorized_risk_matches_actual_frozen_certificate():
    rng = np.random.default_rng(425599)
    a = np.array([[0]*4, [154]*4, [171]*4, [192]*4, [500]*4,
                  [500, 500, 500, 50], [500]*4, [500]*4])
    k = np.array([[0]*4]*6+[[1, 2, 3, 4], [2, 2, 2, 20]])
    _, _, accepted = power.risk_passes(a, k)
    plants = np.repeat(['nominal', 'gain_drag', 'lag_mild', 'lag_strong'], power.PLANT_COUNT)
    for row in range(len(a)):
        adoption = np.zeros((power.CALIBRATION_N, 3), dtype=bool)
        harm = np.zeros_like(adoption)
        for s in range(4):
            indices = s*power.PLANT_COUNT+rng.permutation(power.PLANT_COUNT)
            adoption[indices[:a[row, s]]] = True
            harm[indices[:k[row, s]]] = True
        actual = _risk_certificate(adoption, harm, np.zeros_like(harm), plants)
        assert bool(accepted[row]) == actual['all_fixed_models_both_targets_certified']


def test_zero_and_one_success_monte_carlo_intervals_keep_boundary_uncertainty():
    zero = power.monte_carlo_interval(0, 20000, .01/22)
    one = power.monte_carlo_interval(20000, 20000, .01/22)
    assert zero['lower'] == 0 < zero['upper'] < .001
    assert .999 < one['lower'] < one['upper'] == 1


def test_adoption_and_normal_gain_boundary_size_and_monotonicity():
    assert power.adoption_power(power.ADOPTION_MIN)['power'] <= power.ATTEMPT_ALPHA
    assert power.adoption_power(.4)['power'] > power.adoption_power(.25)['power']
    assert power.normal_gain_power(power.GAIN_MIN_PERCENT, 3.) == pytest.approx(power.ATTEMPT_ALPHA)
    assert power.normal_gain_power(1., 3.) > power.normal_gain_power(.75, 3.)
    assert power.normal_gain_power(1., 3.) > power.normal_gain_power(1., 5.)
    assert t.sf(t.ppf(1-power.ATTEMPT_ALPHA, 1023), 1023) == pytest.approx(power.ATTEMPT_ALPHA)


def test_joint_bound_never_multiplies_dependent_success_probabilities():
    result = power.all_model_power_bounds(dict(lower=.95, upper=.98), .97, .99)
    assert result['lower'] == pytest.approx(1-3*(.05+.03+.01))
    assert result['upper'] == .97
    assert not result['endpoint_or_model_independence_assumed']
    assert power.all_model_power_bounds(dict(lower=.2, upper=.4), .4, .4)['lower'] == 0


@pytest.mark.parametrize('a,k', [([[1, 2, 3]], [[0, 0, 0]]),
    ([[1., 2., 3., 4.]], [[0, 0, 0, 0]]), ([[1, 2, 3, 4]], [[0, 0, 0, 5]])])
def test_invalid_count_inputs_rejected(a, k):
    with pytest.raises(ValueError):
        power.risk_passes(a, k)
