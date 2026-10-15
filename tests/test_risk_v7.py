import numpy as np
import pytest
from scipy.stats import binom

from coopgate.risk_v7 import binomial_upper, calibrate_threshold, choose


def test_binomial_bound_endpoints_and_tail_inversion():
    assert binomial_upper(0, 0, .05) == 1
    assert binomial_upper(10, 10, .05) == 1
    assert binomial_upper(0, 100, .05) == pytest.approx(1-.05**.01)
    u = binomial_upper(3, 100, .01)
    assert binom.cdf(3, 100, u) == pytest.approx(.01)
    with pytest.raises(ValueError):
        binomial_upper(2, 1, .05)


def test_robust_calibration_rejects_threshold_unsafe_in_one_regime():
    # Broad threshold looks attractive in one regime but fails the other.
    scores = np.r_[np.full(200, -2.), np.full(200, -.5)]
    safe = {'proposal': np.ones(400), 'predicted': scores, 'actual': np.full(400, -1.)}
    shifted = {**safe, 'actual': np.r_[np.full(200, -1.), np.ones(200)]}
    result = calibrate_threshold({'a': safe, 'b': shifted}, [-2., -.5])
    assert result['per_test_delta'] == .05/4
    assert result['threshold_s'] == -2.
    assert not result['candidates'][1]['certified']


def test_empty_acceptance_and_runtime_fallback():
    group = {'proposal': np.zeros(100), 'predicted': np.full(100, -10.), 'actual': np.zeros(100)}
    assert calibrate_threshold({'a': group}, [-2., -1.])['threshold_s'] is None
    assert choose(2, -2., -2.) == 2
    assert choose(2, -1.9, -2.) == 0
    assert choose(2, -10., None) == 0
    assert choose(2, float('nan'), -2.) == 0
    assert choose(0, -2., -2.) == 0
