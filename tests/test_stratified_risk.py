import numpy as np
import pytest
from scipy.stats import binomtest

from coopgate.stratified_risk import conditional_risk_certificate, conditional_risk_upper


def test_stratum_bounds_match_independent_scipy_exact_reference():
    k, a, n, w = [0, 7], [30, 100], [200, 300], [.8, .2]
    result = conditional_risk_upper(k, a, n, w, alpha=.05)
    delta = .05 / 4
    qu = [binomtest(x, z, alternative='less').proportion_ci(1-delta).high
          for x, z in zip(k, n)]
    al = [binomtest(x, z, alternative='greater').proportion_ci(1-delta).low
          for x, z in zip(a, n)]
    assert result['risk_upper'] == pytest.approx(min(1., np.dot(w, qu)/np.dot(w, al)))
    assert result['stratum_joint_harm_uppers'] == pytest.approx(qu)
    assert result['stratum_adoption_lowers'] == pytest.approx(al)


def test_zero_adoption_is_uninformative_and_never_passes():
    result = conditional_risk_certificate([0, 0], [0, 0], [500, 500], [.5, .5],
                                           alpha=.01, risk_limit=.05)
    assert result['risk_upper'] == 1
    assert result['pointwise_p_value'] == 1
    assert not result['certified']


def test_population_weights_not_sample_shares_determine_target():
    low = conditional_risk_upper([0, 80], [800, 800], [1000, 1000], [.99, .01], alpha=.01)
    high = conditional_risk_upper([0, 80], [800, 800], [1000, 1000], [.01, .99], alpha=.01)
    assert low['risk_upper'] < .05 < high['risk_upper']


def test_inverted_p_value_and_confidence_bound_agree():
    args = ([0] * 4, [300] * 4, [768] * 4, [.25] * 4)
    result = conditional_risk_certificate(*args, alpha=.05/6, risk_limit=.05)
    assert result['certified']
    p = result['pointwise_p_value']
    assert 0 < p < .05/6
    assert conditional_risk_upper(*args, alpha=p * 1.01)['risk_upper'] < .05
    assert conditional_risk_upper(*args, alpha=p * .99)['risk_upper'] > .05


def test_stronger_confidence_cannot_tighten_the_bound():
    args = ([0, 4], [80, 200], [500, 500], [.5, .5])
    assert conditional_risk_upper(*args, alpha=.001)['risk_upper'] > conditional_risk_upper(
        *args, alpha=.1)['risk_upper']


@pytest.mark.parametrize('args', [
    ([0, 0], [1, 1], [0, 20], [.5, .5]),
    ([2, 0], [1, 1], [20, 20], [.5, .5]),
    ([0, 0], [1, 21], [20, 20], [.5, .5]),
    ([0., 0.], [1, 1], [20, 20], [.5, .5]),
    ([0, 0], [1, 1], [20, 20], [.2, .2]),
    ([0, 0], [1, 1], [20, 20], [0., 1.]),
])
def test_incomplete_or_invalid_strata_cannot_certify(args):
    with pytest.raises(ValueError):
        conditional_risk_upper(*args, alpha=.01)
