import numpy as np
import pytest
from scipy.stats import binom

from coopgate.risk_comparators_v19 import conditional_ltt, paired_mean_improvement_bands


def test_ltt_binary_null_test_has_exact_finite_sample_type_one_control():
    alpha, delta, n = .05, .05, 100
    rejected = binom.cdf(np.arange(n + 1), n, alpha) <= delta
    for true_risk in (.05, .08, .2):
        assert binom.pmf(np.arange(n + 1), n, true_risk)[rejected].sum() <= delta + 1e-12


def test_fixed_sequence_stops_and_does_not_certify_later_low_pvalues():
    adoption = np.ones((300, 3), bool)
    harm = np.zeros_like(adoption)
    harm[:60, 1] = True
    result = conditional_ltt(range(300), ('first', 'failure', 'later'), adoption, harm,
                             procedure='fixed_sequence', order=[0, 1, 2])
    assert [r['certified'] for r in result['candidates']] == [True, False, False]
    assert [r['tested'] for r in result['candidates']] == [True, True, False]


def test_nonadopted_scenes_cannot_dilute_conditional_risk():
    adoption = np.zeros((1000, 1), bool)
    adoption[:20] = True
    harm = np.zeros_like(adoption)
    result = conditional_ltt(range(1000), ('rare',), adoption, harm)
    assert not result['candidates'][0]['certified']
    assert result['candidates'][0]['adopted'] == 20
    assert result['candidates'][0]['threshold_upper'] > .05


def test_mean_test_and_individual_harm_test_have_different_targets():
    gain = np.tile([-.1, .3], 250)[:, None]
    mean = paired_mean_improvement_bands(range(500), ('g',), gain, simulations=1000)
    risk = conditional_ltt(range(500), ('g',), np.ones_like(gain, bool), gain < 0)
    assert mean['candidates'][0]['mean_improvement_selected']
    assert not mean['finite_sample_guarantee']
    assert not risk['candidates'][0]['certified']
    assert risk['candidates'][0]['empirical_conditional_risk'] == .5


def test_repeated_initials_and_zero_empirical_variance_cannot_establish_certainty():
    with pytest.raises(ValueError, match='Unique independent'):
        paired_mean_improvement_bands([1, 1, 2], ('g',), np.ones((3, 1)))
    result = paired_mean_improvement_bands(range(5), ('g',), np.ones((5, 1)))
    assert not result['candidates'][0]['mean_improvement_selected']
    assert result['candidates'][0]['lower'] is None
