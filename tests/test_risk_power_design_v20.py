from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from design_risk_power_v20 import exact_cutoffs, joint_lower_bound, single_target_power


def test_vector_cutoffs_match_brute_force_exact_binomial_tests():
    cutoff = exact_cutoffs(512, .05, 1 / 360)
    for n in (0, 50, 114, 115, 153, 205, 307, 410, 512):
        valid = np.flatnonzero(binom.cdf(np.arange(n + 1), n, .05) <= 1 / 360)
        assert cutoff[n] == (valid[-1] if len(valid) else -1)
    assert np.flatnonzero(cutoff >= 0)[0] == 115


def test_random_adoption_power_matches_zero_risk_and_controls_null_error():
    cutoff = exact_cutoffs(1024, .05, 1 / 360)
    for q in (.1, .2, .3, .5):
        assert single_target_power(1024, q, 0., cutoff) == pytest.approx(binom.sf(114, 1024, q), abs=1e-12)
        assert single_target_power(1024, q, .05, cutoff) <= 1 / 360 + 1e-12
    assert single_target_power(1024, .2, .01, cutoff) == pytest.approx(.6049, abs=.0001)


def test_joint_guarantee_uses_union_bound_without_assuming_target_independence():
    assert joint_lower_bound([.99] * 6) == pytest.approx(.94)
    assert joint_lower_bound([.6] * 6) == 0.
    assert joint_lower_bound([.99, .95]) == pytest.approx(.94)
    with pytest.raises(ValueError):
        joint_lower_bound([])
    with pytest.raises(ValueError):
        joint_lower_bound([np.nan])
