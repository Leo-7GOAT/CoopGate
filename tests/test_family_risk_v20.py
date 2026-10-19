import numpy as np
import pytest
from scipy.stats import binom

from coopgate.family_risk_v20 import MEMBERS, certify_declared_families, family_certificate


def counts(k=0, n=200):
    return {name: (k, n) for name in MEMBERS}


def test_one_bad_model_target_prevents_the_entire_family_claim():
    data = counts()
    data['1822:bundle'] = (30, 200)
    result = family_certificate(data)
    assert not result['all_fixed_models_both_targets_certified']
    assert result['family_p_value'] == result['members']['1822:bundle']['pointwise_p_value']
    assert result['maximum_conditional_risk_upper'] > .05
    assert not result['separate_member_intervals_simultaneously_covered']


def test_complete_family_can_pass_without_individual_eighteen_way_certificates():
    result = certify_declared_families({'fixed': counts(3)}, ['fixed'], total_error=.05/3)
    family = result['families']['fixed']
    assert family['all_fixed_models_both_targets_certified']
    assert family['maximum_conditional_risk_upper'] < .05
    assert binom.cdf(3, 200, .05) > .05 / 18


def test_zero_adoption_and_missing_members_cannot_be_certified():
    data = counts()
    data['1802:actual'] = (0, 0)
    result = family_certificate(data)
    assert not result['all_fixed_models_both_targets_certified']
    assert result['family_p_value'] == result['maximum_conditional_risk_upper'] == 1.
    del data['1802:actual']
    with pytest.raises(ValueError, match='All three'):
        family_certificate(data)
    with pytest.raises(ValueError, match='predeclared'):
        certify_declared_families({'one': counts()}, ['one', 'two'])


@pytest.mark.parametrize('families', [1, 3, 9])
def test_exact_boundary_null_error_for_one_bad_target_and_independent_family_union(families):
    # A family can be falsely certified only if its true-null member passes.
    # The exact marginal calculation holds without assumptions about other members.
    level, n = .05 / families, 200
    rejected = [k for k in range(n + 1) if family_certificate(
        {name: ((k, n) if name == MEMBERS[0] else (0, n)) for name in MEMBERS}, family_error=level
    )['all_fixed_models_both_targets_certified']]
    error = float(binom.pmf(rejected, n, .05).sum())
    assert error <= level + 1e-14
    # This independent-family example is a numeric check, not the general proof.
    assert 1. - (1. - error) ** families <= .05 + 1e-14


@pytest.mark.parametrize('pair', [(True, 10), (1, -1), (11, 10), (.5, 10), (0, np.nan)])
def test_invalid_counts_are_rejected(pair):
    data = counts()
    data[MEMBERS[0]] = pair
    with pytest.raises(ValueError):
        family_certificate(data)
