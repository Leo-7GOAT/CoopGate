from dataclasses import replace

import pytest

from coopgate.fair_comparison_v39 import METHODS, summarize_matched_records as original
from coopgate.fair_failure_analysis_v44 import summarize_matched_records
from test_fair_comparison_v39 import complete_records


def test_successful_comparison_is_numerically_unchanged():
    spec, rows = complete_records(60)
    before, after = original(rows, spec), summarize_matched_records(rows, spec)
    for name, value in before.items():
        if name != 'methods':
            assert after[name] == value
    for method, values in before['methods'].items():
        for name, value in values.items():
            assert after['methods'][method][name] == value
    assert after['deleted_records'] == 0


@pytest.mark.parametrize('failure', [dict(completed=False), dict(timed_out=True),
    dict(collision=True), dict(aborted=True), dict(contract_violations=1)])
def test_pre_adoption_failure_keeps_cost_and_denominator(failure):
    spec, rows = complete_records(60)
    index = next(i for i,r in enumerate(rows) if r.method == 'proposed_bank_performance_gate')
    rows[index] = replace(rows[index], adopted=False, team_cost=25.,
                         per_vehicle_cost=(25.,)*3, per_vehicle_arrival=(25.,)*3, **failure)
    with pytest.raises(ValueError, match='zero-adoption'):
        original(rows, spec)
    report = summarize_matched_records(rows, spec)
    result = report['methods']['proposed_bank_performance_gate']
    # Two conditions share each cluster in this synthetic fixture. The other
    # condition adopts: the cluster still adopts and its retained failure harms.
    assert result['independent_clusters'] == result['adopted_clusters'] == 30
    assert result['failed_cases'] == result['failed_clusters'] == 1
    assert result['failed_before_adoption_cases'] == 1
    assert result['harmful_adopted_clusters'] == 1
    assert not result['comparison_eligible']
    assert report['cases'] == 60 and report['deleted_records'] == 0


def test_failure_without_any_adoption_is_separate_from_conditional_risk():
    spec, rows = complete_records(4)
    for i,r in enumerate(rows):
        if r.method == 'proposed_bank_performance_gate':
            rows[i] = replace(r, adopted=False, completed=False, aborted=True,
                team_cost=25., per_vehicle_cost=(25.,)*3, per_vehicle_arrival=(25.,)*3)
    result = summarize_matched_records(rows, spec)['methods']['proposed_bank_performance_gate']
    assert result['adopted_clusters'] == result['harmful_adopted_clusters'] == 0
    assert result['failed_clusters'] == result['failed_before_adoption_clusters'] == 2
    assert result['simultaneous_conditional_harm_upper'] == 1.
    assert result['mean_paired_cluster_gain_percent'] == -150.
    assert not result['comparison_eligible']


def test_failed_reference_cannot_make_an_apparent_winner():
    spec, rows = complete_records(60)
    for i,r in enumerate(rows):
        if r.method == 'always_baseline':
            rows[i] = replace(r, completed=False)
    old = original(rows, spec)
    assert old['selected_method'] is not None
    report = summarize_matched_records(rows, spec)
    assert report['selected_method'] is None
    for method in METHODS:
        result = report['methods'][method]
        assert result['baseline_failed_clusters'] == 30
        assert not result['relative_improvement_assessable']
        assert not result['comparison_eligible']
        if method != 'always_baseline':
            assert result['harmful_adopted_clusters'] == result['adopted_clusters']
            assert result['mean_paired_cluster_gain_percent'] == old['methods'][method]['mean_paired_cluster_gain_percent']


def test_successful_zero_adoption_drift_still_rejected():
    spec, rows = complete_records(4)
    rows[0] = replace(rows[0], adopted=False)
    with pytest.raises(ValueError, match='successful zero-adoption'):
        summarize_matched_records(rows, spec)


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'protocol', 'case'])
def test_protocol_and_pair_integrity_still_enforced(kind):
    spec, rows = complete_records(4)
    if kind == 'missing':
        rows.pop()
    elif kind == 'duplicate':
        rows.append(rows[0])
    elif kind == 'protocol':
        rows[0] = replace(rows[0], protocol_sha256='0'*64)
    else:
        rows[0] = replace(rows[0], case=replace(rows[0].case, scenario='changed'))
    with pytest.raises(ValueError):
        summarize_matched_records(rows, spec)
