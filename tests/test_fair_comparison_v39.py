from dataclasses import replace

import numpy as np
import pytest

from coopgate.fair_comparison_v39 import (
    COST_CONTRACT, DESCRIPTORS, MASKS, METHODS, CaseIdentity, ComputeBudget,
    DecisionEvidence, EpisodeRecord, FairComparisonProtocol, ProtocolSpec,
    WorkLedger, fair_schedule, fingerprint, summarize_matched_records,
)


def case(index=0, cluster=None):
    return CaseIdentity(case_id=f'case-{index}', independent_cluster=cluster or f'cluster-{index}',
        policy_id='actor-1802', policy_sha256=fingerprint('actor-1802'),
        scenario='dense', initial_id=index, dynamics='lag_strong',
        disturbance=f'tape-{index}', initial_sha256=fingerprint(f'initial-{index}'),
        dynamics_sha256=fingerprint('dynamics-fixed'),
        disturbance_sha256=fingerprint(f'disturbance-{index}'))


def protocol_spec(**changes):
    values = dict(baseline_implementation_sha256=fingerprint('baseline-source'),
        proposed_gate_implementation_sha256=fingerprint('bank-gate-source'),
        calibration_sha256=fingerprint('calibration-bytes'),
        confidence_score_definition_sha256=fingerprint('confidence-score-source'),
        forecast_bank_sha256=fingerprint('fixed-forecast-bank'),
        physical_oracle_sha256=fingerprint('physical-oracle-source'))
    values.update(changes)
    return ProtocolSpec(**values)


def ledger(**changes):
    values = dict(observation_reads=1, learned_proposal_calls=1, baseline_proposal_calls=1,
        forecast_paths=33, safety_mask_checks=8, proposed_gate_replays=1,
        proposed_gate_forecast_calls=100, elapsed_ms=100., timed_out=False)
    values.update(changes)
    return WorkLedger(**values)


def evidence(*, flags=None, active=(True, True, True), work=None, confidence=.9,
             predicted_delta=-.2, quantile=.05, domain=True, complete=True):
    flags = flags or tuple((True, True, True) for _ in MASKS)
    samples = tuple(np.linspace(.5, .7, 33))
    return DecisionEvidence(case=case(), step=3,
        observation_fingerprint=fingerprint('observed-state'),
        proposal=(1., 2., 2.5), baseline=(-1., -2., -2.5), active=active,
        confidence=confidence, predicted_delta_s=predicted_delta,
        calibration_quantile_s=quantile, in_calibration_domain=domain,
        prediction_complete=complete, advantage_samples_s=samples,
        safety_flags_by_mask=flags, proposed_bank_command=(1., 2., 2.5),
        proposed_bank_admitted=True, proposed_bank_physical_admissible=True,
        proposed_bank_prediction_complete=True,
        proposed_bank_replay_sha256=fingerprint('independent-full-bank-replay'),
        proposed_bank_candidate_count=7, ledger=work or ledger())


def test_method_lineage_distinguishes_controls_full_rules_and_adapters():
    assert tuple(DESCRIPTORS) == METHODS
    assert DESCRIPTORS['proposed_bank_performance_gate'].family == 'proposed_method'
    assert DESCRIPTORS['proposed_bank_performance_gate'].decision_rule_status == 'project_full_reproduction'
    assert DESCRIPTORS['proposed_bank_performance_gate'].closed_loop_status == 'project_full_reproduction'
    assert DESCRIPTORS['existing_calibrated_gate'].decision_rule_status == 'full_reproduction'
    assert DESCRIPTORS['existing_calibrated_gate'].closed_loop_status == 'adapter'
    assert DESCRIPTORS['uncalibrated_confidence_threshold'].decision_rule_status == 'full_reproduction'
    for name in ('advantage_lcb_adapter', 'mps_adapter', 'mamps_adapter'):
        assert DESCRIPTORS[name].decision_rule_status == 'adapter'
        assert DESCRIPTORS[name].closed_loop_status == 'adapter'


def test_every_rule_consumes_one_identical_full_budget_evidence_packet():
    protocol = FairComparisonProtocol(protocol_spec())
    decisions = protocol.select_all(evidence())
    assert tuple(item.method for item in decisions) == METHODS
    assert len({item.evidence_sha256 for item in decisions}) == 1
    by_name = {item.method:item for item in decisions}
    assert by_name['always_baseline'].command == (-1., -2., -2.5)
    for name in METHODS:
        if name != 'always_baseline':
            assert by_name[name].command == (1., 2., 2.5)
            assert by_name[name].adopted
    assert by_name['existing_calibrated_gate'].score == pytest.approx(-.15)
    assert by_name['existing_calibrated_gate'].threshold == pytest.approx(-.05)
    # NumPy Boolean evidence must remain hashable and auditable.
    numpy_flags = tuple((np.bool_(True), np.bool_(True), np.bool_(True)) for _ in MASKS)
    assert protocol.select('always_baseline', evidence(flags=numpy_flags)).evidence_sha256


def test_proposed_method_routes_the_actual_bank_command_not_a_generic_mask_command():
    protocol = FairComparisonProtocol(protocol_spec())
    packet = replace(evidence(), proposed_bank_command=(-1., 1., 2.),
                     proposed_bank_candidate_count=19,
                     proposed_bank_replay_sha256=fingerprint('weighted-bank-winner'))
    decision = protocol.select('proposed_bank_performance_gate', packet)
    assert decision.command == (-1., 1., 2.)
    assert decision.learning_mask == (False, True, True)
    assert decision.adopted and decision.physical_admissible
    assert decision.reason == 'proposed_bank_performance_admitted'


@pytest.mark.parametrize('changes,match', [
    ({'proposed_bank_admitted': False}, 'adoption differs'),
    ({'proposed_bank_physical_admissible': False}, 'fallback'),
    ({'proposed_bank_prediction_complete': False}, 'fallback'),
    ({'proposed_bank_replay_sha256': 'bad'}, 'SHA-256'),
    ({'proposed_bank_candidate_count': 29}, 'candidate count'),
])
def test_proposed_method_rejects_inconsistent_or_unverifiable_replay(changes, match):
    with pytest.raises(ValueError, match=match):
        FairComparisonProtocol(protocol_spec()).select(
            'proposed_bank_performance_gate', replace(evidence(), **changes))


def test_mps_is_all_or_none_while_mamps_uses_monotone_common_oracle_mask():
    flags = [(True, True, True) for _ in MASKS]
    flags[7] = (True, False, True)
    flags[5] = (True, True, True)
    protocol = FairComparisonProtocol(protocol_spec())
    decisions = {item.method:item for item in protocol.select_all(evidence(flags=tuple(flags)))}
    assert decisions['mps_adapter'].learning_mask == (False, False, False)
    assert decisions['mps_adapter'].command == (-1., -2., -2.5)
    assert decisions['mamps_adapter'].learning_mask == (True, False, True)
    assert decisions['mamps_adapter'].command == (1., -2., 2.5)
    assert decisions['mamps_adapter'].physical_admissible
    assert decisions['always_learned'].command == (1., 2., 2.5)
    assert not decisions['always_learned'].physical_admissible
    for name in ('uncalibrated_confidence_threshold', 'existing_calibrated_gate',
                 'advantage_lcb_adapter'):
        assert decisions[name].command == (-1., -2., -2.5)


def test_calibrated_rule_reproduces_domain_complete_upper_cost_inequality():
    protocol = FairComparisonProtocol(protocol_spec())
    assert protocol.select('existing_calibrated_gate', evidence()).adopted
    assert not protocol.select('existing_calibrated_gate', evidence(domain=False)).adopted
    assert not protocol.select('existing_calibrated_gate', evidence(complete=False)).adopted
    assert not protocol.select('existing_calibrated_gate',
                               evidence(predicted_delta=-.05, quantile=.05)).adopted


def test_common_timeout_falls_back_for_every_method_and_incomplete_unmarked_work_is_rejected():
    protocol = FairComparisonProtocol(protocol_spec())
    timeout = ledger(observation_reads=0, learned_proposal_calls=0, baseline_proposal_calls=0,
        forecast_paths=0, safety_mask_checks=0, proposed_gate_replays=0,
        proposed_gate_forecast_calls=0, elapsed_ms=500., timed_out=True)
    timeout_packet = replace(evidence(work=timeout),
        proposed_bank_command=(-1., -2., -2.5), proposed_bank_admitted=False,
        proposed_bank_prediction_complete=False, proposed_bank_candidate_count=0,
        proposed_bank_replay_sha256=fingerprint('timed-out-bank-replay'))
    decisions = protocol.select_all(timeout_packet)
    assert all(item.command == (-1., -2., -2.5) and not item.adopted for item in decisions)
    assert {item.reason for item in decisions} == {'common_evidence_timeout'}
    incomplete = ledger(forecast_paths=32)
    with pytest.raises(ValueError, match='identical full budget'):
        protocol.select_all(evidence(work=incomplete))
    false_timeout = ledger(elapsed_ms=10., timed_out=True)
    with pytest.raises(ValueError, match='reaching the common deadline'):
        protocol.select_all(evidence(work=false_timeout))


def test_inactive_agents_and_baseline_physical_failure_cannot_enter_protocol():
    protocol = FairComparisonProtocol(protocol_spec())
    with pytest.raises(ValueError, match='Inactive agents'):
        protocol.select_all(evidence(active=(False, True, True)))
    flags = [(True, True, True) for _ in MASKS]
    flags[0] = (True, False, True)
    with pytest.raises(ValueError, match='common baseline'):
        protocol.select_all(evidence(flags=tuple(flags)))


def test_schedule_contains_every_method_once_per_identical_case_with_fixed_balancing():
    cases = tuple(case(i, f'cluster-{i//2}') for i in range(5))
    schedule = fair_schedule(cases)
    assert len(schedule) == len(cases) * len(METHODS)
    for item in cases:
        selected = [row for row in schedule if row.case == item]
        assert {row.method for row in selected} == set(METHODS)
        assert sorted(row.order for row in selected) == list(range(len(METHODS)))
    assert [row.method for row in schedule[:len(METHODS)]] == list(METHODS)
    assert [row.method for row in schedule[len(METHODS):2*len(METHODS)]] != list(METHODS)


def record(item, method, spec, *, adopted, gain=0., baseline_cost=10., completed=True):
    cost = baseline_cost - gain
    return EpisodeRecord(case=item, method=method, protocol_sha256=fingerprint(spec),
        cost_contract=COST_CONTRACT, timeout_ms=spec.budget.decision_timeout_ms,
        adopted=adopted, completed=completed, timed_out=False, collision=False, aborted=False,
        contract_violations=0, per_vehicle_cost=(cost, cost, cost),
        per_vehicle_arrival=(cost, cost, cost), team_cost=cost)


def complete_records(n=24):
    spec = protocol_spec()
    rows = []
    for index in range(n):
        item = case(index, f'cluster-{index//2}')
        gain = .8 + .02 * index
        for method in METHODS:
            rows.append(record(item, method, spec, adopted=method != 'always_baseline',
                               gain=0. if method == 'always_baseline' else gain))
    return spec, rows


def test_preregistered_summary_uses_independent_clusters_and_excludes_always_baseline():
    spec, rows = complete_records()
    report = summarize_matched_records(rows, spec)
    assert report['cases'] == 24 and report['independent_clusters'] == 12
    assert report['always_baseline_eligible'] is False
    baseline = report['methods']['always_baseline']
    assert baseline['adopted_clusters'] == 0
    assert baseline['simultaneous_conditional_harm_upper'] == 1.
    assert not baseline['usefulness_constraints_passed']
    for method in METHODS:
        if method != 'always_baseline':
            assert report['methods'][method]['usefulness_constraints_passed']
            assert report['methods'][method]['simultaneous_adoption_lower'] >= spec.adoption_minimum
            assert report['methods'][method]['simultaneous_paired_t_gain_lower_percent'] >= spec.gain_minimum_percent
    assert report['selected_method'] == 'proposed_bank_performance_gate'
    assert report['primary_metric'] == spec.primary_metric
    assert sum(report['error_allocation'][name] for name in
               ('risk_family', 'adoption_family', 'gain_family')) == pytest.approx(spec.family_error)


def test_summary_rejects_missing_methods_and_zero_adoption_nonbaseline_drift():
    spec, rows = complete_records(4)
    with pytest.raises(ValueError, match='complete comparator family'):
        summarize_matched_records(rows[:-1], spec)
    changed = list(rows)
    index = next(i for i, row in enumerate(changed) if row.method == 'mps_adapter')
    changed[index] = replace(changed[index], adopted=False)
    with pytest.raises(ValueError, match='zero-adoption comparator'):
        summarize_matched_records(changed, spec)


def test_gain_constraint_uses_mean_paired_cluster_percent_not_ratio_of_means():
    spec, rows = protocol_spec(), []
    expected_percentages = []
    for index in range(60):
        item = case(index, f'cluster-{index}')
        baseline_cost = 20. if index < 30 else 1.
        gain = .2 if index < 30 else -.004
        expected_percentages.append(100. * gain / baseline_cost)
        for method in METHODS:
            rows.append(record(item, method, spec, adopted=method != 'always_baseline',
                gain=0. if method == 'always_baseline' else gain,
                baseline_cost=baseline_cost))
    report = summarize_matched_records(rows, spec)
    result = report['methods']['existing_calibrated_gate']
    assert result['mean_paired_cluster_gain_percent'] == pytest.approx(np.mean(expected_percentages))
    assert result['mean_paired_cluster_gain_percent'] == pytest.approx(.3)
    assert result['ratio_of_means_gain_percent'] > spec.gain_minimum_percent
    assert result['simultaneous_paired_t_gain_lower_percent'] < spec.gain_minimum_percent
    assert not result['usefulness_constraints_passed']
    assert report['selected_method'] is None


def test_protocol_cannot_silently_drop_a_comparator_or_use_an_unequal_mask_budget():
    with pytest.raises(ValueError, match='complete pre-registered'):
        protocol_spec(methods=METHODS[:-1])
    with pytest.raises(ValueError, match='all eight'):
        ComputeBudget(safety_mask_checks=7)
    with pytest.raises(ValueError, match='sum to one'):
        protocol_spec(risk_error_share=.5, adoption_error_share=.5, gain_error_share=.5)
    with pytest.raises(ValueError, match='SHA-256'):
        protocol_spec(calibration_sha256='not-a-real-hash')
