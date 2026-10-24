from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

import freeze_fair_comparison_v39 as planner
from coopgate.fair_comparison_v39 import (
    MASKS, METHODS, DecisionEvidence, WorkLedger, fingerprint,
)


@pytest.fixture(scope='module')
def real_plan():
    return planner.build_plan(ROOT, created_utc='2026-10-24T00:00:00+00:00',
                              runtime={'test_fixture': True})


def evidence(case):
    return DecisionEvidence(case=case, step=0,
        observation_fingerprint=fingerprint(('synthetic-routing-smoke', case.case_id)),
        proposal=(1., 2., 2.5), baseline=(-1., -2., -2.5),
        active=(True, True, True), confidence=.9, predicted_delta_s=-.2,
        calibration_quantile_s=.05, in_calibration_domain=True,
        prediction_complete=True, advantage_samples_s=tuple(np.linspace(.5, .7, 33)),
        safety_flags_by_mask=tuple((True, True, True) for _ in MASKS),
        proposed_bank_command=(.5, 1., 2.), proposed_bank_admitted=True,
        proposed_bank_physical_admissible=True,
        proposed_bank_prediction_complete=True,
        proposed_bank_replay_sha256=fingerprint(('actual-v35-bank-replay', case.case_id)),
        proposed_bank_candidate_count=7,
        ledger=WorkLedger(observation_reads=1, learned_proposal_calls=1,
            baseline_proposal_calls=1, forecast_paths=33, safety_mask_checks=8,
            proposed_gate_replays=1, proposed_gate_forecast_calls=100,
            elapsed_ms=100., timed_out=False))


def test_plan_is_exact_eight_way_cross_with_explicit_resource_ceiling(real_plan):
    assert real_plan['methods'] == list(METHODS)
    assert real_plan['primary_decision']['proposed_method'] == 'proposed_bank_performance_gate'
    assert real_plan['method_count'] == 8
    assert real_plan['independent_clusters'] == 640
    assert real_plan['blocks'] == 160
    assert real_plan['actual_sumo_runs'] == 5_120
    assert real_plan['run_budget'] == {
        'sumo_episodes': 5_120,
        'maximum_decisions_per_episode': 50,
        'maximum_control_decisions': 256_000,
        'fixed_forecast_paths_per_packet': 33,
        'maximum_proposed_gate_forecast_calls_per_packet': 957,
        'maximum_fixed_path_forecasts': 8_448_000,
        'maximum_raw_forecast_kernel_calls': 16_896_000,
        'maximum_proposed_gate_forecast_calls': 244_992_000,
        'maximum_physical_mask_checks': 2_048_000,
        'common_decision_timeout_ms': 500.,
        'aggregate_controller_deadline_hours': pytest.approx(35.55555555555556),
        'maximum_simulated_episode_hours': pytest.approx(35.55555555555556),
        'clusters_per_block': 4,
        'blocks': 160,
    }
    for group in real_plan['groups']:
        assert len(group['case_ids']) == 4
        assert group['actual_sumo_runs'] == 32
        for case_id in group['case_ids']:
            rows = [row for row in group['assignments'] if row['case_id'] == case_id]
            assert {row['method'] for row in rows} == set(METHODS)
            assert sorted(row['order'] for row in rows) == list(range(8))


def test_primary_clusters_are_iid_mixture_draws_not_fixed_stratified_cells(real_plan):
    design = real_plan['design']
    assert design['kind'] == 'iid_frozen_uniform_mixture_with_replacement'
    assert design['no_rejection_no_cell_completion_no_outcome_selection'] is True
    assert design['pooled_exact_binomial_target'] == 'marginal over the frozen uniform mixture'
    assert design['fixed_cell_results'].startswith('secondary descriptive only')
    assert [case['initial_id'] for case in real_plan['cases']] == list(
        range(planner.SEED_BASE, planner.SEED_BASE + planner.PRIMARY_CLUSTERS))
    assert len({case['independent_cluster'] for case in real_plan['cases']}) == 640
    assert all(len([case for case in real_plan['cases']
                        if case['independent_cluster'] == cluster]) == 1
               for cluster in {case['independent_cluster'] for case in real_plan['cases']})
    assert real_plan['comparison_scope'] == {
        'proposed_candidate_set': 'projected/repaired search over at most 28 commands',
        'ordinary_comparator_candidate_set': 'raw learned/baseline masks only',
        'equal_shadow_computation': True,
        'pure_gate_criterion_isolation': False,
        'limitation': ('Candidate sets differ: this study compares complete deployed selectors, '
                       'not only their scalar admission criteria.'),
    }


def test_raw_paired_kernel_budget_cannot_be_halved(real_plan):
    changed = deepcopy(real_plan)
    changed['run_budget']['maximum_raw_forecast_kernel_calls'] //= 2
    with pytest.raises(ValueError, match='execution budget'):
        planner.validate_plan_structure(changed)


def test_real_model_calibration_confidence_baseline_forecast_and_oracle_bytes_are_bound(real_plan):
    assert planner.plan_check(real_plan, ROOT) is real_plan
    hashes = real_plan['dependency_sha256']
    for name in planner.DEPENDENCY_FILES:
        assert hashes[name] == planner.digest(ROOT / name)
    assert real_plan['models']['1802']['sha256'] == (
        '2a53d50100b11ac5ecd3a7749cc06a51851d6ab84f2fd1e416af0ea9b4e71b6d')
    assert real_plan['models']['1812']['sha256'] == (
        '49943cf074e6dfff83e41ed0e7f0a44f07c89e0771e024f8744f197d1ee0c959')
    assert real_plan['models']['1822']['sha256'] == (
        '8df0829c3474050443a0cc9127451550d8585f1a0d7690a8a1a0a0e93660d6a3')
    assert real_plan['protocol']['proposed_gate_implementation_sha256'] == (
        planner.component_sha256(hashes, planner.PROPOSED_GATE_COMPONENT_FILES))
    assert real_plan['calibration']['adapter_only'] is True
    assert real_plan['calibration']['inherited_risk_certificate'] is False
    assert real_plan['calibration']['values_read_during_plan_build'] is False
    assert real_plan['existing_or_running_effectiveness_outcomes_read'] is False


def test_forecast_hash_covers_the_scaled_bank_bytes_actually_executed_by_v35(real_plan):
    names, raw = planner.temporal_stress_bank('sampled')
    executed = np.ascontiguousarray(np.asarray(raw, dtype='<f8') * .2)
    expected = hashlib.sha256(json.dumps(list(names), separators=(',', ':')).encode('utf-8'))
    expected.update(executed.tobytes(order='C'))
    unscaled = hashlib.sha256(json.dumps(list(names), separators=(',', ':')).encode('utf-8'))
    unscaled.update(np.asarray(raw, dtype='<f8').tobytes(order='C'))
    assert real_plan['forecast_bank_sha256'] == expected.hexdigest()
    assert real_plan['forecast_bank_sha256'] != unscaled.hexdigest()
    assert real_plan['forecast_bank_contract']['executed_external_innovation_radius'] == .1


def test_builder_parses_only_the_structural_model_plan(monkeypatch):
    calls = []
    original = planner.read_json

    def audited_read(path):
        calls.append(Path(path).resolve())
        return original(path)

    monkeypatch.setattr(planner, 'read_json', audited_read)
    plan = planner.build_plan(ROOT, created_utc='fixed', runtime={'fixture': True})
    assert calls == [(ROOT / planner.MODEL_PLAN).resolve()]
    assert plan['calibration']['values_read_during_plan_build'] is False


@pytest.mark.parametrize('mutation,match', [
    (lambda value: value['methods'].pop(), 'scope|method'),
    (lambda value: value['groups'][0]['assignments'].pop(), 'crossed schedule'),
    (lambda value: value['cases'][1].update(
        independent_cluster=value['cases'][0]['independent_cluster']), 'cluster identities'),
    (lambda value: value['design'].update(
        no_rejection_no_cell_completion_no_outcome_selection=False), 'IID draws'),
    (lambda value: value['primary_decision'].update(proposed_method='mps_adapter'),
     'Primary decision'),
    (lambda value: value['comparison_scope'].update(pure_gate_criterion_isolation=True),
     'Candidate-set limitation'),
])
def test_plan_check_rejects_scope_schedule_cluster_design_and_proposed_method_tampering(
        real_plan, mutation, match):
    changed = deepcopy(real_plan)
    mutation(changed)
    with pytest.raises(ValueError, match=match):
        planner.validate_plan_structure(changed)


def test_plan_check_rejects_501ms_budget_even_with_a_rehashed_protocol(real_plan):
    changed = deepcopy(real_plan)
    changed['protocol']['budget']['decision_timeout_ms'] = 501.
    changed['protocol_sha256'] = fingerprint(planner.protocol_from_dict(changed['protocol']))
    changed['run_budget']['common_decision_timeout_ms'] = 501.
    with pytest.raises(ValueError, match='500 ms budget'):
        planner.validate_plan_structure(changed)


def test_plan_check_rejects_case_generator_and_real_dependency_tampering(real_plan, tmp_path):
    changed_case = deepcopy(real_plan)
    changed_case['cases'][0]['initial_sha256'] = '0' * 64
    changed_case['groups'] = planner.grouped_schedule(tuple(
        planner.case_from_dict(item) for item in changed_case['cases']))
    with pytest.raises(ValueError, match='IID mixture draw'):
        planner.validate_plan_structure(changed_case)

    changed_hash = deepcopy(real_plan)
    changed_hash['dependency_sha256'][planner.CALIBRATION] = '0' * 64
    with pytest.raises(ValueError, match='dependency bytes changed'):
        planner.validate_dependencies(changed_hash, ROOT)

    dependency = tmp_path / 'dependency.bin'
    dependency.write_bytes(b'original')
    expected = {'dependency.bin': planner.digest(dependency)}
    planner.verify_file_hashes(tmp_path, expected)
    dependency.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='dependency bytes changed'):
        planner.verify_file_hashes(tmp_path, expected)


def test_one_block_synthetic_routing_uses_one_packet_for_all_eight_methods(real_plan):
    group = real_plan['groups'][0]
    cases = {item['case_id']: planner.case_from_dict(item) for item in real_plan['cases']
             if item['case_id'] in group['case_ids']}
    packets = {case_id: evidence(case) for case_id, case in cases.items()}
    rows = planner.route_block(real_plan, 0, packets)
    assert len(rows) == 32
    for case_id in group['case_ids']:
        selected = [row for row in rows if row['case_id'] == case_id]
        assert {row['method'] for row in selected} == set(METHODS)
        assert len({row['evidence_sha256'] for row in selected}) == 1
        proposed = next(row for row in selected
                        if row['method'] == 'proposed_bank_performance_gate')
        assert proposed['decision']['command'] == (.5, 1., 2.)
        assert proposed['decision']['adopted'] is True

    missing = dict(packets)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match='[Oo]ne common evidence packet'):
        planner.route_block(real_plan, 0, missing)
    wrong = dict(packets)
    first = next(iter(wrong))
    wrong[first] = evidence(planner.case_from_dict(real_plan['cases'][4]))
    with pytest.raises(ValueError, match='frozen case identity'):
        planner.route_block(real_plan, 0, wrong)
