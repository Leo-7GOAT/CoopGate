from pathlib import Path
import os
import sys
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import freeze_fair_comparison_v39 as design
import run_candidate_ablation_v43 as runner
from coopgate.candidate_ablation_v43 import (CandidateAblationGate, VARIANTS,
    raw_compositions, rule_for, _RAW_SELECT, paired_mechanism_summary)
from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.disturbance_v9 import disturbance_tape
from coopgate.episode_store_v21 import observations, inventory
from coopgate.matched_shift_v21 import matched_plant_parameters
from coopgate.recovery_joint_gate_v35 import RecoveryJointGate, compositions


@pytest.fixture(scope='module')
def model():
    torch.set_num_threads(1)
    return load_coupled(ROOT/design.load_models(ROOT)['1802']['path'])[0]


def test_raw_injection_preserves_original_module_and_all_other_selector_dependencies():
    assert _RAW_SELECT.__code__ is RecoveryJointGate.select.__code__
    assert RecoveryJointGate.select.__globals__['compositions'] is compositions
    assert _RAW_SELECT.__globals__['compositions'] is raw_compositions
    assert {key for key in _RAW_SELECT.__globals__ if
            _RAW_SELECT.__globals__[key] is not RecoveryJointGate.select.__globals__[key]} == {'compositions'}


@pytest.mark.parametrize('profile', ['nominal', 'lag_strong'])
def test_raw_family_contains_only_safe_unrepaired_masks(model, profile):
    x = design.planned_initial(139_884_001, 'nominal')
    gate = CandidateAblationGate(model, profile, 'raw_bank')
    _, info = gate.select(x, np.zeros(3))
    base, proposal = np.array(info['baseline_command']), np.array(info['actor_proposal'])
    args = (x, np.zeros(3), proposal, base, np.array(info['actuator_memory_lower']),
            np.array(info['actuator_memory_upper']), gate.bounds, gate.packed)
    candidates, count = raw_compositions(*args)
    projected, _ = compositions(*args)
    assert count <= 7 and count == len(candidates)
    for code, weight, command, vertex in candidates:
        assert weight == 1. and vertex == -2
        np.testing.assert_array_equal(command, np.where([bool(code & (1<<i)) for i in range(3)], proposal, base))
        assert any(np.array_equal(command, value[2]) for value in projected)


@pytest.mark.parametrize('variant', ['projected_nominal', 'projected_bank'])
def test_projected_cell_matches_original_frozen_selector(model, variant):
    x = design.planned_initial(139_884_001, 'nominal')
    gate = CandidateAblationGate(model, 'lag_strong', variant)
    expected = RecoveryJointGate(model, 'lag_strong', rule_for(variant))
    actual_command, actual = gate.select(x, np.zeros(3))
    original_command, original = expected.select(x, np.zeros(3))
    np.testing.assert_array_equal(actual_command, original_command)
    for key, value in original.items():
        if key != 'gate_ms': assert actual[key] == value


@pytest.mark.parametrize('family', ['raw', 'projected'])
def test_nominal_and_bank_start_with_identical_candidates_and_nominal_rankings(model, family):
    x = design.planned_initial(139_884_001, 'nominal')
    nominal = CandidateAblationGate(model, 'lag_strong', family+'_nominal')
    bank = CandidateAblationGate(model, 'lag_strong', family+'_bank')
    _, left = nominal.select(x, np.zeros(3))
    _, right = bank.select(x, np.zeros(3))
    for name in ('actor_proposal', 'baseline_command', 'candidate_count', 'original_candidate_count', 'nominal_evidence'):
        assert left[name] == right[name]


@pytest.mark.skipif(os.environ.get('COOPGATE_RUN_SUMO_SMOKE') != '1', reason='Opt-in actual SUMO')
def test_five_variants_persist_and_replay_in_actual_sumo(model, tmp_path):
    seed, scenario, plant, mode = 139_884_001, 'nominal', 'lag_strong', 'clean'
    case = dict(cluster_id='v43-development-only', seed=seed, scenario=scenario, plant=plant,
        disturbance=mode, policy_id='1802', variants=list(VARIANTS),
        initial_sha256=design.array_sha256(design.planned_initial(seed, scenario), decimals=8),
        dynamics_sha256=design.canonical_sha256(matched_plant_parameters(seed, plant, .1)),
        disturbance_sha256=design.array_sha256(disturbance_tape(seed, mode, .1, 250)))
    result = runner.collect_case(case, tmp_path/'case', model)
    assert result['actual_sumo_runs'] == 5 and result['formal_efficacy_samples'] == 0
    rows = list(observations(tmp_path/'case/observations.sqlite',
        dict(cluster_id=case['cluster_id'], scope='development-only-mechanism-check'),
        [[case['cluster_id'], v] for v in VARIANTS]))
    assert len(rows) == 5
    assert RecoveryJointGate.select.__globals__['compositions'] is compositions


def synthetic_records(cluster='fixture'):
    values = [11., 10.8, 10.9, 10.6, 10.65]
    return [dict(cluster_id=cluster, variant=variant, row=dict(per_vehicle_cost=[cost]*3,
        per_vehicle_finish=[9.]*3, team_cost=cost, learned_action_steps=int(variant != 'baseline'),
        completed=True, aborted=False, collision=False, contract_violations=0))
        for variant, cost in zip(VARIANTS, values)]


def test_factorial_interaction_is_a_paired_difference_of_differences():
    result = paired_mechanism_summary(['fixture'], synthetic_records())
    effects = result['paired_contrasts']
    raw = effects['bank_minus_nominal_with_raw_candidates']['gain_percentage_points']
    projected = effects['bank_minus_nominal_with_projected_candidates']['gain_percentage_points']
    assert raw == pytest.approx(-100*.1/11)
    assert projected == pytest.approx(-100*.05/11)
    assert effects['interaction']['gain_percentage_points'] == pytest.approx(projected-raw)
    assert result['descriptive_only'] and not result['statistical_risk_certified']


def test_failed_baseline_and_nonadopted_failure_are_not_removed_from_contrasts():
    records = synthetic_records()
    records[0]['row'].update(completed=False, aborted=True)
    records[1]['row'].update(completed=False, aborted=True, learned_action_steps=0)
    result = paired_mechanism_summary(['fixture'], records)
    assert result['clusters'] == 1 and result['actual_sumo_runs'] == 5
    assert result['baseline_failures'] == 1 and result['failed_pairs_removed'] == 0
    raw = result['method_means']['raw_nominal']
    assert raw['failure_probability'] == 1 and raw['adoption_probability'] == 0
    assert raw['joint_harm_probability'] == 0


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'foreign', 'cluster'])
def test_factorial_summary_rejects_incomplete_or_reused_samples(mutation):
    records, clusters = synthetic_records(), ['fixture']
    if mutation == 'missing': records.pop()
    elif mutation == 'duplicate': records.append(records[0])
    elif mutation == 'foreign': records[0]['cluster_id'] = 'other'
    else: clusters.append('fixture')
    with pytest.raises(ValueError):
        paired_mechanism_summary(clusters, records)


def test_failed_replay_preserves_first_record_and_never_substitutes_a_variant(tmp_path, monkeypatch):
    case = dict(cluster_id='failure-fixture', variants=list(VARIANTS), plant='nominal')
    target = tmp_path/'failed'
    class Env:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setattr(runner, 'MatchedShiftedMergeEnv', Env)
    monkeypatch.setattr(runner, 'execute_episode', lambda case, variant, *args:
        dict(cluster_id=case['cluster_id'], variant=variant, trace=[], row={}))
    def fail(*args):
        assert len(inventory(target/'observations.sqlite')['cases']) == 1
        raise ValueError('Injected audit inconsistency')
    monkeypatch.setattr(runner, 'audit_episode', fail)
    with pytest.raises(ValueError, match='Injected'):
        runner.collect_case(case, target, None)
    assert len(inventory(target/'observations.sqlite')['cases']) == 1
    assert not (target/'complete.json').exists()
    assert not (target/'observations.sqlite.seal.json').exists()
    with pytest.raises(FileExistsError):
        runner.collect_case(case, target, None)
