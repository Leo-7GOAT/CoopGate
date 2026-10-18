"""Adversarial statistical/reporting cases, with explicitly synthetic outcomes."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
_spec = importlib.util.spec_from_file_location('final_report_v19', TOOLS / 'audit_final_report_v19.py')
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)

from compare_established_v19 import evaluate
from coopgate.risk_comparators_v19 import conditional_ltt, paired_mean_improvement_bands
from probe_disturbed_gate_v19 import digest, save


def manifest(directory):
    save(directory / 'sha256.json', {p.name: digest(p) for p in directory.iterdir()
                                    if p.is_file() and p.name != 'sha256.json'})


def row_for(i, mode, name):
    # Synthetic arithmetic fixture; never represented as an actual SUMO run.
    finish = np.full(3, 10. + .4 * i)
    variation = np.full(3, 2.)
    adopted = name != 'baseline' and (not name.startswith('nominal') or i % 2 == 0)
    if adopted:
        seed = name.split('_')[-1]
        scale = {'1802': 1., '1812': .7, '1822': -.2}[seed]
        improvement = {'raw': .2, 'nominal': .1, 'portfolio0': .08}[name.rsplit('_', 1)[0]]
        finish -= scale * improvement * (i + 1)
        if mode == 'uniform' and name.startswith('raw') and i % 2:
            finish[2] += 2.
    cost = finish + .05 * variation
    failed = i == 7 and mode == 'pulse' and name.startswith('raw')
    return dict(initial_index=i, seed=10000 + i, controller=name, disturbance=mode,
                scenario='dense' if i % 2 else 'nominal', initial=[[-80.-i, 12.], [-85., 13.], [-120., 11.]],
                per_vehicle_finish=finish.tolist(), per_vehicle_variation=variation.tolist(),
                per_vehicle_cost=cost.tolist(), team_cost=float(cost.mean()), completed=not failed,
                collision=False, aborted=failed, contract_violations=[], learned_action_steps=int(adopted),
                controller_ms=[10. + i, 600. if i == 7 else 20.], setup_ms=3.)


@pytest.fixture(scope='module')
def example(tmp_path_factory):
    folder = tmp_path_factory.mktemp('synthetic_final_report')
    plan_dir, cal_dir, final_dir, out = [folder / name for name in ('plan', 'calibration', 'final', 'comparison')]
    for directory in (plan_dir, cal_dir, final_dir):
        directory.mkdir()
    seeds = ['1802', '1812', '1822']
    profiles = [kind + '_' + seed for seed in seeds for kind in ('raw', 'nominal', 'portfolio0')]
    methods = {'baseline': {'kind': 'baseline'}}
    methods.update({name: {'kind': name.rsplit('_', 1)[0], 'seed': int(name.split('_')[-1])} for name in profiles})
    design = dict(seed_base=10000, n_initials=8, sampling='iid_mixture', modes=['clean', 'uniform', 'bias', 'pulse'])
    plan = dict(profiles=profiles, profile_definitions=methods, models={seed: {'synthetic': True} for seed in seeds},
                seeds=seeds, source_sha256={}, conformal=dict(alpha=.05),
                final_test=dict(**design, mode_selection_stream=19417, bootstrap_seed=19423, bootstrap_samples=5000,
                                reported_methods=['raw', 'nominal', 'portfolio0', 'conformal', 'ltt', 'mean_spi'],
                                actual_adoption_minimum=.20, mean_gain_percent_minimum=.5))
    save(plan_dir / 'plan.json', plan)
    manifest(plan_dir)
    # An artificial calibration result lets tests exercise all branches.
    # It is not passed off as an empirical certificate or saved outside tmp.
    calibration = dict(plan_sha256=digest(plan_dir / 'plan.json'), names=profiles, models=plan['models'],
                       calibration_seed_base=20000, calibration_n_initials=200,
                       conformal=dict(infinite_q=False, q=0.),
                       both_targets_certified=['portfolio0_' + seed for seed in seeds],
                       ltt_selection={seed: 'portfolio0_' + seed for seed in seeds},
                       mean_spi_selection={seed: 'nominal_' + seed for seed in seeds})
    save(cal_dir / 'calibration.json', calibration)
    manifest(cal_dir)
    spec = dict(**design, data_role='final_confirmation', methods=methods, expected_actual_sumo_runs=320)
    save(final_dir / 'protocol.json', dict(spec=spec, models=plan['models']))
    rows = [row_for(i, mode, name) for i in range(8) for mode in design['modes'] for name in methods]
    save(final_dir / 'episodes.json', rows)
    save(final_dir / 'summary.json', dict(episodes_sha256=digest(final_dir / 'episodes.json'),
                                         protocol_sha256=digest(final_dir / 'protocol.json')))
    manifest(final_dir)
    evaluate(plan_dir, cal_dir, final_dir, out)
    return dict(lookup={(r['initial_index'], r['disturbance'], r['controller']): r for r in rows},
                spec=spec, plan=plan, calibration=calibration,
                comparison=json.loads((out / 'comparison.json').read_text()),
                selections=json.loads((out / 'selections.json').read_text()))


def test_frozen_report_roundtrip_preserves_models_and_failure_counts(example):
    result = audit.audit_final(**example)
    strict = result['portfolio0']
    assert strict['fixed_model_average']['independent_initials'] == 8
    assert strict['fixed_model_average']['mean_team_gain_seconds'] > 0
    assert strict['passing_training_seeds'] == ['1802', '1812']
    assert not strict['all_replicates_meet_targets']
    third = strict['per_training_seed']['1822']
    assert third['conditional_certificate']['certified']
    assert not third['target_check']['conditions']['mean_gain']
    assert not third['target_check']['conditions']['positive_paired_gain_interval']
    assert all(not r['conditional_certificate']['certified'] for r in result['conformal']['per_training_seed'].values())
    assert all(not r['physical_outcomes']['zero_adoption_outcome_mismatches']
               for method in result.values() for r in method['per_training_seed'].values())


def test_a_report_cannot_replace_initial_clusters_with_pooled_training_replicates(example):
    changed = copy.deepcopy(example)
    changed['comparison']['results']['portfolio0']['independent_initials'] = 24
    with pytest.raises(AssertionError, match='independent_initials'):
        audit.audit_final(**changed)
    changed = copy.deepcopy(example)
    changed['comparison']['results']['portfolio0']['mean_team_gain_ci95'] = [0., 0.]
    with pytest.raises(AssertionError, match='mean_team_gain_ci95'):
        audit.audit_final(**changed)


def test_cross_model_selection_and_hidden_failure_counts_are_detected(example):
    changed = copy.deepcopy(example)
    changed['selections'][0]['profile'] = 'raw_1812'
    with pytest.raises(ValueError, match='whole-profile choice'):
        audit.audit_final(**changed)
    changed = copy.deepcopy(example)
    changed['comparison']['results']['raw']['per_training_seed']['1802']['worst_mode_bundle']['harmful_adopted'] = 0
    with pytest.raises(AssertionError, match='harmful_adopted'):
        audit.audit_final(**changed)


def test_actual_future_does_not_change_the_conformal_forecast_choice(example):
    data = copy.deepcopy(example)
    args = ('conformal', '1802', 1, data['lookup'], data['plan']['profiles'], data['calibration'])
    before = audit.expected_selection(*args)
    data['lookup'][1, 'uniform', before]['per_vehicle_cost'] = [999., 999., 999.]
    assert audit.expected_selection(*args) == before


def test_state_dependent_selection_cannot_inherit_individual_conditional_certificates():
    # Both always-adopted profiles have 3% risk. Choosing according to context
    # exposes the disjoint failure regions and gives the selector 6% risk.
    a = np.arange(100) < 3
    b = (np.arange(100) >= 3) & (np.arange(100) < 6)
    choose_b = b
    assert a.mean() == b.mean() == .03
    assert np.where(choose_b, b, a).mean() == .06
    certificate = audit.certificate_for('conformal', ['a', 'b'], ['a', 'b'])
    assert not certificate['certified']
    assert not audit.certificate_for('conformal', ['a'] * 100, ['a'])['certified']
    assert audit.certificate_for('ltt', ['a'] * 100, ['a'])['certified']


def test_zero_adoption_or_a_positive_mean_alone_cannot_pass_all_requirements():
    metric = dict(actual_random_mode=dict(adoption_fraction=0.), mean_team_gain_percent=10.,
                  mean_team_gain_ci95=[1., 2.])
    design = dict(actual_adoption_minimum=.2, mean_gain_percent_minimum=.5)
    assert not audit.target_check(metric, dict(certified=True), design)['all_conditions_met']
    metric['actual_random_mode']['adoption_fraction'] = .8
    assert not audit.target_check(metric, dict(certified=False), design)['all_conditions_met']
    assert not audit.certificate_for('ltt', ['baseline'], [])['certified']


def test_fallback_identity_retains_baseline_failures_without_certifying_safety():
    base = row_for(0, 'uniform', 'baseline')
    base.update(completed=False, aborted=True)
    result = audit.observation(copy.deepcopy(base), base)
    assert result['failed'] and result['harmful'] and not result['adopted']
    assert result['fallback_outcomes_match']
    conditional = audit.counts([False], [True])
    assert conditional['conditional_harm'] is None
    assert conditional['harmful_adopted'] == 0
    assert audit.exact_bound(0, 0, 1 / 360, .05) == (1., 1., False)


def test_calibration_counts_use_actual_adoption_and_initial_bundles(example):
    data = copy.deepcopy(example)
    spec = data['spec']
    spec['risk_validation'] = dict(alpha=.05, delta=.05, mode_selection_stream=19297)
    names = data['plan']['profiles']
    actual_modes = audit.modes_for(spec, 19297)
    adopted = np.zeros((8, 4, 9), bool)
    harmful = np.zeros_like(adopted)
    gain = np.zeros_like(adopted, float)
    for i in range(8):
        for mi, mode in enumerate(spec['modes']):
            for j, name in enumerate(names):
                r = audit.observation(data['lookup'][i, mode, name], data['lookup'][i, mode, 'baseline'])
                adopted[i, mi, j], harmful[i, mi, j], gain[i, mi, j] = r['adopted'], r['harmful'], r['gain']
    selected_modes = [spec['modes'].index(m) for m in actual_modes]
    indices = np.arange(8)
    ids = list(range(spec['seed_base'], spec['seed_base'] + 8))
    targets = dict(actual_random_mode=conditional_ltt(ids, names, adopted[indices, selected_modes], harmful[indices, selected_modes], delta=.025),
                   worst_mode_bundle=conditional_ltt(ids, names, adopted.any(1), harmful.any(1), delta=.025))
    risk = dict(alpha=.05, total_delta=.05, independent_initials=8, targets=targets, both_targets_certified=[],
                actual_random_modes=dict(zip(map(str, ids), actual_modes)),
                mean_improvement_comparator=paired_mean_improvement_bands(ids, names, gain[indices, selected_modes], simulations=1000))
    result = audit.audit_risk(data['lookup'], spec, risk)
    assert result['nominal_1802']['actual_random_mode']['adopted'] == 4
    assert result['raw_1802']['worst_mode_bundle']['harmful'] == 4
    risk['targets']['actual_random_mode']['candidates'][1]['adopted'] = 8
    with pytest.raises(AssertionError, match='adopted'):
        audit.audit_risk(data['lookup'], spec, risk)


def test_conditional_binomial_boundary_agrees_with_direct_probability():
    assert not audit.exact_bound(114, 0, 1 / 360, .05)[2]
    upper, p, passed = audit.exact_bound(115, 0, 1 / 360, .05)
    assert passed and upper <= .05
    assert p == pytest.approx(.95 ** 115)
    assert audit.exact_bound(205, 2, 1 / 360, .05)[2]
    assert not audit.exact_bound(205, 3, 1 / 360, .05)[2]
