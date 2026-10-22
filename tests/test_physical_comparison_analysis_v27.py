from copy import deepcopy

import numpy as np
import pytest

from coopgate import physical_comparison_analysis_v27 as analysis


METHODS = ['baseline', 'phase05_1802', 'mamps_shared_1802', 'mps_shared_1802']
RULES = ['mamps_shared', 'mps_shared']


def row(method='baseline', finish=(10., 10., 10.), used=0, original=None, index=0, mode='clean'):
    return analysis.annotated(dict(controller=method, initial_index=index, seed=100+index,
        scenario='dense' if index % 2 else 'nominal', disturbance=mode, radius=.5,
        initial=[[-80.-index, 10.], [-90., 12.], [-110., 10.]],
        per_vehicle_finish=list(finish), per_vehicle_variation=[0.]*3,
        per_vehicle_cost=list(finish), team_cost=float(np.mean(finish)), completed=True,
        collision=False, aborted=False, contract_violations=0, control_steps=2,
        learned_action_steps=used if original is None else original), used)


def phase_fixture(n=4):
    spec = dict(n_initials=n, seed_base=100, modes=['clean', 'uniform', 'bias', 'pulse'], mode_selection_stream=21107)
    rows = [row(name, (10.,)*3 if name == 'baseline' else (9.,)*3,
                used=int(name != 'baseline'), index=i, mode=mode)
            for i in range(n) for mode in spec['modes'] for name in METHODS]
    return spec, rows


def transfer_fixture(n=2):
    spec = dict(n_initials_per_scenario=n, seed_base=100, scenarios=['nominal', 'dense'],
        plants=['nominal', 'lag_strong'], mode='uniform', external_innovation_radius=.1, physical_error_radius=.5)
    rows = []
    for si, scene in enumerate(spec['scenarios']):
        for i in range(n):
            for plant in spec['plants']:
                for name in METHODS:
                    item = row(name, used=int(name != 'baseline'), index=i, mode='uniform')
                    item.update(seed=100+1000*si+i, scenario=scene, global_initial_index=si*n+i,
                        plant_name=plant, plant=dict(kind=plant, tau=[0. if plant == 'nominal' else 1.]*3),
                        external_innovation_radius=.1, physical_error_radius=.5,
                        innovation_tape_sha256=f'noise_{si}_{i}', max_actual_error=.1,
                        error_contract_exceeded_substeps=0)
                    rows.append(item)
    return spec, rows


def test_team_improvement_cannot_hide_an_individual_regression():
    base = row()
    learned = row('phase05_1802', (11., 7., 7.), used=1)
    assert learned['team_cost'] < base['team_cost']
    result = analysis.paired_metrics([learned], [base], np.zeros((20, 1), dtype=int))
    assert result['mean_gain_seconds'] > 0
    assert result['harmful_adopted'] == 1 and result['empirical_conditional_risk'] == 1
    assert result['per_vehicle_cost_regressions'] == [1, 0, 0]


def test_arrival_regression_counts_even_when_cost_improves():
    base, learned = row(), row('phase05_1802', (10.5, 9., 9.), used=1)
    base.update(per_vehicle_variation=[20.]*3, per_vehicle_cost=[11.]*3, team_cost=11.)
    analysis.validate_outcomes(base)
    assert all(a < b for a, b in zip(learned['per_vehicle_cost'], base['per_vehicle_cost']))
    assert analysis.harmful(learned, base)


@pytest.mark.parametrize('change', [dict(completed=False), dict(aborted=True), dict(collision=True), dict(contract_violations=1)])
def test_every_failure_is_harm_even_without_adoption(change):
    learned = row('phase05_1802', (9.,)*3, used=0, original=2)
    learned.update(change)
    result = analysis.paired_metrics([learned], [row()], np.zeros((10, 1), dtype=int))
    assert result['failed'] == result['harmful'] == 1
    assert result['adopted'] == result['harmful_adopted'] == 0
    assert result['empirical_conditional_risk'] is None
    assert result['original_adopted'] == result['changed_episode_adoption_events'] == 1


def test_capped_failures_and_baseline_failures_stay_in_cost_average():
    bases = [row(), row(finish=(25.,)*3)]
    bases[1]['completed'] = False
    learners = [row('phase05_1802', (9.,)*3, used=1), row('phase05_1802', (25.,)*3, used=1)]
    learners[1]['completed'] = False
    result = analysis.paired_metrics(learners, bases, np.array([[0, 1], [0, 1]]))
    assert result['mean_gain_seconds'] == .5
    assert result['independent_initials'] == 2 and result['baseline_failed'] == result['failed'] == 1
    assert result['empirical_conditional_risk'] == .5


def test_original_counts_are_not_replaced_by_common_counts():
    source = row('mamps_shared_1802', used=0, original=2)
    changed = analysis.annotated(source, 1)
    assert source['common_learning_steps'] == 0 and source['learned_action_steps'] == 2
    assert changed['common_learning_steps'] == 1 and changed['original_learning_steps'] == 2


@pytest.mark.parametrize('field,value', [('team_cost', float('nan')), ('per_vehicle_cost', [9., 10., 10.]),
    ('common_learning_steps', 3), ('original_learning_steps', -1), ('completed', 'false')])
def test_invalid_metrics_cannot_silently_enter_report(field, value):
    data = row('phase05_1802', used=1)
    data[field] = value
    with pytest.raises(ValueError):
        analysis.validate_outcomes(data)


def test_bundle_retains_cross_mode_adoption_and_harm():
    a = np.array([[True, False, False, False], [False]*4])
    h = np.array([[False, True, False, False], [True, False, False, False]])
    result = analysis.bundle_counts(a, h)
    assert result['adopted'] == 1 and result['harmful'] == 2 and result['harmful_adopted'] == 1
    assert (a & h).sum() == 0


def test_phase_uses_one_frozen_mode_not_four_independent_samples():
    spec, rows = phase_fixture()
    modes = analysis.actual_modes(spec)
    for item in rows:
        if item['controller'] == 'phase05_1802':
            item['common_learning_steps'] = int(item['disturbance'] == modes[item['initial_index']])
    report = analysis.analyze_phase(rows, spec, METHODS, ['1802'], RULES, 50, 27113)
    result = next(r for r in report['methods'] if r['method'] == 'phase05_1802')
    assert result['actual']['independent_initials'] == result['actual']['adopted'] == 4
    assert result['all_mode_diagnostics']['executions'] == 16
    assert result['four_mode_bundle']['adopted'] == 4
    assert report['actual_mode_by_initial'] == modes
    assert len(report['primary_contrasts']) == 2
    assert sum(r['independent_initials'] for r in report['scenario_subgroups'] if r['method'] == 'baseline') == 4


@pytest.mark.parametrize('edit', ['delete', 'duplicate', 'substitute', 'noise', 'initial'])
def test_phase_grid_requires_every_matched_case(edit):
    spec, rows = phase_fixture()
    if edit == 'delete': rows.pop()
    elif edit == 'duplicate': rows.append(deepcopy(rows[0]))
    elif edit == 'substitute': rows[1]['controller'] = 'unknown_1802'
    elif edit == 'noise': rows[1]['radius'] = .6
    else: rows[1]['initial'][0][0] += 1.
    with pytest.raises(ValueError):
        analysis.validate_grid('phase', rows, spec, METHODS)


def test_transfer_keeps_all_cells_models_failures_and_noise_pairing():
    spec, rows = transfer_fixture()
    target = next(r for r in rows if r['scenario'] == 'dense' and r['plant_name'] == 'lag_strong'
                  and r['controller'] == 'phase05_1802' and r['initial_index'] == 0)
    target.update(completed=False, max_actual_error=1., error_contract_exceeded_substeps=5)
    report = analysis.analyze_transfer(rows, spec, METHODS, ['1802'], RULES, 50, 27113)
    assert report['independent_initials'] == 4 and report['initials_per_cell'] == 2
    assert len(report['cells']) == 16 and len(report['primary_contrasts']) == 8
    cell = next(r for r in report['cells'] if r['scenario'] == 'dense' and r['plant'] == 'lag_strong'
                and r['method'] == 'phase05_1802')
    assert cell['failed'] == cell['exceeded_error_runs'] == 1
    target['innovation_tape_sha256'] = 'different'
    with pytest.raises(ValueError, match='Paired external inputs'):
        analysis.validate_grid('transfer', rows, spec, METHODS)


def test_primary_contrast_direction_and_undefined_risk():
    primary = [row('phase05_1802', (8.,)*3, used=0)]
    competitor = [row('mamps_shared_1802', (9.,)*3, used=1)]
    result = analysis.paired_contrast(primary, competitor, [row()], np.zeros((20, 1), dtype=int))
    assert result['mean_increment_seconds'] == 1.
    assert result['increment_pointwise_bootstrap95'] == [1., 1.]
    assert result['adoption_rate_difference'] == -1.
    assert result['empirical_conditional_risk_difference'] is None


def test_bootstrap_is_deterministic_and_resamples_paired_initials():
    draws = analysis.bootstrap_draws(3, 5000, 27113)
    np.testing.assert_array_equal(draws, analysis.bootstrap_draws(3, 5000, 27113))
    # Pair differences have zero variance even though each arm varies strongly.
    a, b = np.array([1., 100., 1000.]), np.array([3., 102., 1002.])
    assert analysis.mean_interval(b-a, draws) == [2., 2.]
    with pytest.raises(ValueError, match='matching initial-cluster'):
        analysis.mean_interval(b-a, np.zeros((10, 12), dtype=int))
