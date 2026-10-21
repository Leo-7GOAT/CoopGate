from copy import deepcopy

import numpy as np
import pytest

from coopgate.prospective_data_v21 import MODES, actual_mode_indices, paired_observations, evaluate_adapters


def rows():
    data = []
    for i in range(2):
        for mode in MODES:
            for name in ('baseline', 'raw_1', 'gate_1'):
                cost = [10., 10., 10.]
                used = int(name != 'baseline')
                if name == 'raw_1':
                    cost = [9., 10.1, 9.]
                data.append(dict(initial_index=i, seed=100 + i, disturbance=mode, controller=name,
                    initial=[[-80., 12.], [-85., 13.], [-120., 11.]], scenario='nominal',
                    per_vehicle_finish=cost, per_vehicle_variation=[0., 0., 0.], per_vehicle_cost=cost,
                    team_cost=float(np.mean(cost)), learned_action_steps=used, control_steps=1,
                    controller_ms=[1.], setup_ms=.1,
                    completed=True, collision=False, aborted=False, contract_violations=0))
    return data


def build(data):
    return paired_observations(iter(data), profiles=('raw_1', 'gate_1'), independent_initials=2, seed_base=100,
                               methods={'baseline': {'kind':'baseline'}, 'raw_1': {'kind':'raw'}, 'gate_1': {'kind':'repair'}})


def test_team_improvement_does_not_hide_an_individual_regression():
    paired = build(list(reversed(rows())))
    assert paired['actual_controller_runs'] == 24
    assert paired['independent_initials'] == 2
    assert paired['harmful'][:, :, 0].all()
    assert not paired['harmful'][:, :, 1].any()
    assert (paired['differences'][:, :, 0, :, 0].mean(-1) < 0).all()


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'seed', 'initial', 'cost', 'nonfinite', 'adoption', 'fallback'])
def test_unmatched_or_incomplete_inputs_cannot_enter_risk_calibration(change):
    data = rows()
    if change == 'missing':
        data.pop()
    elif change == 'duplicate':
        data.append(deepcopy(data[0]))
    elif change == 'seed':
        data[1]['seed'] += 1
    elif change == 'initial':
        data[1]['initial'][0][0] += 1
    elif change == 'cost':
        data[1]['per_vehicle_cost'] = [1., 1., 1.]
    elif change == 'nonfinite':
        data[1]['per_vehicle_finish'] = [float('inf')] * 3
    elif change == 'adoption':
        data[1]['learned_action_steps'] = 2
    else:
        data[2].update(learned_action_steps=0, completed=False)
    with pytest.raises(ValueError):
        build(data)


def test_execution_failure_is_harm_even_with_lower_cost_and_no_learning_adoption():
    data = rows()
    data[1].update(completed=False, learned_action_steps=0)
    paired = build(data)
    assert paired['failed'][0, 0, 0]
    assert paired['harmful'][0, 0, 0]
    assert not paired['adopted'][0, 0, 0]


def test_mode_draws_are_one_per_initial_independent_of_outcomes():
    indices = actual_mode_indices(17, 63500001, 22107)
    expected = [np.random.default_rng(np.random.SeedSequence([63500001+i, 22107])).integers(4) for i in range(17)]
    np.testing.assert_array_equal(indices, expected)
    assert indices.shape == (17,)


def comparison_data(improve_disturbed_raw=False):
    data = rows()
    for row in list(data):
        if row['controller'] == 'gate_1':
            nominal = deepcopy(row)
            nominal.update(controller='nominal_1', per_vehicle_finish=[8.5]*3,
                           per_vehicle_cost=[8.5]*3, team_cost=8.5)
            data.append(nominal)
        if improve_disturbed_raw and row['controller'] == 'raw_1' and row['disturbance'] != 'clean':
            row.update(per_vehicle_finish=[5.]*3, per_vehicle_cost=[5.]*3, team_cost=5.)
    return paired_observations(data, profiles=('raw_1','nominal_1','gate_1'), independent_initials=2, seed_base=100,
        methods={name:dict(kind='raw' if name.startswith('raw') else 'nominal')
                 for name in ('baseline','raw_1','nominal_1','gate_1')})


def evaluate(data, infinite=True, ltt='gate_1'):
    calibration = dict(ltt_selection={'1':ltt}, mean_spi_selection={'1':'raw_1'},
        conformal=dict(infinite_q=infinite, q=None if infinite else 0., calibration_initials=100))
    return evaluate_adapters(data, calibration, family='gate', seeds=(1,), seed_base=100,
                             mode_stream=22117, bootstrap_seed=22123, bootstrap_samples=30)


def test_calibrated_choices_are_not_replaced_by_better_test_outcomes():
    result = evaluate(comparison_data())
    members = result['results']
    assert members['ltt']['per_training_seed']['1']['selected_profiles'] == {'gate_1':2}
    assert members['ltt']['mean_team_gain_seconds'] == 0
    assert members['nominal']['mean_team_gain_seconds'] > members['mean_spi']['mean_team_gain_seconds']
    conformal = members['conformal']['per_training_seed']['1']['actual_random_mode']
    assert conformal['adopted'] == 0 and conformal['conditional_harm'] is None
    with pytest.raises(ValueError):
        evaluate(comparison_data(), ltt='raw_2')


def test_conformal_profile_selection_does_not_read_actual_disturbed_outcomes():
    first = evaluate(comparison_data(), infinite=False)
    changed = evaluate(comparison_data(True), infinite=False)
    selections = lambda result:[r for r in result['selections'] if r['adapter']=='conformal']
    assert selections(first) == selections(changed)
    assert all(r['profile']=='nominal_1' for r in selections(first))
