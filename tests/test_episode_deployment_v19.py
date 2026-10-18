import copy
from pathlib import Path
import sys

import numpy as np
import pytest

from coopgate.episode_selection_v19 import select_deployment_profile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from replay_comparison_v19 import coverage_cases, deployment_factory


def setup_records():
    names = ['raw_1802', 'nominal_1802', 'portfolio0_1802', 'raw_1812', 'nominal_1812', 'portfolio0_1812']
    plan = dict(profiles=names, seeds=['1802', '1812'], models={'1802': 'a', '1812': 'b'}, conformal=dict(alpha=.05))
    calibration = dict(names=names, models=plan['models'], calibration_n_initials=100,
                       conformal=dict(q=0., infinite_q=False),
                       ltt_selection={'1802': 'nominal_1802', '1812': 'baseline'},
                       mean_spi_selection={'1802': 'portfolio0_1802', '1812': 'raw_1812'})
    row = dict(per_vehicle_finish=[10., 10., 10.], per_vehicle_variation=[0., 0., 0.],
               completed=True, collision=False, aborted=False, contract_violations=0)
    forecasts = {name: copy.deepcopy(row) for name in ['baseline', *names]}
    forecasts['raw_1802']['per_vehicle_finish'] = [8., 8., 8.]
    forecasts['nominal_1802']['per_vehicle_finish'] = [9., 9., 9.]
    return copy.deepcopy(plan), copy.deepcopy(calibration), forecasts


def test_runtime_conformal_keeps_failure_and_infinite_quantile_fallback():
    plan, calibration, forecasts = setup_records()
    assert select_deployment_profile('conformal', '1802', plan, calibration, forecasts) == 'raw_1802'
    forecasts['raw_1802']['aborted'] = True
    assert select_deployment_profile('conformal', '1802', plan, calibration, forecasts) == 'nominal_1802'
    calibration['conformal'] = dict(q=None, infinite_q=True)
    assert select_deployment_profile('conformal', '1802', plan, calibration, forecasts) == 'baseline'


def test_calibrated_selection_cannot_substitute_another_training_seed():
    plan, calibration, _ = setup_records()
    calibration['ltt_selection']['1802'] = 'raw_1812'
    with pytest.raises(ValueError, match='cross training'):
        select_deployment_profile('ltt', '1802', plan, calibration)


def test_factory_exposes_only_observation_to_forecaster_and_executes_one_profile():
    plan, calibration, rows = setup_records()
    observed, executed, decisions = [], [], []
    class Forecasts:
        def get(self, initial):
            observed.append(np.asarray(initial).copy())
            return dict(index=0, rows=rows), False
    fs = {name: (lambda initial, name=name: executed.append(name) or name) for name in ['baseline', *plan['profiles']]}
    initial = np.array([[-80., 12.], [-85., 13.], [-120., 11.]])
    factory = deployment_factory('conformal', '1802', plan, calibration, fs, Forecasts(), decisions)
    assert factory(initial) == 'raw_1802'
    assert executed == ['raw_1802'] and len(observed) == 1
    np.testing.assert_array_equal(observed[0], initial)
    assert decisions[0]['profile'] == 'raw_1802'
    observed.clear()
    factory = deployment_factory('ltt', '1812', plan, calibration, fs, Forecasts(), decisions)
    assert factory(initial) == 'baseline' and not observed


def test_replay_coverage_keeps_all_failures_even_without_learning_admission():
    selections, lookup = [], {}
    for i in range(4):
        selections.append(dict(initial_index=i, seed=100 + i, adapter='ltt', training_seed='1802', profile='baseline'))
        lookup[i, 'uniform', 'baseline'] = dict(per_vehicle_cost=[10.] * 3, per_vehicle_finish=[10.] * 3,
            completed=i < 2, collision=False, aborted=i >= 2, contract_violations=0, learned_action_steps=0)
    cases = coverage_cases(selections, lookup, ['uniform'], ['ltt'], ['1802'])
    assert [case['initial_index'] for case in cases] == [0, 2, 3]
    assert all('physical_failure_or_noncompletion' in case['reasons'] for case in cases[1:])
