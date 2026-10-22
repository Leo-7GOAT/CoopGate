from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from coopgate.prospective_validation_v21 import MODELS, profile_names
from observable_profile_prediction_v21 import choose_profiles, prediction_from_rows, verify_clean_identity


def fixture():
    family = 'repair'
    names = profile_names(family)
    state = [[-50., 10.], [-55., 11.], [-80., 12.]]
    rows = {}
    for name in ('baseline', *names):
        finish = [10. if name == 'baseline' else 9.]*3
        rows[name] = dict(controller=name, initial=state, seed=1, scenario='unused', disturbance='clean',
                          per_vehicle_finish=finish, per_vehicle_variation=[0.]*3, per_vehicle_cost=finish,
                          completed=True, collision=False, aborted=False, contract_violations=0)
    calibration = dict(primary_family=family, common_profiles=list(names),
        conformal=dict(infinite_q=False, q=0., calibration_initials=4096),
        ltt_selection={str(s):'repair_'+str(s) for s in MODELS},
        mean_spi_selection={str(s):'nominal_'+str(s) for s in MODELS})
    return state, rows, calibration


def test_reference_changes_cannot_change_observed_only_selections():
    state, rows, cal = fixture()
    prediction = prediction_from_rows(state, 'repair', rows)
    choices = choose_profiles(prediction, cal, 'repair')
    clean = deepcopy(rows)
    verify_clean_identity(prediction, clean)
    clean['repair_1802']['per_vehicle_finish'] = [20.]*3
    clean['repair_1802']['per_vehicle_cost'] = [20.]*3
    with pytest.raises(AssertionError):
        verify_clean_identity(prediction, clean)
    assert choose_profiles(prediction, cal, 'repair') == choices


def test_failed_shadow_and_failed_baseline_cannot_authorize_conformal_use():
    state, rows, cal = fixture()
    for name in profile_names('repair'):
        rows[name]['completed'] = False
    p = prediction_from_rows(state, 'repair', rows)
    assert set(choose_profiles(p, cal, 'repair')['conformal'].values()) == {'baseline'}
    state, rows, cal = fixture()
    rows['baseline']['completed'] = False
    assert set(choose_profiles(prediction_from_rows(state, 'repair', rows), cal, 'repair')['conformal'].values()) == {'baseline'}


@pytest.mark.parametrize('field,value', [('seed', 63500001), ('scenario', 'dense'), ('disturbance', 'uniform')])
def test_actual_episode_identifiers_cannot_enter_the_shadow_interface(field, value):
    state, rows, _ = fixture()
    rows['baseline'][field] = value
    with pytest.raises(ValueError):
        prediction_from_rows(state, 'repair', rows)


def test_calibration_cannot_cross_training_models_or_families():
    state, rows, cal = fixture()
    p = prediction_from_rows(state, 'repair', rows)
    cal['ltt_selection']['1802'] = 'repair_1812'
    with pytest.raises(ValueError):
        choose_profiles(p, cal, 'repair')
    _, _, cal = fixture()
    cal['primary_family'] = 'phase05'
    with pytest.raises(ValueError):
        choose_profiles(p, cal, 'repair')


def test_infinite_conformal_bound_keeps_zero_adoption_as_fallback():
    state, rows, cal = fixture()
    cal['conformal']['infinite_q'] = True
    choices = choose_profiles(prediction_from_rows(state, 'repair', rows), cal, 'repair')
    assert set(choices['conformal'].values()) == {'baseline'}
    assert all(choices['ltt'][str(s)] == 'repair_'+str(s) for s in MODELS)
