import copy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
specification = importlib.util.spec_from_file_location('screen_action_repair_v20', TOOLS / 'screen_action_repair_v20.py')
screen = importlib.util.module_from_spec(specification)
specification.loader.exec_module(screen)


def prediction(weight=1., gain=.1):
    differences = np.zeros((3, 3, 2))
    differences[:, :, 0] = -gain
    return dict(weight=weight, differences=differences.tolist(), forecast_errors=[None] * 3)


def test_improved_team_cost_does_not_hide_a_harmed_other_vehicle_or_time_component():
    for component in (0, 1):
        bad = prediction(gain=1.)
        bad['differences'][1][2][component] = .01
        good = prediction(weight=.5, gain=.1)
        assert screen.choose([bad, good], {'bank': [0, 1, 2]}) == {'bank': 1}


def test_choice_uses_nominal_gain_and_larger_weight_for_exact_ties():
    candidates = [prediction(.25, .1), prediction(.5, .1), prediction(1., .05)]
    assert screen.choose(candidates, {'bank': [0, 1, 2]}) == {'bank': 1}
    candidates[2] = prediction(1., .2)
    assert screen.choose(candidates, {'bank': [0, 1, 2]}) == {'bank': 2}


def test_forecast_exception_rejects_only_banks_that_require_that_path():
    candidate = prediction()
    candidate['forecast_errors'][2] = 'outside contract'
    candidate['differences'][2] = None
    assert screen.choose([candidate], {'short': [0, 1], 'full': [0, 1, 2]}) == {'short': 0, 'full': None}


def test_no_nominal_gain_falls_back_and_silent_nonfinite_value_is_an_error():
    candidate = prediction(gain=0.)
    assert screen.choose([candidate], {'bank': [0, 1, 2]}) == {'bank': None}
    candidate['differences'][1][0][0] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite forecast'):
        screen.choose([candidate], {'bank': [0, 1, 2]})


def test_selection_does_not_consult_post_hoc_labels_or_modify_predictions():
    candidates = [prediction(1., .1), prediction(.5, .2)]
    original = copy.deepcopy(candidates)
    assert screen.choose(candidates, {'bank': [0, 1, 2]}) == {'bank': 1}
    assert candidates == original
    candidates[0]['hindsight_positive'] = False
    candidates[1]['hindsight_positive'] = True
    assert screen.choose(candidates, {'bank': [0, 1, 2]}) == {'bank': 1}
