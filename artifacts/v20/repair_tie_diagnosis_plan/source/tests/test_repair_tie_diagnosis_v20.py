from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from diagnose_repair_ties_v20 import deviation_key, exact_gain_ties


def record(command, difference, weight=1., composition=1):
    return dict(command=command, nominal_difference=difference, weight=weight, composition=composition)


def test_equal_team_gain_does_not_override_individual_constraints():
    selected = [[-.5, 0.], [0., 0.], [0., 0.]]
    violates_vehicle = [[-.75, 0.], [.25, 0.], [0., 0.]]
    violates_time = [[-.5, .5], [0., 0.], [0., 0.]]
    records = [record([1, 0, 0], d) for d in (selected, violates_vehicle, violates_time)]
    assert exact_gain_ties(records, selected) == records[:1]


def test_nearly_equal_nominal_gain_is_not_an_exact_tie():
    selected = [[-.5, 0.], [0., 0.], [0., 0.]]
    changed = [[-.5 - 1e-8, 0.], [0., 0.], [0., 0.]]
    records = [record([1, 0, 0], d) for d in (selected, changed)]
    assert exact_gain_ties(records, selected) == records[:1]


def test_distance_uses_executed_joint_command_not_raw_attenuation_weight():
    baseline = np.array([1., 1., 1.])
    low_weight = record([3., 1., 1.], None, .25)
    high_weight = record([1.5, 1., 1.], None, 1.)
    assert deviation_key(high_weight, baseline) < deviation_key(low_weight, baseline)
    same_linf_larger_l1 = record([1.5, 1.5, 1.], None, .25)
    assert deviation_key(high_weight, baseline) < deviation_key(same_linf_larger_l1, baseline)


def test_nonfinite_forecast_cannot_be_a_tie():
    with pytest.raises(ValueError):
        exact_gain_ties([record([0, 0, 0], [[np.nan, 0], [0, 0], [0, 0]])], [[-.5, 0], [0, 0], [0, 0]])
