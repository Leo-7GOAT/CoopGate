"""Check joint-selection hazards rather than a favorable simulated success rate."""
import numpy as np
import pytest
from coopgate.joint_admission_v15 import calibrate_joint


def test_finite_sample_rank_coverage_with_correlated_modes_and_candidates():
    # Enumerate every possible held-out rank; no Monte Carlo luck is involved.
    rng = np.random.default_rng(1515)
    actual = rng.normal(size=(20, 4, 3, 3, 2))
    actual[:, 1:] += actual[:, :1]  # Strong within-initial dependence is allowed.
    predicted = rng.normal(size=(20, 3, 3, 2))
    misses = 0
    for held_out in range(20):
        keep = np.arange(20) != held_out
        bound = calibrate_joint(actual[keep], predicted[keep], ('a', 'b', 'c'), alpha=.1)
        upper = bound.upper(predicted[held_out], ('a', 'b', 'c'))
        misses += bool(np.any(actual[held_out] > upper))
    assert misses == 2


def test_team_gain_cannot_hide_one_harmed_vehicle_or_arrival_metric():
    outcomes = np.zeros((39, 4, 2, 3, 2));predictions = np.zeros((39, 2, 3, 2))
    bound = calibrate_joint(outcomes, predictions, ('cooperative', 'mixed'), alpha=.05)
    forecast = np.array([[[-2., -1.], [-1., -.5], [.1, .1]],
                         [[-.2, -.1], [-.2, -.1], [-.2, -.1]]])
    selected, _ = bound.select(forecast, ('cooperative', 'mixed'))
    assert selected == 1
    forecast[1, 2, 1] = .01
    assert bound.select(forecast, ('cooperative', 'mixed'))[0] is None


def test_disturbance_repetitions_do_not_inflate_calibration_sample_size():
    bound = calibrate_joint(np.zeros((5, 100, 1, 3, 2)), np.zeros((5, 1, 3, 2)), ('a',))
    assert bound.n_initials == 5 and np.isinf(bound.q)
    assert bound.select(np.full((1, 3, 2), -10.), ('a',))[0] is None


def test_changed_profile_identity_and_nonfinite_prediction_cannot_be_admitted():
    bound = calibrate_joint(np.zeros((39, 4, 1, 3, 2)), np.zeros((39, 1, 3, 2)), ('sha_a',))
    with pytest.raises(ValueError, match='family changed'):
        bound.select(np.full((1, 3, 2), -1.), ('sha_b',))
    with pytest.raises(ValueError, match='Nonfinite'):
        bound.select(np.full((1, 3, 2), np.nan), ('sha_a',))
