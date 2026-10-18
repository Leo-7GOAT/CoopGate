import numpy as np
import pytest

from coopgate.comparison_v19 import (
    conditional_counts, conformal_episode_bound, conformal_profile, select_fixed_replicates,
)


def test_conformal_physical_failure_cannot_be_hidden_by_improved_costs():
    actual = np.full((40, 4, 2, 3, 2), -10.)
    predicted = np.full((40, 2, 3, 2), -10.)
    failed = np.zeros((40, 4, 2), bool)
    failed[:3, 0, 1] = True
    names = ('raw_1802', 'portfolio0_1802')
    bound, meta = conformal_episode_bound(actual, predicted, failed, names)
    assert meta['infinite_q'] and meta['failed_initial_bundles'] == 3
    assert conformal_profile(bound, predicted[0], names, 1802) == 'baseline'


def test_simultaneous_conformal_selection_does_not_replace_training_replicates():
    names = ('raw_1802', 'portfolio0_1802', 'raw_1812')
    actual = np.zeros((39, 4, 3, 3, 2))
    predictions = np.zeros((39, 3, 3, 2))
    bound, _ = conformal_episode_bound(actual, predictions, np.zeros(actual.shape[:3], bool), names)
    forecast = np.array([np.full((3, 2), -1.), np.full((3, 2), -2.), np.full((3, 2), -9.)])
    assert conformal_profile(bound, forecast, names, 1802) == 'portfolio0_1802'
    assert conformal_profile(bound, forecast, names, 1812) == 'raw_1812'
    forecast[1, 2, 1] = .001
    assert conformal_profile(bound, forecast, names, 1802) == 'raw_1802'
    flags = np.array([True, False, False])
    assert conformal_profile(bound, forecast, names, 1802, flags) == 'baseline'


def test_conformal_rank_coverage_includes_failed_cluster():
    # Exhaustive held-out ranks, including one physical failure. Clustered
    # modes/candidates stay together, so the guarantee has 20 units, not 160.
    rng = np.random.default_rng(19349)
    actual = rng.normal(size=(20, 4, 2, 3, 2))
    predicted = rng.normal(size=(20, 2, 3, 2))
    failed = np.zeros((20, 4, 2), bool)
    failed[3, 1, 0] = True
    misses = 0
    for held in range(20):
        keep = np.arange(20) != held
        bound, _ = conformal_episode_bound(actual[keep], predicted[keep], failed[keep], ('a', 'b'), alpha=.1)
        covered = np.isinf(bound.q) or (not failed[held].any() and
                  np.all(actual[held] <= bound.upper(predicted[held], ('a', 'b'))))
        misses += not covered
    assert misses == 2


def test_selected_profiles_keep_all_seeds_and_reject_unknown_family():
    names = ('raw_1802', 'portfolio0_1802', 'raw_1812')
    means = dict(zip(names, (2., 1., 10.)))
    assert select_fixed_replicates(names, ['portfolio0_1802'], means, [1802, 1812]) == {
        '1802': 'portfolio0_1802', '1812': 'baseline'}
    with pytest.raises(ValueError, match='family mismatch'):
        select_fixed_replicates(names, ['unknown'], means, [1802])


def test_rejection_cannot_dilute_actual_adoption_risk():
    adopted = np.zeros(100, bool)
    adopted[:2] = True
    harmful = np.zeros(100, bool)
    harmful[0] = True
    result = conditional_counts(adopted, harmful)
    assert result['conditional_harm'] == .5
    assert result['adoption_fraction'] == .02
    assert conditional_counts(np.zeros(100, bool), harmful)['conditional_harm'] is None
