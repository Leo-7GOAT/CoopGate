import numpy as np

from coopgate.advantage_gate_v18 import action_then_baseline, baseline_suffix
from coopgate.bounded_portfolio_v19 import bounded_action_then_baseline, optimistic_arrival_lower_bound
from coopgate.cached_portfolio_v19 import new_command_cache
from coopgate.disturbed_advantage_v19 import disturbed_action_difference, disturbed_suffix
from coopgate.fast_v8 import parameters
from coopgate.merge_v4 import MergeConfig
from coopgate.stress_prediction_v17 import fixed_stress_bank


def test_optimistic_arrival_bounds_the_independent_reference_feedback_tail():
    _, tapes = fixed_stress_bank()
    packed = parameters(MergeConfig())
    previous = np.zeros(3)
    for state in (np.array([[-80., 12.], [-85., 13.], [-120., 11.]]),
                  np.array([[-60., 4.], [-65., 5.], [-85., 4.]])):
        for index in (0, 3, 12):
            lower, _ = optimistic_arrival_lower_bound(state, previous, 0, tapes[index], packed, index == 0)
            full = (baseline_suffix(state, previous, 0, packed)[0] if index == 0 else
                    disturbed_suffix(state, previous, 0, tapes[index], packed)[0])
            assert np.all(lower <= full[:, 1] + 1e-9)


def test_pruned_values_are_true_lower_bounds_and_never_reject_the_baseline_command():
    _, tapes = fixed_stress_bank()
    packed = parameters(MergeConfig())
    state = np.array([[-80., 12.], [-85., 13.], [-120., 11.]])
    previous = np.zeros(3)
    baseline, baseline_command, _, _ = baseline_suffix(state, previous, 0, packed)
    pruned_count = 0
    for index in (0, 3, 12):
        base = baseline if index == 0 else disturbed_suffix(state, previous, 0, tapes[index], packed)[0]
        for command in (baseline_command, np.full(3, -2.5)):
            value, _, _, pruned, _ = bounded_action_then_baseline(
                state, previous, 0, command, tapes[index], packed, base, new_command_cache(), index == 0)
            full = (action_then_baseline(state, previous, 0, command, packed)[0] if index == 0 else
                    disturbed_action_difference(state, previous, 0, command, tapes[index], packed)[0] + base)
            if pruned:
                assert np.all(value <= full + 1e-9)
                assert np.any(full > base + 1e-10)
                pruned_count += 1
            else:
                np.testing.assert_array_equal(value, full)
            if np.array_equal(command, baseline_command):
                assert not pruned
    assert pruned_count > 0
