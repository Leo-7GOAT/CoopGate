import numpy as np

from coopgate.advantage_gate_v18 import baseline_suffix
from coopgate.disturbance_v9 import disturbance_tape
from coopgate.disturbed_advantage_v19 import disturbed_action_difference, disturbed_suffix
from coopgate.fast_v8 import parameters
from coopgate.joint_planner_v15 import joint_rollout
from coopgate.merge_v4 import MergeConfig
from coopgate.stress_prediction_v17 import fixed_stress_bank


def test_disturbed_potential_is_the_same_feedback_baseline_and_has_zero_bellman_residual():
    state = np.array([[-80., 12.], [-85., 13.], [-120., 11.]])
    previous = np.zeros(3)
    packed = parameters(MergeConfig())
    _, tapes = fixed_stress_bank()
    nominal, _, _, _ = baseline_suffix(state, previous, 0, packed)
    reconstructed, _, _ = disturbed_suffix(state, previous, 0, tapes[0], packed)
    np.testing.assert_allclose(nominal, reconstructed, atol=1e-12, rtol=0)
    for clock in (0, 3):
        command, _, _, _ = joint_rollout(state, previous, clock, packed)
        for index in (0, 2, 7, 10, 16):
            difference, _, _ = disturbed_action_difference(state, previous, clock, command, tapes[index], packed)
            np.testing.assert_allclose(difference, 0., atol=1e-12, rtol=0)


def test_bank_covers_constant_bias_but_not_independent_pulse_signs_or_uniform_errors():
    names, tapes = fixed_stress_bank()
    assert len(names) == len(tapes) == 17
    for seed in range(64):
        for mode in ('clean', 'bias'):
            actual = disturbance_tape(seed, mode, .5, 250)
            assert any(np.array_equal(actual, tape) for tape in tapes)
        actual = disturbance_tape(seed, 'uniform', .5, 250)
        assert not any(np.array_equal(actual, tape) for tape in tapes)
    # The actual generator draws NEW signs for each pulse. The fixed-sign bank
    # is not closed under this switching and cannot certify those trajectories.
    actual = disturbance_tape(0, 'pulse', .5, 250)
    assert not any(np.array_equal(actual, tape) for tape in tapes)
