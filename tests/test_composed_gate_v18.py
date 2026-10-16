import numpy as np
import pytest
import torch
from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.cooperative_v15 import JointPolicyGate
from coopgate.composed_gate_v18 import ComposedProfile, nominal_independent_decision
from coopgate.joint_planner_v15 import JointRolloutGate
from coopgate.fast_v8 import parameters, advance_one
from coopgate.merge_v4 import MergeConfig
from coopgate.robust_v9 import robust_certificate


def test_endpoint_profiles_and_mixed_recovery_share_the_same_physics():
    torch.set_num_threads(1)
    model, _ = load_coupled('artifacts/v17/development_models/coupled_1701/policy.pt')
    initial = np.array([[-77., 13.], [-78., 14.], [-112., 12.]])
    packed = parameters(MergeConfig())
    for mask in ([False]*3, [True]*3, [True, False, True], [False, True, False]):
        controller = ComposedProfile(model, np.array(mask))
        reference = JointPolicyGate(model) if all(mask) else JointRolloutGate()
        state = initial.copy()
        previous = np.zeros(3)
        for step in range(50):
            if np.all(state[:, 0] >= 60):
                break
            command, info = controller.select(state, previous)
            if all(mask) or not any(mask):
                expected, _ = reference.select(state, previous)
                np.testing.assert_array_equal(command, expected)
            assert info['component_learning_mask'] == mask
            assert robust_certificate(state, previous, command, packed, .5)
            assert np.max(np.abs(command-previous)) <= 2.5+1e-9
            assert advance_one(state, command, packed)
            previous = command
        assert np.all(state[:, 0] >= 60)


def test_isolated_decisions_have_explicit_scope_and_validation():
    np.testing.assert_array_equal(nominal_independent_decision([[-.2, 0], [-.1, .5], [0, 0]]), [True, False, False])
    with pytest.raises(ValueError):
        nominal_independent_decision([[np.nan, 0]]*3)
    with pytest.raises(ValueError):
        ComposedProfile(None, [1, 0, 1])
