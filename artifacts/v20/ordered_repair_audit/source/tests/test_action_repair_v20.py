import numpy as np
import pytest

from coopgate.action_repair_v20 import REPAIR_WEIGHTS, attenuated_candidates
from coopgate.fast_v8 import parameters
from coopgate.joint_planner_v15 import joint_rollout
from coopgate.merge_v4 import MergeConfig
from coopgate.portfolio_gate_v19 import joint_compositions
from coopgate.robust_v9 import robust_certificate


def setup_case():
    state = np.array([[-80., 12.], [-85., 13.], [-120., 11.]])
    previous, packed = np.zeros(3), parameters(MergeConfig())
    baseline, _, _, _ = joint_rollout(state, previous, 0, packed)
    learned = joint_compositions(state, previous, np.array([-2.5, 1.875, 0.]), baseline, packed)[0][1]
    return state, previous, learned, baseline, packed


def test_original_command_is_exactly_preserved_and_repairs_are_jointly_physical():
    state, previous, learned, baseline, packed = setup_case()
    candidates = attenuated_candidates(state, previous, learned, baseline, packed)
    assert candidates[0][0] == 1.
    np.testing.assert_array_equal(candidates[0][1], learned)
    assert len(candidates) <= len(REPAIR_WEIGHTS)
    assert len({command.tobytes() for _, command in candidates}) == len(candidates)
    for weight, command in candidates:
        assert weight > 0
        assert robust_certificate(state, previous, command, packed, .5)
        assert not np.array_equal(command, baseline)


def test_baseline_or_unchanged_projection_cannot_count_as_learning_adoption():
    state, previous, _, baseline, packed = setup_case()
    assert attenuated_candidates(state, previous, baseline, baseline, packed) == []


@pytest.mark.parametrize('bad', [[0., 0.], [float('nan'), 0., 0.], [20., 20., 20.]])
def test_malformed_or_uncertified_endpoint_is_rejected(bad):
    state, previous, _, baseline, packed = setup_case()
    with pytest.raises(ValueError):
        attenuated_candidates(state, previous, bad, baseline, packed)
