import numpy as np
import pytest

from coopgate.fast_v8 import parameters
from coopgate.joint_planner_v15 import joint_rollout
from coopgate.merge_v4 import MergeConfig
from coopgate.portfolio_gate_v19 import JointPortfolioGate, joint_compositions
from coopgate.robust_v9 import robust_certificate


def test_compositions_reproject_the_complete_joint_command_and_deduplicate():
    state = np.array([[-80., 12.], [-85., 13.], [-120., 11.]])
    previous = np.zeros(3)
    packed = parameters(MergeConfig())
    base, _, _, _ = joint_rollout(state, previous, 0, packed)
    candidates = joint_compositions(state, previous, np.array([-2.5, 1.875, 0.]), base, packed)
    assert candidates
    for code, command, _ in candidates:
        assert 1 <= code <= 7
        assert not np.array_equal(command, base)
        assert robust_certificate(state, previous, command, packed, .5)
    assert len({tuple(row[1]) for row in candidates}) == len(candidates)
    assert joint_compositions(state, previous, base, base, packed) == []


@pytest.mark.parametrize('budget', [-1, 17, .5, True])
def test_invalid_failure_budget_is_not_silently_relaxed(budget):
    with pytest.raises(ValueError):
        JointPortfolioGate(None, budget)
