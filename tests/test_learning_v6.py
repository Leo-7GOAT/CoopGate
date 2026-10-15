import numpy as np
import pytest
import torch

from coopgate.merge_v4 import MergeConfig, advance
from coopgate.planning_v5 import plan_episode, portfolio
from coopgate.learning_v6 import ProfileGate, Selector, decide, features, upper_quantile


def test_conformal_rank_and_small_sample_infinity():
    assert upper_quantile(np.arange(9.), .2) == 7.
    assert upper_quantile(np.arange(9.), .1) == 8.
    assert np.isinf(upper_quantile(np.arange(9.), .01))
    with pytest.raises(ValueError):
        upper_quantile([], .1)


def test_all_profile_feedback_costs_match_frozen_teacher():
    c = MergeConfig(); initial = np.array([[-80., 12.], [-85., 13.], [-120., 13.]])
    teacher = plan_episode(initial, np.zeros(3), c)
    # All 37 schedules, including coasting and post-goal lifecycle.
    for index in range(len(portfolio())):
        x = initial.copy(); prev = np.zeros(3); finish = np.full(3, c.horizon); arrived = np.zeros(3, dtype=bool)
        gate = ProfileGate(c, index)
        for step in range(50):
            command, info = gate.select(x, prev, np.zeros(3))
            assert info['certified'] and info['in_set']
            nxt = x[None].copy(); assert advance(nxt, command[None], c)[0]
            x, prev = nxt[0], command
            new = (x[:, 0] >= c.goal) & ~arrived
            finish[new] = (step+1)*c.decision_dt; arrived |= new
            if arrived.all():
                break
        assert finish.mean() == teacher.costs[index]


def test_performance_gate_rejects_without_changing_proposal_estimate():
    torch.set_num_threads(1)
    model = Selector()
    for parameter in model.parameters():
        parameter.data.zero_()
    model.actor.bias.data[4] = 5.
    model.cost.bias.data[4] = -.5
    state = np.array([[-80., 12.], [-85., 13.], [-120., 13.]])
    assert features(state).shape == (12,)
    action, info = decide([model], state, .6)
    assert action == 0 and info['proposal'] == 4 and not info['accepted']
    action, info = decide([model], state, .4)
    assert action == 4 and info['accepted']
