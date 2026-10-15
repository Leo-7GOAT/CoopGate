import numpy as np
import pytest

from coopgate.merge_v4 import MergeConfig, MergeEnv, MergeGate, admissible, advance, backup, certify
from coopgate.planning_v5 import PlannedGate, batch_certify, plan_episode, portfolio


def test_batch_certificate_matches_frozen_scalar_reference():
    c = MergeConfig()
    rng = np.random.default_rng(881)
    x = np.array([[-80., 12.], [-85., 13.], [-120., 13.]])
    previous = np.zeros(3)
    states, memories, commands = [], [], []
    gate = MergeGate(c, 'joint_rl')
    for _ in range(25):
        for proposal in admissible(rng.uniform(-5., 2.5, (4, 3)), previous, c):
            states.append(x.copy()); memories.append(previous.copy()); commands.append(proposal)
        command, _ = gate.select(x, previous, rng.uniform(-5., 2.5, 3))
        future = x[None].copy(); advance(future, command[None], c)
        x, previous = future[0], command
    result = batch_certify(np.array(states), np.array(memories), np.array(commands), c)
    reference = [certify(s, p, u, c)[0] for s, p, u in zip(states, memories, commands)]
    np.testing.assert_array_equal(result, reference)
    # Geometric feasibility alone cannot authorize an actuator-infeasible command.
    assert not batch_certify(np.array(states[:1]), np.zeros((1, 3)), np.full((1, 3), -5.), c)[0]


def test_incumbent_is_preserved_and_every_saved_action_has_a_witness():
    c = MergeConfig()
    x = np.array([[-80., 12.], [-85., 13.], [-120., 13.]])
    plan = plan_episode(x, np.zeros(3), c)
    assert len(plan.costs) == 37
    assert plan.predicted_cost <= plan.costs[0]
    for state, prev, command in zip(plan.states, plan.previous, plan.commands):
        assert batch_certify(state[None], prev[None], command[None], c)[0]
    with pytest.raises(ValueError, match='incumbent'):
        plan_episode(x, np.zeros(3), c, profiles=portfolio()[1:])


def test_stale_plan_is_invalidated_and_feedback_fallback_is_certified():
    c = MergeConfig()
    x = np.array([[-80., 12.], [-85., 13.], [-120., 13.]])
    plan = plan_episode(x, np.zeros(3), c)
    gate = PlannedGate(c, plan)
    shifted = x.copy(); shifted[:, 0] -= .2  # stays within K_B
    command, info = gate.select(shifted, np.zeros(3), np.full(3, 2.5))
    assert gate.invalidated and info['plan_fallback'] and info['certified']
    expected, _ = MergeGate(c, 'constant').select(shifted, np.zeros(3), np.full(3, 2.5))
    np.testing.assert_array_equal(command, expected)
    assert certify(shifted, np.zeros(3), backup(np.zeros(3), c), c)[0]


def test_real_sumo_cost_and_baseline_equal_model_rollout():
    with MergeEnv(method='constant') as env:
        env.reset(503, 'dense')
        plan = plan_episode(env.x, env.prev, env.c)
        while True:
            _, _, t, u, baseline = env.step([4]*3)
            if t or u:
                break
        assert baseline['mean_capped_finish_s'] == plan.costs[0]
        env.reset(503, 'dense'); env.gate = PlannedGate(env.c, plan)
        while True:
            _, _, t, u, result = env.step([4]*3)
            if t or u:
                break
        assert result['completed'] and not result['collision']
        assert result['mean_capped_finish_s'] == plan.predicted_cost <= baseline['mean_capped_finish_s']
        assert result['contract_violations'] == result['uncertified_steps'] == result['outside_steps'] == 0
        assert result['max_prediction_error'] < 1e-6 and env.gate.fallback_count == 0
