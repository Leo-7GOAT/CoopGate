import numpy as np
import pytest
pytest.importorskip('numba')

from coopgate.fast_v8 import certificate, parameters, transition, optimistic_cost, plan_fast, FastPlannedGate
from coopgate.merge_v4 import MergeConfig, admissible, advance, certify, transition_safe
from coopgate.planning_v5 import batch_certify, plan_episode, PlannedGate


def test_compiled_transition_matches_boundary_and_random_reference():
    c = MergeConfig(); packed = parameters(c); rng = np.random.default_rng(81001)
    starts = rng.uniform(-150, 80, (2000, 3)); ends = starts+rng.uniform(0, 2, (2000, 3))
    for boundary in (c.zone_start, c.zone_end, c.separation):
        for epsilon in (-1e-9, 0., 1e-9):
            starts = np.vstack((starts, [boundary+epsilon, boundary, -100.]))
            ends = np.vstack((ends, starts[-1]+[.5, 0., 0.]))
    expected = transition_safe(starts, ends, c)
    np.testing.assert_array_equal([transition(p, q, packed) for p, q in zip(starts, ends)], expected)


def test_compiled_certificates_match_reference_and_reject_slew_violation():
    c = MergeConfig(); packed = parameters(c); rng = np.random.default_rng(81002)
    states = np.zeros((150, 3, 2)); states[:, :, 0] = rng.uniform(-130, 70, (150, 3))
    states[:, :, 1] = rng.uniform(0, 20, (150, 3)); previous = rng.uniform(-5, 2.5, (150, 3))
    proposals = admissible(rng.uniform(-5, 2.5, (150, 3)), previous, c)
    expected = batch_certify(states, previous, proposals, c)
    np.testing.assert_array_equal([certificate(x, p, u, packed) for x, p, u in zip(states, previous, proposals)], expected)
    state = np.array([[-80., 12.], [-85., 13.], [-120., 11.]])
    assert not certificate(state, np.zeros(3), np.full(3, -5.), packed)


def test_full_and_reordered_pruning_match_frozen_costs_and_witness():
    c = MergeConfig(); state = np.array([[-80., 12.], [-85., 13.], [-120., 11.]])
    original = plan_episode(state, np.zeros(3), c)
    for pruning, order in [(False, None), (True, None), (True, [0, *range(36, 0, -1)])]:
        plan, stats = plan_fast(state, np.zeros(3), c, order=order, pruning=pruning)
        assert plan.selected == original.selected and plan.predicted_cost == original.predicted_cost
        np.testing.assert_allclose(plan.commands, original.commands, rtol=0, atol=1e-9)
        np.testing.assert_allclose(plan.states, original.states, rtol=0, atol=1e-8)
        if not pruning:
            np.testing.assert_array_equal(plan.costs, original.costs)
        for step, (x, prev, u) in enumerate(zip(plan.states, plan.previous, plan.commands)):
            assert certify(x, prev, u, c)[0]
            arrived = x[:, 0] >= c.goal
            # An independent optimistic rollout cannot arrive later than the
            # selected actual continuation; completed cars' times are irrelevant.
            finish = np.full(3, c.horizon)
            finish[arrived] = 0.
            lower = optimistic_cost(x, prev, finish, arrived, step, parameters(c))
            actual_finish = np.full(3, c.horizon); actual_finish[arrived] = 0.
            done = arrived.copy()
            for k in range(step, len(plan.commands)):
                future = plan.states[k][None].copy(); advance(future, plan.commands[k][None], c)
                new = (future[0, :, 0] >= c.goal) & ~done
                actual_finish[new] = (k+1)*c.decision_dt; done |= new
            assert lower <= actual_finish.mean()+1e-9


def test_invalid_inputs_and_completed_episode():
    c = MergeConfig()
    with pytest.raises(ValueError, match='Finite'):
        plan_fast(np.full((3, 2), np.nan), np.zeros(3), c)
    x = np.array([[100., 0.], [90., 0.], [80., 0.]])
    plan, _ = plan_fast(x, np.zeros(3), c)
    assert plan.predicted_cost == 0 and len(plan.commands) == 0
    with pytest.raises(ValueError, match='incumbent'):
        plan_fast(x, np.zeros(3), c, order=np.arange(36, -1, -1))


def test_compiled_gate_and_stale_plan_fallback_match_reference():
    c = MergeConfig(); x = np.array([[-80., 12.], [-85., 13.], [-120., 11.]])
    plan, _ = plan_fast(x, np.zeros(3), c, pruning=True)
    for mismatch in (False, True):
        slow, fast = PlannedGate(c, plan), FastPlannedGate(c, plan)
        state = x.copy(); previous = np.zeros(3)
        if mismatch:
            state[:, 0] -= .2
        for _ in range(3):
            a, info = slow.select(state, previous, np.full(3, c.accel))
            b, compiled_info = fast.select(state, previous, np.full(3, c.accel))
            np.testing.assert_array_equal(a, b)
            for key in ('in_set', 'certified', 'intervention', 'plan_fallback'):
                assert compiled_info[key] == info[key]
            future = state[None].copy(); advance(future, b[None], c); state, previous = future[0], b
        assert fast.invalidated == mismatch
