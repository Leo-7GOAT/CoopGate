import json
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip('numba')
from coopgate.fast_v8 import parameters, advance_one
from coopgate.merge_v4 import MergeConfig
from coopgate.robust_v9 import feedback, robust_certificate
from coopgate.planning_v5 import portfolio
from coopgate.rollout_v10 import active_count, earliest_ticks, tail_ticks, improve, RolloutGate

ROOT = Path(__file__).resolve().parents[1]


def reference_tail(x, prev, k, c, p):
    """Independent, unpruned Python loop with integer running costs."""
    x, prev = x.copy(), prev.copy()
    ticks = 0
    for step in range(k, 50):
        active = int(np.count_nonzero(x[:, 0] < 60))
        if not active: break
        ticks += active
        prev = feedback(x, prev, c, .5, np.array(p['weights']), p['first'], p['coast_steps'], step)
        assert advance_one(x, prev, c)
    return ticks


def test_integer_tail_deadline_and_admissible_bound_against_reference():
    c = parameters(MergeConfig()); p = portfolio()[12]
    rng = np.random.default_rng(101601)
    for step in (0, 5, 20, 48, 49, 50):
        for _ in range(10):
            x = np.column_stack((-rng.uniform(75, 115, 3), rng.uniform(8, 14, 3)))
            x[2, 0] = x[0, 0]-40
            prev = np.zeros(3)
            ref = reference_tail(x, prev, step, c, p)
            ticks, _, stopped = tail_ticks(x, prev, step, c, .5, np.array(p['weights']), p['first'], p['coast_steps'], 10**9, False)
            assert ticks == ref and not stopped
            assert earliest_ticks(x, prev, step, c) <= ref
    x = np.array([[70., 20.], [-90., 8.], [62., 15.]])
    assert active_count(x, c) == 1
    assert earliest_ticks(x, np.zeros(3), 49, c) == 1
    assert tail_ticks(x, np.zeros(3), 50, c, .5, np.ones(3), 0, 0, 10**9, True)[0] == 0


def test_pruning_matches_unpruned_coordinates_on_old_closed_loop_states():
    c = parameters(MergeConfig()); records = []
    for path in sorted((ROOT/'artifacts/v9/study').glob('trace_*robust_learned_pair.json')):
        trace = json.loads(path.read_text())
        for k in range(1, len(trace), 2):
            records.append((np.array(trace[k-1]['x']), np.array(trace[k-1]['command']), k))
    assert len(records) >= 60
    pruned_total, saved_steps = 0, 0
    for index, (x, prev, k) in enumerate(records):
        p = portfolio()[(index*7) % 37]; weights = np.array(p['weights']); order = np.argsort(-weights, kind='stable')
        options = (x, prev, k, c, .5, weights, p['first'], p['coast_steps'], 5, order)
        fast, ref = improve(*options, True), improve(*options, False)
        np.testing.assert_array_equal(fast[0], ref[0])
        assert fast[1:3] == ref[1:3]
        assert fast[2] <= fast[1] and robust_certificate(x, prev, fast[0], c, .5)
        assert fast[3] <= ref[3]
        pruned_total += fast[6]; saved_steps += ref[3]-fast[3]
    assert pruned_total > 0 and saved_steps > 0


def test_nominal_closed_loop_improves_best_base_and_keeps_global_clock():
    c = MergeConfig(); packed = parameters(c)
    initial = json.loads((ROOT/'artifacts/v9/study/initial_states.json').read_text())
    for index, item in enumerate(initial[::8]):
        x = np.array(item['state']); prev = np.zeros(3)
        proposal = (index*11) % 37
        gate = RolloutGate(x, prev, c, .5, 'pair', proposal)
        total = 0
        for k in range(50):
            if not active_count(x, packed): break
            command, info = gate.select(x, prev, None)
            assert info['selected_ticks'] <= info['base_ticks']
            if k == 0: initial_value = info['base_ticks']
            assert info['base_ticks'] <= remaining_bound if k else True
            stage = active_count(x, packed); total += stage
            remaining_bound = info['selected_ticks']-stage
            assert advance_one(x, command, packed)
            prev = command
        assert total <= initial_value
        assert total*c.decision_dt/3 <= gate.predicted_cost+1e-10
        assert gate.step == k or gate.step == 50


def test_rollout_rejects_invalid_settings_and_exhausted_clock():
    x = np.array([[-80., 12.], [-85., 13.], [-120., 11.]])
    for options in ({'levels':1}, {'levels':2.5}, {'passes':0}, {'passes':1.5}):
        with pytest.raises(ValueError): RolloutGate(x, np.zeros(3), **options)
    gate = RolloutGate(x, np.zeros(3)); gate.step = 50
    with pytest.raises(ValueError, match='horizon'): gate.select(x, np.zeros(3), None)
