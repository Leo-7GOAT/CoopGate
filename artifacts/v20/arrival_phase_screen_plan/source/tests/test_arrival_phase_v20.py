import json
from pathlib import Path

import numpy as np
import pytest

from coopgate.arrival_phase_v20 import arrival_phase_suffix, crossing_time
from coopgate.disturbance_v9 import disturbance_tape
from coopgate.disturbed_advantage_v19 import disturbed_action_difference, disturbed_suffix
from coopgate.fast_v8 import parameters
from coopgate.merge_v4 import MergeConfig

ROOT = Path(__file__).resolve().parents[1]


def test_interpolation_detects_order_hidden_by_control_period_quantization():
    earlier = crossing_time(59., 61., 60., 4.2, .1)
    later = crossing_time(59., 60.25, 60., 4.2, .1)
    assert earlier == pytest.approx(4.25)
    assert later == pytest.approx(4.28)
    assert np.ceil(earlier/.5)*.5 == np.ceil(later/.5)*.5
    assert later > earlier
    assert np.isnan(crossing_time(59., 59.5, 60., 4.2, .1))


@pytest.mark.parametrize('mode', ['clean', 'uniform', 'pulse'])
def test_extra_phase_preserves_frozen_baseline_and_candidate_costs(mode):
    study = ROOT / 'artifacts/v20/temporal_repair_extension'
    rows = json.loads((study / 'episodes.json').read_text())
    row = next(r for r in rows if r['learned_action_steps'] > 0)
    trace = json.loads((study / f"shard_{row['initial_index']%2}/trace_{row['initial_index']}_{row['disturbance']}_{row['controller']}.json").read_text())
    step = next(k for k, item in enumerate(trace) if item['portfolio_joint_action_admitted'])
    state = np.asarray(trace[step-1]['x'] if step else row['initial'])
    previous = np.asarray(trace[step-1]['command']) if step else np.zeros(3)
    command = np.asarray(trace[step]['command'])
    c = MergeConfig()
    packed = parameters(c)
    tape = disturbance_tape(314159, mode, .5, 250)
    base, bt, bcomplete = arrival_phase_suffix(state, previous, step, tape, packed, command, False)
    candidate, ct, ccomplete = arrival_phase_suffix(state, previous, step, tape, packed, command, True)
    np.testing.assert_allclose(base, disturbed_suffix(state, previous, step, tape, packed)[0], atol=1e-10, rtol=0)
    np.testing.assert_allclose(candidate-base, disturbed_action_difference(state, previous, step, command, tape, packed)[0], atol=1e-10, rtol=0)
    assert bcomplete and ccomplete
    for values, event in ((base, bt), (candidate, ct)):
        assert np.all(event <= values[:, 1] + 1e-10)
        assert np.all(values[:, 1] - event < c.decision_dt + 1e-10)
        np.testing.assert_allclose(np.ceil((event-1e-10)/c.decision_dt)*c.decision_dt, values[:, 1], atol=1e-10, rtol=0)


def test_incomplete_forecast_keeps_missing_arrival_instead_of_certifying_it():
    c = MergeConfig()
    packed = parameters(c)
    state = np.array([[-100., 0.], [-100., 0.], [-120., 0.]])
    values, arrival, complete = arrival_phase_suffix(state, np.zeros(3), 50, np.zeros((250,3)), packed, np.zeros(3), False)
    assert not complete and np.isnan(arrival).all()
