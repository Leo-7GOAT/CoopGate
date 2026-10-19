import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from coopgate.fast_repair_v20 import FastRepairPortfolioGate
from coopgate.repair_portfolio_v20 import RepairPortfolioGate

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'artifacts/v20/repair_development'


def admitted_state(seed):
    rows = json.loads((STUDY / 'episodes.json').read_text())
    row = next(r for r in rows if r['controller'] == f'repair_{seed}' and r['repair_action_steps'] > 0)
    index, mode, name = row['initial_index'], row['disturbance'], row['controller']
    trace = json.loads((STUDY / f'shard_{index % 2}/trace_{index}_{mode}_{name}.json').read_text())
    step = next(i for i, item in enumerate(trace) if item['repair_action_admitted'])
    state = np.asarray(trace[step - 1]['x'] if step else row['initial'])
    previous = np.asarray(trace[step - 1]['command'] if step else np.zeros(3))
    return state, previous, step, trace[step]


def configured(cls, step, item, **kwargs):
    gate = cls(None, **kwargs)
    gate.step = step
    gate.primary = SimpleNamespace(step=step, select=lambda *_: (
        np.asarray(item['baseline_command']), {'proposal': item['actor_proposal']}))
    return gate


@pytest.mark.parametrize('seed', [1802, 1812, 1822])
def test_actual_repaired_admission_and_full_bank_survive_acceleration(seed):
    state, previous, step, item = admitted_state(seed)
    reference = configured(RepairPortfolioGate, step, item)
    expected, info = reference.select(state, previous)
    assert info['repair_action_admitted']
    for bounded in (False, True):
        fast = configured(FastRepairPortfolioGate, step, item, bounded=bounded)
        command, actual = fast.select(state, previous)
        np.testing.assert_array_equal(command, expected)
        for key in ('composition', 'repair_weight', 'candidate_count', 'original_candidate_count',
                    'nominal_candidate_count', 'selected_bank_differences', 'forecast_errors', 'generation_errors'):
            assert actual[key] == info[key]
        assert len(actual['selected_bank_differences']) == 33
        assert actual['performance_model_transitions'] <= info['performance_model_transitions']
        assert actual['exact_cache_entries'] > 0
        assert not actual['candidate_evidence'][-1]['lower_bound_scenarios']


@pytest.mark.parametrize('kind', ['exception', 'nonfinite'])
def test_unavailable_forecasts_never_authorize_cached_learning(monkeypatch, kind):
    state, previous, step, item = admitted_state(1802)
    gate = configured(FastRepairPortfolioGate, step, item)
    def broken(*args):
        if kind == 'exception':
            raise RuntimeError('unavailable counterfactual')
        return np.full((3, 2), np.nan), 0, 0, False, 0
    monkeypatch.setattr(gate, '_future', broken)
    command, info = gate.select(state, previous)
    assert info['forecast_errors']
    assert not info['portfolio_joint_action_admitted'] and not info['work_counts_complete']
    np.testing.assert_array_equal(command, item['baseline_command'])


def test_optimization_mode_must_be_explicit_boolean():
    with pytest.raises(ValueError, match='boolean'):
        FastRepairPortfolioGate(None, bounded=1)
