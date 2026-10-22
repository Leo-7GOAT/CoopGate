import ast
import inspect
from pathlib import Path
import sys
import textwrap

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_fast_repair_v20 import admitted_state, configured
from coopgate.cached_portfolio_v19 import new_command_cache
from coopgate.fast_v8 import advance_one
from coopgate.persistent_phase_v24 import PersistentPhaseRepairGate, retain_future_commands
from coopgate.phase_repair_v21 import PhaseRepairGate


def semantic(info):
    return {k: v for k, v in info.items()
            if k not in ('gate_ms', 'performance_model_transitions', 'exact_cache_entries')
            and not k.startswith('persistent_cache_')}


def test_selection_algorithm_changes_only_cache_source():
    original = textwrap.dedent(inspect.getsource(PhaseRepairGate.select))
    expected = original.replace('def select(', 'def _select_cached(', 1).replace(
        'cache = new_command_cache()', 'cache = self._command_cache', 1)
    actual = textwrap.dedent(inspect.getsource(PersistentPhaseRepairGate._select_cached))
    assert ast.dump(ast.parse(actual)) == ast.dump(ast.parse(expected))


def key(step, tag=0.):
    return (float(step), float(tag), 2., 3., 4., 5., 6., 7., 8., 9.)


def test_eviction_drops_past_and_retains_nearest_future_with_exact_values():
    cache = new_command_cache()
    for step in range(8):
        cache[key(step)] = (float(step), 1., 2.)
    assert retain_future_commands(cache, 3, 2) == 6
    assert set(cache.keys()) == {key(3), key(4)}
    assert cache[key(4)] == (4., 1., 2.)


def test_configuration_changes_and_clock_rewind_clear_stored_commands():
    gate = PersistentPhaseRepairGate(None)
    assert gate._prepare_cache() == 'initial_or_changed_configuration'
    gate._command_cache[key(4)] = (1., 2., 3.)
    gate.step, gate._last_step = 2, 1
    assert gate._prepare_cache() == 'reuse'
    assert len(gate._command_cache) == 1
    gate.packed[3] = np.nextafter(gate.packed[3], np.inf)
    assert gate._prepare_cache() == 'initial_or_changed_configuration'
    assert len(gate._command_cache) == 0
    gate._command_cache[key(4)] = (1., 2., 3.)
    gate.step = 1
    assert gate._prepare_cache() == 'nonincreasing_clock'
    assert len(gate._command_cache) == 0


def test_invalid_state_clears_partial_cache_and_propagates_error():
    gate = PersistentPhaseRepairGate(None)
    gate._prepare_cache()
    gate._command_cache[key(4)] = (1., 2., 3.)
    with pytest.raises((ValueError, IndexError)):
        gate.select(np.zeros((1, 1)), np.zeros(3))
    assert len(gate._command_cache) == 0
    assert gate._cache_configuration is None


@pytest.mark.parametrize('limit', [0, -1, True, 1.5, np.nan])
def test_invalid_retained_limits_are_rejected(limit):
    with pytest.raises(ValueError, match='positive integer'):
        PersistentPhaseRepairGate(None, max_cache_entries=limit)


@pytest.mark.parametrize('seed', [1802, 1812, 1822])
@pytest.mark.parametrize('limit', [1, 8192])
def test_full_consecutive_decisions_match_with_aggressive_and_normal_eviction(seed, limit):
    state, previous, step, item = admitted_state(seed)
    reference = configured(PhaseRepairGate, step, item)
    cached = configured(PersistentPhaseRepairGate, step, item, max_cache_entries=limit)
    for _ in range(2):
        expected, original = reference.select(state, previous)
        actual, checked = cached.select(state, previous)
        np.testing.assert_array_equal(actual, expected)
        assert semantic(checked) == semantic(original)
        assert checked['performance_model_transitions'] <= original['performance_model_transitions']
        assert len(cached._command_cache) <= limit
        following = state.copy()
        assert advance_one(following, actual, cached.packed)
        state, previous = following, actual


def test_disabling_reuse_matches_original_work_and_discards_cache():
    state, previous, step, item = admitted_state(1802)
    reference = configured(PhaseRepairGate, step, item)
    cached = configured(PersistentPhaseRepairGate, step, item, reuse=False)
    expected, original = reference.select(state, previous)
    actual, checked = cached.select(state, previous)
    np.testing.assert_array_equal(actual, expected)
    assert semantic(checked) == semantic(original)
    assert checked['performance_model_transitions'] == original['performance_model_transitions']
    assert len(cached._command_cache) == 0
