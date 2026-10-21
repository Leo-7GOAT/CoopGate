from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from recover_interrupted_v21 import (
    assert_no_existing_workers, compare_replay, prefix_layout, replay_cases,
)


CASES = [(0, 'clean', 'baseline'), (0, 'clean', 'raw'), (0, 'bias', 'baseline')]


def filename(case):
    return '_'.join(map(str, case))


def test_completed_prefix_and_single_boundary_orphan_are_preserved():
    missing, orphan = prefix_layout(CASES, CASES[:1], list(map(filename, CASES[:2])), filename)
    assert missing == CASES[1:]
    assert orphan == [filename(CASES[1])]


@pytest.mark.parametrize('observed,files', [
    (CASES[1:2], [filename(CASES[1])]),
    (CASES[:1] * 2, [filename(CASES[0])]),
    (CASES[:1], []),
    (CASES[:1], [filename(CASES[0]), filename(CASES[2])]),
    (CASES, list(map(filename, CASES)) + ['unexpected']),
])
def test_recovery_rejects_holes_duplicates_missing_traces_and_unscheduled_data(observed, files):
    with pytest.raises(ValueError):
        prefix_layout(CASES, observed, files, filename)


def test_replay_requires_outcomes_actions_and_errors_to_match_except_actual_timing():
    old = dict(completed=False, aborted=True, team_cost=10., controller_ms=[3.], setup_ms=1.)
    new = dict(old, controller_ms=[8.], setup_ms=2.)
    trace = [dict(command=[1., 0., 0.], forecast_errors=['retained'], gate_ms=3.)]
    other = [dict(trace[0], gate_ms=8.)]
    compare_replay(old, new, trace, other)
    for changed in (dict(new, completed=True), dict(new, team_cost=9.)):
        with pytest.raises(AssertionError):
            compare_replay(old, changed, trace, other)
    with pytest.raises(AssertionError):
        compare_replay(old, new, trace, [dict(other[0], forecast_errors=[])])
    with pytest.raises(AssertionError):
        compare_replay(None, new, trace, [dict(other[0], command=[0., 0., 0.])])


def test_running_process_is_not_treated_as_interruption():
    own = dict(ProcessId=20, ParentProcessId=10)
    parent = dict(ProcessId=10, ParentProcessId=1)
    assert_no_existing_workers([own, parent], 20, 10)
    with pytest.raises(ValueError):
        assert_no_existing_workers([own, parent, dict(ProcessId=99, ParentProcessId=1)], 20, 10)


def test_outcome_blind_replays_cover_controllers_and_shard_boundaries():
    prefixes = {
        0: [dict(initial_index=0, disturbance='clean', controller='baseline', completed=False),
            dict(initial_index=0, disturbance='clean', controller='raw', completed=True),
            dict(initial_index=3, disturbance='bias', controller='raw', completed=False)],
        1: [dict(initial_index=1, disturbance='clean', controller='baseline', completed=True)],
    }
    chosen = replay_cases('phase', prefixes, dict(methods={'baseline': {}, 'raw': {}}))
    assert len(chosen) == 4
    for rows in prefixes.values():
        for row in rows:
            row['completed'] = not row['completed']
    assert replay_cases('phase', prefixes, dict(methods={'baseline': {}, 'raw': {}})) == chosen


def test_complete_prefix_needs_no_resimulation():
    assert prefix_layout(CASES, CASES, list(map(filename, CASES)), filename) == ([], [])
