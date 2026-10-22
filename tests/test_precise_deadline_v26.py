import ast
from functools import partial
import inspect
import textwrap
import time

import numpy as np
import pytest

from coopgate.deadline_profile_v26 import DeadlineProfile
from coopgate.fast_v8 import parameters
from coopgate.merge_v4 import MergeConfig
from coopgate.precise_deadline_v26 import PreciseDeadlineProfile
from coopgate.robust_v9 import robust_certificate
from test_deadline_profile_v26 import Constant, STATE


def test_select_logic_differs_only_in_eight_clock_reads():
    old = ast.parse(textwrap.dedent(inspect.getsource(DeadlineProfile.select)))
    new = ast.parse(textwrap.dedent(inspect.getsource(PreciseDeadlineProfile.select)))
    replacements = 0
    for node in ast.walk(old):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "time" and node.attr == "monotonic":
                node.attr = "perf_counter"
                replacements += 1
    assert replacements == 8
    assert ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False)


@pytest.fixture(autouse=True)
def warm_certificate():
    assert robust_certificate(STATE, np.zeros(3), np.zeros(3), parameters(MergeConfig()), .5)


@pytest.mark.parametrize("mode", ["hang", "slow"])
def test_real_timeout_does_not_fire_before_the_high_resolution_budget(mode):
    profile = PreciseDeadlineProfile(partial(Constant, -2.5, mode), Constant(), budget_seconds=.045).prepare()
    try:
        started = time.perf_counter()
        command, info = profile.select(STATE, np.zeros(3))
        elapsed = time.perf_counter() - started
        np.testing.assert_array_equal(command, np.zeros(3))
        assert .045 <= elapsed < .8  # An observed call, not a universal latency bound.
        assert .045 <= info["gate_ms"] / 1000 <= elapsed
        assert info["deadline_triggered_this_step"] and info["performance_admitted"]
        assert not info["deadline_primary_used"]
        if mode == "slow":
            profile.pending.result(timeout=5.)
        following, record = profile.select(STATE, np.zeros(3))
        np.testing.assert_array_equal(following, command)
        assert record["fault_fallback"] and not record["deadline_primary_used"]
    finally:
        assert not profile.close()["worker_alive"]


def test_valid_proposal_and_preparation_timing():
    profile = PreciseDeadlineProfile(partial(Constant, -2.5), Constant(), budget_seconds=1.)
    started = time.perf_counter()
    profile.prepare()
    setup = time.perf_counter() - started
    try:
        assert 0 < profile.setup_seconds <= setup
        command, info = profile.select(STATE, np.zeros(3))
        np.testing.assert_array_equal(command, np.full(3, -2.5))
        assert info["deadline_primary_used"] and not info["fault_fallback"]
        assert info["deadline_setup_ms"] == 1000 * profile.setup_seconds
    finally:
        profile.close()


def test_baseline_exhaustion_still_prevents_dispatch():
    profile = PreciseDeadlineProfile(partial(Constant, -2.5), Constant(mode="slow"), budget_seconds=.045).prepare()
    try:
        command, info = profile.select(STATE, np.zeros(3))
        np.testing.assert_array_equal(command, np.zeros(3))
        assert info["fault_reason"] == "baseline_preparation_exhausted_budget"
        assert profile.pending is None and not info["deadline_primary_used"]
    finally:
        profile.close()


def test_parent_validation_overrun_is_not_accepted(monkeypatch):
    profile = PreciseDeadlineProfile(partial(Constant, -2.5), Constant(), budget_seconds=.045).prepare()
    checked = profile._checked
    def slow_check(state, previous, command, info):
        result = checked(state, previous, command, info)
        if np.all(np.asarray(command) == -2.5):
            time.sleep(.1)
        return result
    monkeypatch.setattr(profile, "_checked", slow_check)
    try:
        command, info = profile.select(STATE, np.zeros(3))
        np.testing.assert_array_equal(command, np.zeros(3))
        assert info["deadline_observed_overrun"] and not info["deadline_primary_used"]
    finally:
        profile.close()
