from functools import partial
import os
import time

import numpy as np
import pytest

from coopgate.deadline_profile_v26 import DeadlineProfile, MAX_REPLY_BYTES
from coopgate.fast_v8 import parameters
from coopgate.merge_v4 import MergeConfig
from coopgate.robust_v9 import robust_certificate


STATE = np.array([[-80., 12.], [-85., 13.], [-120., 11.]])


class Constant:
    def __init__(self, value=0., mode="normal"):
        self.value, self.mode, self.calls, self.step = value, mode, 0, 0

    def select(self, x, previous):
        self.calls += 1
        if self.mode == "exit":
            os._exit(3)
        if self.mode == "hang":
            while True:
                time.sleep(.1)
        if self.mode == "slow":
            time.sleep(1.)
        if self.mode == "mutate_raise":
            x[:] = 1e9
            previous[:] = 1e9
            raise RuntimeError("Injected mutation followed by error")
        if self.mode == "raise":
            raise RuntimeError("Injected failure")
        if self.mode == "nan":
            return np.full(3, np.nan), {}
        if self.mode == "shape":
            return np.zeros(4), {}
        if self.mode == "slew":
            return np.full(3, -5.), {}
        if self.mode == "limit":
            return np.full(3, 100.), {}
        info = {"intervention": 0., "marker": "primary" if self.value else "baseline"}
        if self.mode == "oversize":
            info["oversize"] = "x" * (MAX_REPLY_BYTES + 1)
        return np.full(3, self.value), info


def startup_failure():
    raise RuntimeError("Injected constructor failure")


@pytest.fixture(autouse=True)
def warm_certificate():
    assert robust_certificate(STATE, np.zeros(3), np.zeros(3), parameters(MergeConfig()), .5)


@pytest.mark.parametrize("mode", ["hang", "slow"])
def test_timeout_returns_baseline_without_waiting_for_worker_or_accepting_late_result(mode):
    baseline = Constant()
    profile = DeadlineProfile(partial(Constant, -2.5, mode), baseline, budget_seconds=.04).prepare()
    worker = profile.process
    try:
        started = time.monotonic()
        command, info = profile.select(STATE, np.zeros(3))
        elapsed = time.monotonic() - started
        np.testing.assert_array_equal(command, np.zeros(3))
        assert elapsed < .8  # Distinguish failover from the injected 1 s sleep; not a hard deadline claim.
        assert worker.is_alive()
        assert info["fault_fallback"] and info["deadline_triggered_this_step"]
        assert info["performance_admitted"] and not info["deadline_primary_used"]
        assert info["fault_reason"] == "proposal_deadline"
        if mode == "slow":
            profile.pending.result(timeout=5.)  # Allow a real late valid reply to arrive.
        following, record = profile.select(STATE, np.zeros(3))
        np.testing.assert_array_equal(following, command)
        assert not record["deadline_primary_used"] and baseline.calls == 2
    finally:
        shutdown = profile.close()
        assert not worker.is_alive() and not shutdown["worker_alive"]


@pytest.mark.parametrize("mode", ["raise", "mutate_raise", "nan", "shape", "slew", "limit", "oversize", "exit"])
def test_invalid_proposals_preserve_inputs_and_initial_admission(mode):
    profile = DeadlineProfile(partial(Constant, -2.5, mode), Constant(), budget_seconds=1.).prepare()
    state, previous = STATE.copy(), np.zeros(3)
    try:
        command, info = profile.select(state, previous)
        np.testing.assert_array_equal(state, STATE)
        np.testing.assert_array_equal(previous, np.zeros(3))
        np.testing.assert_array_equal(command, np.zeros(3))
        assert info["fault_fallback"] and info["performance_admitted"]
        assert not info["deadline_primary_used"]
    finally:
        profile.close()


def test_valid_primary_and_live_step_are_used_before_deadline():
    profile = DeadlineProfile(partial(Constant, -2.5), Constant(), budget_seconds=1.).prepare()
    try:
        for step in range(3):
            command, info = profile.select(STATE, np.zeros(3))
            np.testing.assert_array_equal(command, np.full(3, -2.5))
            assert profile.step == step + 1 and info["deadline_primary_used"]
            assert info["deadline_primary_used_ever"] and not info["fault_fallback"]
    finally:
        profile.close()


def test_initial_rejection_does_not_spawn_or_call_primary():
    profile = DeadlineProfile(startup_failure, Constant(), admitted=False).prepare()
    try:
        command, info = profile.select(STATE, np.zeros(3))
        np.testing.assert_array_equal(command, np.zeros(3))
        assert profile.process is None and not info["performance_admitted"]
        assert not info["fault_fallback"]
    finally:
        profile.close()


def test_failed_baseline_propagates_without_unsafe_primary_fallback():
    profile = DeadlineProfile(None, Constant(mode="raise"), admitted=False).prepare()
    try:
        with pytest.raises(RuntimeError, match="Injected failure"):
            profile.select(STATE, np.zeros(3))
        assert profile.step == 0
    finally:
        profile.close()


def test_baseline_preparation_budget_is_included():
    profile = DeadlineProfile(partial(Constant, -2.5), Constant(mode="slow"), budget_seconds=.03).prepare()
    try:
        _, info = profile.select(STATE, np.zeros(3))
        assert info["fault_reason"] == "baseline_preparation_exhausted_budget"
        assert info["deadline_observed_overrun"] and not info["deadline_primary_used"]
        assert profile.pending is None
    finally:
        profile.close()


def test_mismatched_reply_is_never_applied(monkeypatch):
    profile = DeadlineProfile(partial(Constant, -2.5), Constant(), budget_seconds=1.).prepare()
    monkeypatch.setattr(profile, "_exchange", lambda request: {"kind": "ok", "token": -1})
    try:
        command, info = profile.select(STATE, np.zeros(3))
        np.testing.assert_array_equal(command, np.zeros(3))
        assert "mismatched" in info["fault_reason"]
    finally:
        profile.close()


def test_startup_failure_cleans_up_actual_child():
    profile = DeadlineProfile(startup_failure, Constant())
    with pytest.raises(RuntimeError, match="startup failed"):
        profile.prepare()
    assert profile.closed and not profile.process.is_alive()


def test_physically_uncertified_proposal_is_rejected():
    state = np.array([[-50., 12.], [-51., 12.], [-90., 11.]])
    profile = DeadlineProfile(partial(Constant, 2.5), Constant(-2.5), budget_seconds=1.).prepare()
    try:
        command, info = profile.select(state, np.zeros(3))
        np.testing.assert_array_equal(command, np.full(3, -2.5))
        assert info["fault_fallback"] and "certificate" in info["fault_reason"]
    finally:
        profile.close()


def test_main_side_validation_time_is_included(monkeypatch):
    profile = DeadlineProfile(partial(Constant, -2.5), Constant(), budget_seconds=.04).prepare()
    original = profile._checked
    def slow_check(state, previous, command, info):
        result = original(state, previous, command, info)
        if np.all(np.asarray(command) == -2.5):
            time.sleep(.1)
        return result
    monkeypatch.setattr(profile, "_checked", slow_check)
    try:
        command, info = profile.select(STATE, np.zeros(3))
        np.testing.assert_array_equal(command, np.zeros(3))
        assert info["deadline_triggered_this_step"] and info["deadline_observed_overrun"]
        assert not info["deadline_primary_used"]
    finally:
        profile.close()


@pytest.mark.parametrize("value", [0., -.1, float("nan"), float("inf")])
def test_invalid_budget_rejected(value):
    with pytest.raises(ValueError):
        DeadlineProfile(partial(Constant), Constant(), budget_seconds=value)


def test_select_requires_preparation_and_refuses_after_close():
    profile = DeadlineProfile(None, Constant(), admitted=False)
    with pytest.raises(RuntimeError):
        profile.select(STATE, np.zeros(3))
    profile.prepare()
    profile.close()
    with pytest.raises(RuntimeError):
        profile.select(STATE, np.zeros(3))
