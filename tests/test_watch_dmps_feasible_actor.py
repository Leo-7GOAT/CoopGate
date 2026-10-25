from datetime import datetime,timezone,timedelta

import pytest

import watch_dmps_feasible_actor as observer


def test_rebound_waiter_keeps_exact_code_and_preserves_original_module():
    original_audit,original_output=observer.waiter.audit,observer.waiter.OUT
    configured=observer.configured_waiter()
    assert configured.__code__ is observer.waiter.watch.__code__
    assert configured.__globals__['audit'] is observer.audit
    assert configured.__globals__['OUT']==observer.OUT
    assert observer.waiter.audit is original_audit and observer.waiter.OUT==original_output


def test_actual_new_command_parent_pid_and_creation_must_all_match():
    created=datetime(2026,10,25,tzinfo=timezone.utc)
    start=dict(native_pid=12,parent_pid=10,utc=(created+timedelta(seconds=12)).isoformat())
    current=dict(ProcessId=12,ParentProcessId=10,CommandLine='python tools/train_dmps_feasible_actor.py --source pinned')
    observer.validate_identity(start,current,created)
    for bad in (dict(current,ProcessId=13),dict(current,ParentProcessId=11),
                dict(current,CommandLine='python tools/train_dmps_effective_action.py')):
        with pytest.raises(ValueError):observer.validate_identity(start,bad,created)
    with pytest.raises(ValueError):observer.validate_identity(start,current,created-timedelta(hours=1))
