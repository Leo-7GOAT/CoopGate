from pathlib import Path
import json
import sys
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import joint_recovery_20261031 as recovery


def test_private_engine_changes_only_recovery_location_and_preserves_prior_graph():
    old=recovery.base
    before=(old.OUT,old.PLAN,old.SCRIPT)
    engine=recovery.routed_engine()
    assert engine['PRIOR']==old.OUT
    assert engine['OUT']==recovery.OUT
    assert engine['inputs'].__code__ is old.inputs.__code__
    assert engine['snapshot'].__code__ is old.snapshot.__code__
    assert set(old.SOURCES)<=set(recovery.SOURCES)
    assert old.PLAN.relative_to(recovery.ROOT).as_posix() in recovery.SOURCES
    namespace=engine['routed_functions']()
    assert namespace['collect'].collect_cluster.__globals__['PRIOR']==old.OUT
    assert namespace['collect'].audit_record is old.original.collect.audit_record
    for name in ('worker','watch','finalize'):
        assert namespace[name].__code__ is getattr(old.original,name).__code__
    assert namespace['certificate_from_complete_records'] is old.original.certificate_from_complete_records
    assert (old.OUT,old.PLAN,old.SCRIPT)==before


def test_running_prior_cannot_be_recovered_but_reused_pid_is_not_same_identity(monkeypatch,tmp_path):
    def response(command):return SimpleNamespace(stdout=json.dumps([dict(ProcessId=62592,CommandLine=command)]).encode())
    monkeypatch.setattr(recovery.base,'OUT',tmp_path)
    monkeypatch.setattr(recovery.base.subprocess,'run',lambda *a,**k:response('python tools/joint_recovery_20261028.py worker calibration'))
    with pytest.raises(ValueError,match='still live'):recovery.assert_stopped()
    monkeypatch.setattr(recovery.base.subprocess,'run',lambda *a,**k:response('unrelated process'))
    recovery.assert_stopped()


def test_batch_queue_is_local_one_shot_with_no_automatic_restart_or_network_push():
    launch=(recovery.ROOT/'operations/launch_joint_recovery_task_20261031.ps1').read_text()
    queue=(recovery.ROOT/'operations/continue_joint_recovery_20261031.ps1').read_text()
    assert '-LogonType S4U' in launch and '-RunLevel Limited' in launch
    assert 'triggers=0' in launch and 'automatic_retry=$false' in launch
    assert "'push','origin','main'" not in queue
    assert 'SetThreadExecutionState' in queue
