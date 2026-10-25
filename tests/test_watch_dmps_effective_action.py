from datetime import datetime,timezone,timedelta
import json

import pytest

from watch_dmps_effective_action import successful_receipt,validate_identity,io


def test_reused_pid_or_different_parent_command_and_creation_are_rejected():
    creation=datetime(2026,10,25,tzinfo=timezone.utc)
    start=dict(native_pid=12,parent_pid=10,utc=(creation+timedelta(seconds=12)).isoformat())
    current=dict(ProcessId=12,ParentProcessId=10,CommandLine='python tools/train_dmps_effective_action.py --source pinned')
    validate_identity(start,current,creation)
    for changed in (dict(current,ProcessId=13),dict(current,ParentProcessId=11),dict(current,CommandLine='python other.py')):
        with pytest.raises(ValueError):validate_identity(start,changed,creation)
    with pytest.raises(ValueError):validate_identity(start,current,creation-timedelta(hours=1))


def test_nonzero_exit_or_partial_completion_cannot_trigger_successful_audit(tmp_path):
    (tmp_path/'protocol.json').write_text('{}')
    (tmp_path/'native_start.json').write_text('{}')
    done=dict(training_actual_decisions=32768,new_policy_conditions=10,old_policy_conditions=10,baseline_conditions=10)
    (tmp_path/'complete.json').write_text(json.dumps(done))
    identity=dict(native_pid=12,native_creation_filetime=123,protocol_sha256=io.digest(tmp_path/'protocol.json'),
        native_start_sha256=io.digest(tmp_path/'native_start.json'))
    receipt=successful_receipt(tmp_path,identity,0)
    assert receipt['exit_code']==0 and receipt['complete_sha256']==io.digest(tmp_path/'complete.json')
    with pytest.raises(RuntimeError):successful_receipt(tmp_path,identity,1)
    (tmp_path/'complete.json').write_text(json.dumps(dict(done,new_policy_conditions=9)))
    with pytest.raises(ValueError):successful_receipt(tmp_path,identity,0)
