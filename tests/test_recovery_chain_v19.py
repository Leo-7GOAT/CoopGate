import importlib.util
from pathlib import Path
import sys

import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0,str(TOOLS))
spec = importlib.util.spec_from_file_location('audit_recovery_chain_v19', TOOLS / 'audit_recovery_chain_v19.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_normal_native_exit_must_bind_the_exact_completion_file(tmp_path):
    path = tmp_path / 'complete.json'
    path.write_bytes(b'{"actual_sumo_runs":10240}')
    receipt = dict(shard=2,exit_code=0,complete_record_exists=True,complete_record_sha256=audit.digest(path))
    audit.validate_exit(receipt,path,2)
    path.write_bytes(b'{"actual_sumo_runs":1}')
    with pytest.raises(ValueError,match='different completion'):
        audit.validate_exit(receipt,path,2)


@pytest.mark.parametrize('key,value',[('exit_code',1),('complete_record_exists',False),('shard',1)])
def test_abnormal_missing_or_wrong_worker_receipt_is_rejected(tmp_path,key,value):
    path = tmp_path / 'complete.json'
    path.write_bytes(b'{}')
    receipt = dict(shard=2,exit_code=0,complete_record_exists=True,complete_record_sha256=audit.digest(path))
    receipt[key] = value
    with pytest.raises(ValueError,match='did not exit normally'):
        audit.validate_exit(receipt,path,2)
