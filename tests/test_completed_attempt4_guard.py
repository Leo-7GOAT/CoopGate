"""Postprocessing must refuse unfinished/mismatched studies before effects."""
import hashlib
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from mappo_completed_attempt4 import complete_evidence


def test_missing_owner_cannot_read_partial_effects(tmp_path):
    with pytest.raises(FileNotFoundError):complete_evidence(tmp_path)


def test_nonzero_root_cannot_be_replaced_by_successful_child_files(tmp_path):
    (tmp_path/'orchestrator_native_receipt.json').write_text('{"actual_native_exit_code":1}')
    with pytest.raises(ValueError,match='Root native owner'):complete_evidence(tmp_path)


def test_owner_must_bind_exact_completion_bytes(tmp_path):
    (tmp_path/'completion_receipt.json').write_text('{}')
    (tmp_path/'orchestrator_native_receipt.json').write_text('{"actual_native_exit_code":0,"completion_receipt_sha256":"wrong"}')
    with pytest.raises(ValueError,match='exact completed study'):complete_evidence(tmp_path)


def test_duplicate_native_stage_is_not_full_coverage(tmp_path):
    stages=[dict(stage=s,action=a,shard=i,actual_native_exit_code=0) for s in ('calibration','confirmation') for a in ('collect','review') for i in range(4)]
    stages[-1]=stages[0]
    receipt=dict(actual_native_finalizer=dict(actual_native_exit_code=0),total_actual_sumo_runs=37888,
        main_actual_sumo_runs=16384,independent_initials=4096,stages=stages)
    raw=json.dumps(receipt).encode();(tmp_path/'completion_receipt.json').write_bytes(raw)
    (tmp_path/'orchestrator_native_receipt.json').write_text(json.dumps(dict(actual_native_exit_code=0,
        completion_receipt_sha256=hashlib.sha256(raw).hexdigest())))
    with pytest.raises(ValueError,match='exact native collection/review'):complete_evidence(tmp_path)
