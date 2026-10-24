"""Regression coverage for the corrected V39 memoization recovery retry."""
import json
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import recover_memoized_recovery_v38 as core
import recover_memoized_recovery_v39 as retry
from coopgate.episode_store_v21 import EpisodeStore


def test_retry_uses_fresh_output_and_binds_failed_attempt():
    module = retry.configure()
    assert module.OUT == retry.OUT
    assert module.OUT != module.PRIOR
    assert module.FAILED_ATTEMPT == retry.FAILED_ATTEMPT
    assert module.RECOVERY_SEGMENT == 'recovered_v39'
    assert retry.SCRIPT in module.SOURCES and retry.TEST in module.SOURCES
    identity = module.failed_attempt_identity(processes=[])
    assert identity['exit_code'] == 1
    assert identity['new_sumo_runs'] == 0
    assert identity['plan_sha256'] == '2ea6da13fae8ab9994d3ac06eae460bee703dc1fa7dda904ca1852ceb399217d'


def test_model_paths_are_joined_with_path_operator_not_subscripted():
    text = Path(core.__file__).read_text(encoding='utf-8')
    assert "ROOT[item['path']]" not in text
    assert text.count("ROOT/item['path']") == 2


def test_original_spawn_receipt_is_frozen_and_descendants_forbid_recovery(tmp_path):
    assert 'execution/spawn.json' in core.EVIDENCE_FILES
    prior = tmp_path/'prior'; execution = prior/'execution'; execution.mkdir(parents=True)
    (prior/'watcher_launch.json').write_text(json.dumps({'launcher_pid':11}),encoding='utf-8')
    (execution/'launch.json').write_text(json.dumps(
        {'watcher_pid':12,'launcher_pid':13,'native_pid':14}),encoding='utf-8')
    processes = [
        {'ProcessId':20,'ParentProcessId':14,'CommandLine':'sumo'},
        {'ProcessId':21,'ParentProcessId':20,'CommandLine':'helper'},
    ]
    with pytest.raises(ValueError,match='still be live'):
        core.require_interrupted_processes_absent(prior,processes=processes)


def test_complete_receipt_metadata_is_part_of_block_identity(tmp_path):
    metadata = {'plan_sha256':'p','original_plan_sha256':'o',
                'block_index':74,'stage':'exact_memoization_explicit_recovery'}
    cases = [('case','original'),('case','memoized')]
    directory = tmp_path/'block'; directory.mkdir()
    store = EpisodeStore(directory/'observations.sqlite',metadata)
    for case in cases:
        store.append(case,{'row':{},'calls':[]},[{'command':[0,0,0]}])
    store.seal(cases)
    core.exclusive_json(directory/'complete.json',dict(**metadata,reviews=[
        {'case':list(case)} for case in cases],
        database_sha256=core.digest(directory/'observations.sqlite'),
        seal_sha256=core.digest(directory/'observations.sqlite.seal.json')))
    assert core.block_identity(directory,metadata,cases,complete=True)['complete']
    receipt = core.read(directory/'complete.json')
    receipt['plan_sha256'] = 'changed'
    (directory/'complete.json').write_text(json.dumps(receipt),encoding='utf-8')
    with pytest.raises(ValueError,match='proof'):
        core.block_identity(directory,metadata,cases,complete=True)


def test_summary_persists_exact_receipts_blocks_and_rejects_extra_directories():
    text = Path(core.__file__).read_text(encoding='utf-8')
    assert 'actual_recovery_names != expected_recovery_names' in text
    assert 'execution_receipts_sha256=execution_receipts' in text
    assert "('observations.sqlite','observations.sqlite.seal.json','complete.json')" in text
    assert "target/'completion_verification.json'" in text
    assert 'proof_metadata != expected_metadata' in text


def test_failed_attempt_guard_rejects_added_progress_and_incomplete_evidence(tmp_path,monkeypatch):
    target = tmp_path/'attempt'
    shutil.copytree(retry.FAILED_ATTEMPT,target)
    monkeypatch.setattr(core,'ROOT',tmp_path)
    monkeypatch.setattr(core,'committed',lambda _:None)
    progress = target/'execution/progress.json'
    progress.write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='pre-simulation'):
        core.failed_attempt_identity(target,processes=[])
    progress.unlink()
    proof_path = target/'failed_attempt_verification.json'
    proof = json.loads(proof_path.read_text(encoding='utf-8'))
    proof['execution_evidence_sha256'].pop('execution/spawn.json')
    proof_path.write_text(json.dumps(proof),encoding='utf-8')
    with pytest.raises(ValueError,match='evidence set'):
        core.failed_attempt_identity(target,processes=[])
