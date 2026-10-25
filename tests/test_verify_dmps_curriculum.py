import json
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import verify_dmps_curriculum as audit


def native_fixture(tmp_path):
    source=audit.run.PROBE/'execution'
    if not source.exists():pytest.skip('Completed native curriculum probe fixture not available')
    record=audit.io.read(source/'episode_000/episode.json')
    target=tmp_path/'execution';target.mkdir()
    shutil.copytree(source/'episode_000',target/'episode_000')
    for index in range(len(record['decisions'])):
        shutil.copy2(source/f'call_{index:04d}.json',target/f'call_{index:04d}.json')
    return [record['case']]


def test_replay_accepts_real_native_first_curriculum_episode(tmp_path):
    cases=native_fixture(tmp_path)
    result=audit.audit_execution_tree(tmp_path,'execution',cases,curriculum_training=True)
    assert result['episodes'][0]['metadata']['review']['completed']
    assert len(result['calls'])>0


def test_changed_real_reward_rejected_even_if_physics_and_completion_are_intact(tmp_path):
    cases=native_fixture(tmp_path);path=tmp_path/'execution/call_0000.json'
    data=audit.io.read(path);data['reward']+=.1;path.write_text(json.dumps(data),encoding='utf-8')
    with pytest.raises(AssertionError):audit.audit_execution_tree(tmp_path,'execution',cases,curriculum_training=True)


def test_curriculum_reset_cannot_be_passed_off_as_ordinary_test_case(tmp_path):
    cases=native_fixture(tmp_path)
    with pytest.raises(AssertionError):audit.audit_execution_tree(tmp_path,'execution',cases)


def test_wrong_phase_schedule_rejected_with_unchanged_actual_trajectory(tmp_path):
    cases=native_fixture(tmp_path);path=tmp_path/'execution/episode_000/initialization.json'
    data=audit.io.read(path);data['phase']=1;path.write_text(json.dumps(data),encoding='utf-8')
    with pytest.raises(AssertionError):audit.audit_execution_tree(tmp_path,'execution',cases,curriculum_training=True)


def test_unavailable_shadow_reference_is_not_zero_adoption(tmp_path,monkeypatch):
    cases=native_fixture(tmp_path)
    class FailedReference:
        def __init__(self,*args):pass
        def select(self,*args):raise RuntimeError('Reference outside its domain')
    monkeypatch.setattr(audit.run.pairing.baseline,'MemoizedRecoveryGate',FailedReference)
    result=audit.audit_execution_tree(tmp_path,'execution',cases,curriculum_training=True,shadow_reference=True)
    episode=result['episodes'][0]
    assert episode['adoption_vs_reference'] is None
    assert episode['shadow_errors'][0]['step']==0
