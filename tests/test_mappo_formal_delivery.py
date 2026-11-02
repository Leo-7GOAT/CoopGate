import hashlib
import json
from pathlib import Path
import sys
import zipfile
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import archive_mappo_parts as archive
import snapshot_mappo_replay_sources as snapshot

NAME='formal_mappo_attempt3_20261102'


def setup(tmp_path,monkeypatch):
    scope=tmp_path/'artifacts/mappo_ctde_20261101';study=scope/NAME;study.mkdir(parents=True)
    monkeypatch.setattr(archive,'ROOT',tmp_path);monkeypatch.setattr(archive,'SCOPE',scope)
    monkeypatch.setattr(snapshot,'ROOT',tmp_path)
    return study,tmp_path/'delivery'


def write(path,value):path.write_text(json.dumps(value),encoding='utf-8')


def completed(study):
    (study/'analysis').mkdir()
    write(study/'analysis/certificate.json',dict(joint_deployment_certified=False))
    write(study/'analysis/comparisons.json',dict(negative_results_preserved=True))
    receipt=dict(actual_native_finalizer=dict(actual_native_exit_code=0),total_actual_sumo_runs=37888,
        main_actual_sumo_runs=16384,independent_initials=4096,
        stages=[dict(stage=s,action=a,shard=i,actual_native_exit_code=0)
            for s in ('calibration','confirmation') for a in ('collect','review') for i in range(4)])
    for key in ('certificate','comparisons'):
        receipt[key+'_sha256']=hashlib.sha256((study/f'analysis/{key}.json').read_bytes()).hexdigest()
    write(study/'completion_receipt.json',receipt)
    write(study/'orchestrator_native_receipt.json',dict(actual_native_exit_code=0))
    write(study/'source_snapshot.json',dict(all_bytes_verified=True))
    return receipt


def test_live_formal_study_cannot_create_a_completed_delivery(tmp_path,monkeypatch):
    study,out=setup(tmp_path,monkeypatch)
    with pytest.raises(FileNotFoundError):archive.archive(out,[NAME])
    assert not out.exists()


@pytest.mark.parametrize('failure',('nonzero','undersized','duplicate','changed_analysis'))
def test_failed_incomplete_or_changed_evidence_cannot_be_published_as_complete(tmp_path,monkeypatch,failure):
    study,out=setup(tmp_path,monkeypatch);receipt=completed(study)
    if failure=='nonzero':receipt['stages'][0]['actual_native_exit_code']=1
    elif failure=='undersized':receipt['total_actual_sumo_runs']-=1
    elif failure=='duplicate':receipt['stages'][1]=receipt['stages'][0]
    else:write(study/'analysis/certificate.json',dict(joint_deployment_certified=True))
    write(study/'completion_receipt.json',receipt)
    with pytest.raises(ValueError):archive.archive(out,[NAME])
    assert not out.exists()


def test_completed_negative_science_is_preserved_byte_exactly(tmp_path,monkeypatch):
    study,out=setup(tmp_path,monkeypatch);completed(study)
    archive.archive(out,[NAME]);inventory=json.loads((out/'inventory.json').read_text())
    assert inventory['all_archived_bytes_verified']
    with zipfile.ZipFile(out/'completed_part_001.zip') as bundle:
        assert json.loads(bundle.read((study/'analysis/certificate.json').relative_to(tmp_path).as_posix()))['joint_deployment_certified'] is False
        for item in inventory['inventory']:
            assert bundle.read(item['path'])==(tmp_path/item['path']).read_bytes()


def test_nonzero_owner_cannot_create_formal_source_snapshot(tmp_path,monkeypatch):
    study,out=setup(tmp_path,monkeypatch);completed(study)
    write(study/'orchestrator_native_receipt.json',dict(actual_native_exit_code=1))
    with pytest.raises(ValueError):snapshot.snapshot(study)
    assert not (study/'source').exists()
