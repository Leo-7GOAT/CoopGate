"""Read-only delivery glue and actual-live refusal;no new driving episodes."""
import hashlib
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import replay_mappo_coupled_formal_delivery as replay


def test_replay_refuses_live_study_before_report_creation(tmp_path,monkeypatch):
    monkeypatch.setattr(replay.e,'ROOT',tmp_path);monkeypatch.setattr(replay.e,'FORMAL',tmp_path/'study')
    out=tmp_path/'runs/replay.json'
    with pytest.raises(FileNotFoundError):replay.replay('calibration',0,out)
    assert not out.parent.exists()


def test_complete_readonly_replay_preserves_original_bytes(tmp_path,monkeypatch):
    study=tmp_path/'study';directory=study/'calibration/shard0';directory.mkdir(parents=True)
    def write(path,value):path.write_text(json.dumps(value),encoding='utf-8')
    write(study/'plan.json',dict(fixture_only=True))
    for name in ('verification','collection_complete'):write(directory/(name+'.json'),dict(executions=5376))
    receipt=dict(actual_native_finalizer=dict(actual_native_exit_code=0),stages=[])
    for action in ('collect','review'):
        key='verification' if action=='review' else 'collection_complete'
        receipt['stages'].append(dict(stage='calibration',action=action,shard=0,actual_native_exit_code=0,
            **{key+'_sha256':hashlib.sha256((directory/(key+'.json')).read_bytes()).hexdigest()}))
    write(study/'completion_receipt.json',receipt);write(study/'orchestrator_native_receipt.json',dict(actual_native_exit_code=0))
    original={p:p.read_bytes() for p in study.rglob('*') if p.is_file()};guard_calls=[]
    monkeypatch.setattr(replay,'complete_evidence',lambda:guard_calls.append(True))
    monkeypatch.setattr(replay.e,'ROOT',tmp_path);monkeypatch.setattr(replay.e,'FORMAL',study)
    monkeypatch.setattr(replay.e,'bound',lambda *a:{})
    monkeypatch.setattr(replay.e,'load_models',lambda:{})
    monkeypatch.setattr(replay.e,'reviewed_records',lambda *a:(({},dict(method='baseline')) for _ in range(5376)))
    monkeypatch.setattr(replay.runtime,'audit_record',lambda *a:dict(reference_replayed_commands=2,physical_substeps=10,recorded_failure=0))
    out=tmp_path/'runs/replay.json';replay.replay('calibration',0,out)
    assert guard_calls==[True] and json.loads(out.read_bytes())['executions']==5376
    assert all(p.read_bytes()==data for p,data in original.items())
    assert json.loads(out.read_bytes())['new_sumo_runs']==0


def test_plotter_refuses_live_study_without_creating_output(tmp_path,monkeypatch):
    import plot_mappo_coupled_formal_results as plot
    monkeypatch.setattr(plot,'ROOT',tmp_path)
    def incomplete():raise ValueError('Actual root terminal witness absent')
    monkeypatch.setattr(plot,'complete_evidence',incomplete)
    out=tmp_path/'artifacts/mappo_ctde_20261101/figures'
    with pytest.raises(ValueError,match='terminal witness'):plot.render(out)
    assert not out.exists()
