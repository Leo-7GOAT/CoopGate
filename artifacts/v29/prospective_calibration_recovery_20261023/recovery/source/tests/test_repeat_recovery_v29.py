"""Repeated recovery cannot replace outcomes, skip prefix boundaries, or change science."""
import json
from pathlib import Path
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
sys.path.insert(0, str(ROOT/'tests'))
import recover_prospective_v29 as recovery
import snapshot_interruption_v29 as snapshot
from test_recover_prospective_v24 import fixture
from coopgate.episode_store_v21 import readonly


@pytest.mark.parametrize('count', [6, 34])
def test_current_partial_lengths_retain_every_original_blob_and_fill_only_suffix(tmp_path, monkeypatch, count):
    f = fixture(tmp_path, monkeypatch, count)
    with readonly(f.directory/'observations.sqlite') as connection:
        before = connection.execute('SELECT * FROM episodes ORDER BY position').fetchall()
    recovery.mechanics.complete_partial(f.entry, f.plan, f.env, f.fs)
    with readonly(f.output/'calibration/blocks/00000/observations.sqlite') as connection:
        after = connection.execute('SELECT * FROM episodes ORDER BY position').fetchall()
    assert after[:count] == before
    assert len(after) == 50
    assert f.calls == f.expected[count:]
    decision = recovery.original.read(f.output/'calibration/blocks/00000/decision.json')
    assert decision['prediction']['full_prediction_wall_ms'] is None if count < 10 else decision['prediction']['full_prediction_wall_ms'] == 123.456


def test_archive_parts_keep_order_no_file_duplicates_and_bound_size(tmp_path):
    files = []
    for i, size in enumerate([4, 5, 8, 1]):
        path = tmp_path/str(i)
        path.write_bytes(b'a'*size)
        files.append(path)
    parts = snapshot.parts_for(files, limit=9)
    assert parts == [files[:2], files[2:]]
    with pytest.raises(ValueError, match='Single file'):
        snapshot.parts_for(files, limit=7)


def test_source_and_archive_bytes_must_both_remain_exact(tmp_path, monkeypatch):
    source = tmp_path/'source'
    source.mkdir()
    (source/'data.bin').write_bytes(b'original arbitrary outcome bytes')
    target = tmp_path/'evidence/calibration'
    target.mkdir(parents=True)
    archive = target/'part.zip'
    with zipfile.ZipFile(archive, 'x') as handle:
        handle.write(source/'data.bin', 'data.bin')
    evidence = dict(source_directory='source', original_files=snapshot.files_at(source),
        archive_parts=[dict(name='part.zip', sha256=recovery.original.digest(archive), files=['data.bin'])])
    (target/'inventory.json').write_text(json.dumps(evidence), encoding='utf-8')
    monkeypatch.setattr(snapshot, 'ROOT', tmp_path)
    monkeypatch.setattr(snapshot, 'OUT', tmp_path/'evidence')
    assert snapshot.verify_snapshot('calibration', False) == evidence
    (source/'data.bin').write_bytes(b'changed outcome')
    with pytest.raises(ValueError, match='source files'):
        snapshot.verify_snapshot('calibration', False)
    (source/'data.bin').write_bytes(b'original arbitrary outcome bytes')
    with archive.open('ab') as handle:
        handle.write(b'tampered')
    with pytest.raises(ValueError, match='archive part'):
        snapshot.verify_snapshot('calibration', False)


def test_explicit_configuration_reuses_the_same_boundary_worker_and_inference(monkeypatch):
    mechanics = recovery.mechanics
    retained = {name:getattr(mechanics, name) for name in ('boundary','copy_prefix','complete_partial','worker','supervise','complete_stage','analyze')}
    for name in ('PRIOR','OUTPUT','RECOVERY','SCRIPT','inputs'):
        monkeypatch.setattr(mechanics, name, getattr(mechanics, name))
    source_outputs = dict(recovery.original.OUTPUTS)
    recovery.configure()
    assert all(getattr(mechanics, name) is value for name, value in retained.items())
    assert recovery.original.OUTPUTS == source_outputs
    assert mechanics.PRIOR != mechanics.OUTPUT
    assert mechanics.SCRIPT == recovery.SCRIPT
    assert mechanics.inputs is recovery.inputs


def test_live_target_refuses_interruption_even_without_a_current_progress_file(monkeypatch):
    monkeypatch.setattr(snapshot.original, 'require_project_stopped', lambda _:None)
    monkeypatch.setattr(snapshot.original, 'native_snapshot', lambda:[dict(ProcessId=987654,
        ParentProcessId=765432, Name='python.exe', CommandLine='python tools/recover_prospective_v24.py worker calibration collect 0')])
    with pytest.raises(ValueError, match='still live'):
        snapshot.assert_stopped()
