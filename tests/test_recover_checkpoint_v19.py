import copy
import errno
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location('recover_checkpoint_v19', TOOLS / 'recover_checkpoint_v19.py')
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def test_transient_replacement_retries_without_rebuilding_or_losing_data(tmp_path, monkeypatch):
    target = tmp_path / 'episodes.json'
    target.write_bytes(b'old')
    original = os.replace
    calls = []

    def intermittent(source, destination):
        calls.append(Path(source).read_bytes())
        if len(calls) < 3:
            raise PermissionError(errno.EACCES, 'Sharing violation')
        original(source, destination)

    monkeypatch.setattr(recovery.os, 'replace', intermittent)
    monkeypatch.setattr(recovery.time, 'sleep', lambda _: None)
    recovery.atomic_bytes(target, b'new complete checkpoint')
    assert calls == [b'new complete checkpoint'] * 3
    assert target.read_bytes() == b'new complete checkpoint'
    assert not list(tmp_path.glob('*.tmp'))


def test_persistent_lock_exhausts_budget_and_keeps_both_versions(tmp_path, monkeypatch):
    source, target = tmp_path / 'pending', tmp_path / 'episodes.json'
    source.write_bytes(b'new')
    target.write_bytes(b'old')
    attempts = []

    def locked(*_):
        attempts.append(1)
        raise PermissionError(errno.EACCES, 'Still locked')

    monkeypatch.setattr(recovery.os, 'replace', locked)
    with pytest.raises(PermissionError):
        recovery.retry_replace(source, target, attempts=3, delay=0)
    assert len(attempts) == 3 and source.read_bytes() == b'new' and target.read_bytes() == b'old'


def test_nonsharing_failure_is_not_retried(tmp_path, monkeypatch):
    calls = []

    def disk_failure(*_):
        calls.append(1)
        raise OSError(errno.ENOSPC, 'Disk full')

    monkeypatch.setattr(recovery.os, 'replace', disk_failure)
    with pytest.raises(OSError):
        recovery.retry_replace(tmp_path / 'source', tmp_path / 'target', attempts=3, delay=0)
    assert len(calls) == 1


def fixture():
    design = dict(seed_base=61000001, n_initials=8, shards=4, sampling='iid_mixture', modes=['clean','pulse'], methods={'baseline':{'kind':'baseline'}})
    rows = [dict(initial_index=i, disturbance=mode, controller=name, seed=design['seed_base']+i,
                 scenario=recovery.previous.regime_for(design,i), completed=False, original_timing=19.7)
            for i,mode,name in recovery.previous.schedule(design,2)]
    return design, rows


def test_pending_promotion_preserves_failed_results_and_original_timings():
    design, rows = fixture()
    encode = recovery.previous.encoded_rows
    pending, _, newline = recovery.inspect_pending(encode(rows[:2],'\r\n'),encode(rows[:3],'\r\n'),design,2)
    assert pending == rows[:3] and newline == '\r\n'
    assert not pending[-1]['completed'] and pending[-1]['original_timing'] == 19.7


@pytest.mark.parametrize('mutation',['changed_prefix','skipped','two_new','truncated'])
def test_invalid_pending_checkpoint_is_rejected(mutation):
    design, rows = fixture()
    encode = recovery.previous.encoded_rows
    pending = copy.deepcopy(rows[:3])
    if mutation == 'changed_prefix': pending[0]['original_timing'] += 1
    elif mutation == 'skipped': pending[-1] = rows[-1]
    elif mutation == 'two_new': pending = rows
    raw = encode(pending,'\r\n')
    if mutation == 'truncated': raw = raw[:-8]
    with pytest.raises((ValueError,json.JSONDecodeError)):
        recovery.inspect_pending(encode(rows[:2],'\r\n'),raw,design,2)


@pytest.mark.skipif(os.name != 'nt', reason='Actual Windows sharing semantics')
def test_real_windows_read_handle_blocks_rename_then_retry_succeeds(tmp_path):
    import threading
    target, pending = tmp_path / 'episodes.json', tmp_path / 'pending.json'
    target.write_bytes(b'old')
    pending.write_bytes(b'new complete record')
    reader = target.open('rb')
    try:
        with pytest.raises(PermissionError):
            os.replace(pending, target)
        release = threading.Timer(.15, reader.close)
        release.start()
        recovery.retry_replace(pending, target, attempts=30, delay=.02)
        release.join()
        assert target.read_bytes() == b'new complete record' and not pending.exists()
    finally:
        reader.close()
