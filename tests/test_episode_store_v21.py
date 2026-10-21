import json
import os
import sqlite3
import subprocess
import sys

import pytest

from coopgate.episode_store_v21 import EpisodeStore, inventory, observations, seal_path, unpack


def test_atomic_record_retains_failure_trace_and_exact_numerics(tmp_path):
    path = tmp_path / 'block.sqlite'
    metadata = dict(protocol='frozen', block=0)
    case = (0, 'pulse', 'raw_1802')
    row = dict(completed=False, aborted=True, team_cost=25.123456789, reason='retained')
    trace = [dict(command=[-5., 2.5, 0.], forecast_errors=['incomplete'], gate_ms=3.2)]
    store = EpisodeStore(path, metadata)
    store.append(case, row, trace)
    store.seal([case])
    assert list(observations(path, metadata, [case])) == [(case, row, trace)]
    with pytest.raises(FileExistsError):
        EpisodeStore(path, metadata)
    with pytest.raises(ValueError):
        store.append(case, row, trace)


def test_duplicate_transaction_rolls_back_without_losing_prior_observation(tmp_path):
    path = tmp_path / 'block.sqlite'
    store = EpisodeStore(path, {})
    first, second = (0, 'clean', 'baseline'), (0, 'clean', 'raw')
    store.append(first, {'completed': False}, [])
    with pytest.raises(sqlite3.IntegrityError):
        store.append(first, {'completed': True}, [])
    store.append(second, {'completed': True}, [])
    store.seal([first, second])
    assert list(observations(path, {}, [first, second]))[0][1]['completed'] is False


def test_incomplete_or_reordered_block_cannot_be_sealed_or_analyzed(tmp_path):
    path = tmp_path / 'block.sqlite'
    store = EpisodeStore(path, {})
    store.append((0, 'clean', 'baseline'), {}, [])
    with pytest.raises(ValueError):
        store.seal([(0, 'clean', 'baseline'), (0, 'bias', 'baseline')])
    assert not seal_path(path).exists()
    with pytest.raises(FileNotFoundError):
        list(observations(path, {}, [(0, 'clean', 'baseline')]))


def test_seal_rejects_different_protocol_schedule_and_tampered_database(tmp_path):
    path = tmp_path / 'block.sqlite'
    case = (0, 'clean', 'baseline')
    store = EpisodeStore(path, dict(protocol='one'))
    store.append(case, {'cost': 1.}, [])
    store.seal([case])
    with pytest.raises(ValueError):
        list(observations(path, dict(protocol='two'), [case]))
    with pytest.raises(ValueError):
        list(observations(path, dict(protocol='one'), [(1, 'clean', 'baseline')]))
    with sqlite3.connect(path) as connection:
        connection.execute('UPDATE episodes SET row_sha256=?', ('changed',))
    with pytest.raises(ValueError):
        list(observations(path, dict(protocol='one'), [case]))


def test_committed_row_and_trace_survive_abrupt_writer_exit(tmp_path):
    path = tmp_path / 'interrupted.sqlite'
    script = """
import os,sys
from coopgate.episode_store_v21 import EpisodeStore
s=EpisodeStore(sys.argv[1], {'protocol':'crash-test'})
s.append((0,'pulse','raw'), {'completed':False}, [{'aborted':True}])
os._exit(7)
"""
    result = subprocess.run([sys.executable, '-c', script, str(path)], capture_output=True)
    assert result.returncode == 7
    state = inventory(path)
    assert state['cases'] == [[0, 'pulse', 'raw']]
    with sqlite3.connect(path) as connection:
        row_blob, trace_blob, row_sha, trace_sha = connection.execute(
            'SELECT row_blob,trace_blob,row_sha256,trace_sha256 FROM episodes').fetchone()
    assert unpack(row_blob, row_sha) == {'completed': False}
    assert unpack(trace_blob, trace_sha) == [{'aborted': True}]
    assert not seal_path(path).exists()
