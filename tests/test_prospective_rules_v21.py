import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from prepare_prospective_validation_v21 import declared_blocks, require_terminal


def save(path, value): path.write_text(json.dumps(value), encoding='utf-8')


def test_live_or_failed_supervisors_cannot_unlock_outcome_reads(tmp_path):
    logs = tmp_path / 'execution'
    logs.mkdir()
    with pytest.raises(ValueError, match='has not exited'): require_terminal(logs, 'protocol', 3)
    save(Path(str(logs) + '.supervisor_exit.json'), {'exit_code': 1})
    with pytest.raises(ValueError, match='successfully'): require_terminal(logs, 'protocol', 3)


def test_all_original_exits_and_correct_protocol_are_required(tmp_path):
    logs = tmp_path / 'execution'
    logs.mkdir()
    save(Path(str(logs) + '.supervisor_exit.json'), {'exit_code': 0})
    save(logs / 'launch.json', dict(protocol_sha256='protocol', worker_pids=[101, 102, 103]))
    save(logs / 'state.json', dict(phase='completed'))
    exits = dict(workers=[dict(pid=p, exit_code=0) for p in (101, 102, 103)])
    save(logs / 'worker_exits.json', exits)
    assert len(require_terminal(logs, 'protocol', 3)) == 4
    with pytest.raises(ValueError, match='protocol'): require_terminal(logs, 'other', 3)
    exits['workers'][2]['pid'] = 999
    save(logs / 'worker_exits.json', exits)
    with pytest.raises(ValueError, match='original worker'): require_terminal(logs, 'protocol', 3)


def test_reserved_range_logic_keeps_all_shifted_scenario_offsets():
    assert declared_blocks(dict(seed_base=100, n_initials_per_scenario=32,
                                scenarios=['nominal', 'dense', 'slow'])) == [(100, 32), (1100, 32), (2100, 32)]
    assert declared_blocks(dict(seed_base=4000, n_initials=4096)) == [(4000, 4096)]
    assert declared_blocks(dict(seed_base=1, initials=512)) == [(1, 512)]
