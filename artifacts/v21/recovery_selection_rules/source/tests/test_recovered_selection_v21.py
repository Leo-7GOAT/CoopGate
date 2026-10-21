import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from select_recovered_v21 import completion_receipts


def write(path, value):
    path.write_text(json.dumps(value))


def receipts(path):
    write(path / 'supervisor_exit.json', dict(exit_code=0, recovery_plan_sha256='frozen'))
    write(path / 'completed.json', dict(original_exit_status='unknown'))
    for stage, count in (('preflight', 3), ('suffix', 3), ('review', 2)):
        commands = ([['worker', stage, str(i)] for i in range(3)] if stage != 'review'
                    else [['finalize', 'phase'], ['finalize', 'transfer']])
        write(path / f'{stage}_launch.json', dict(stage=stage, recovery_plan_sha256='frozen',
            supervisor_pid=100, worker_pids=list(range(count)),
            commands=[['python', '-X', 'utf8', 'tools/recover_interrupted_v21.py', *c] for c in commands]))
        write(path / f'{stage}_exits.json', dict(workers=[dict(pid=i, exit_code=0) for i in range(count)]))


def test_complete_recovery_discloses_unknown_original_exits(tmp_path):
    receipts(tmp_path)
    assert completion_receipts(tmp_path, 'frozen')['original_exit_status'] == 'unknown'


@pytest.mark.parametrize('mode', ['missing_exit', 'failed_worker', 'wrong_plan', 'substituted_handle', 'false_original_exit', 'wrong_command'])
def test_recovery_guard_rejects_incomplete_or_substituted_evidence(tmp_path, mode):
    receipts(tmp_path)
    if mode == 'missing_exit':
        (tmp_path / 'supervisor_exit.json').unlink()
    elif mode == 'failed_worker':
        write(tmp_path / 'review_exits.json', dict(workers=[dict(pid=0, exit_code=0), dict(pid=1, exit_code=1)]))
    elif mode == 'wrong_plan':
        write(tmp_path / 'supervisor_exit.json', dict(exit_code=0, recovery_plan_sha256='different'))
    elif mode == 'substituted_handle':
        write(tmp_path / 'review_exits.json', dict(workers=[dict(pid=0, exit_code=0), dict(pid=9, exit_code=0)]))
    elif mode == 'false_original_exit':
        write(tmp_path / 'completed.json', dict(original_exit_status='success'))
    else:
        target = tmp_path / 'review_launch.json'
        value = json.loads(target.read_text())
        value['commands'][1][-1] = 'phase'
        write(target, value)
    with pytest.raises((ValueError, FileNotFoundError)):
        completion_receipts(tmp_path, 'frozen')
