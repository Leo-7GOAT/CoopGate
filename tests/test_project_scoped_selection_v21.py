from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import select_completed_v21_scoped as selector


def process(pid, parent, command):
    return dict(ProcessId=pid, ParentProcessId=parent, CommandLine=command, Name='python.exe')


def test_unrelated_projects_do_not_prevent_completed_study_selection(monkeypatch):
    monkeypatch.setattr(selector, 'ROOT', Path('C:/workspace/CoopGate'))
    rows = [process(10, 2, 'C:/other/.venv/python.exe C:/other/train.py')]
    assert selector.require_project_stopped(rows, own_pid=90, parent_pid=91)['unrelated_process_count'] == 1


def test_same_virtual_environment_is_conservatively_part_of_the_project():
    root = Path('C:/workspace/CoopGate')
    rows = [process(10, 2, 'C:/workspace/.venv/Scripts/python.exe tools/some_job.py')]
    assert len(selector.project_processes(rows, root=root)) == 1


def test_relative_frozen_worker_and_its_sumo_descendants_are_not_ignored():
    rows = [process(10, 2, 'python tools/continue_v21_20261022.py finalize transfer'),
            process(11, 10, 'sumo -c short.sumocfg'), process(12, 11, None)]
    assert len(selector.project_processes(rows)) == 3
    with pytest.raises(ValueError):
        selector.require_project_stopped(rows, own_pid=90, parent_pid=91)


def test_a_live_project_worker_remains_a_blocker_with_unrelated_workers():
    rows = [process(10, 2, 'python tools/study_phase_repair_v21.py worker 0'),
            process(20, 2, 'C:/other/python.exe C:/other/train.py')]
    with pytest.raises(ValueError):
        selector.require_project_stopped(rows, own_pid=90, parent_pid=91)


def test_similar_directory_prefix_is_not_this_repository():
    rows = [process(10, 2, 'python C:/workspace/CoopGate-old/train.py'),
            process(11, 2, 'C:/workspace/.venv-old/python.exe train.py')]
    assert selector.project_processes(rows, root=Path('C:/workspace/CoopGate')) == []


def test_own_selector_and_ancestors_are_allowed_but_siblings_are_not():
    rows = [process(10, 2, 'python tools/continue_v21_20261022.py select'),
            process(11, 10, 'python tools/continue_v21_20261022.py select')]
    selector.require_project_stopped(rows, own_pid=11, parent_pid=10)
    with pytest.raises(ValueError):
        selector.require_project_stopped(rows+[process(12, 10, 'python tools/continue_v21_20261022.py finalize phase')],
                                         own_pid=11, parent_pid=10)
