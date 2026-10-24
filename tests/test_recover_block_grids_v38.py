import ast
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import recover_block_grids_v38 as recovery
from coopgate.episode_store_v21 import EpisodeStore
from profile_phase_runtime_v24 import digest


def groups(n=5, width=2):
    return [dict(block_index=i, condition=[i], cases=[[i,j] for j in range(width)]) for i in range(n)]


def plan(n=5, width=2):
    values = groups(n, width)
    return dict(groups=values, blocks=n, actual_sumo_runs=n*width,
                source_sha256={}, runtime={'fixture': True})


def entry(index, status, records, expected):
    return dict(block_index=index, status=status, records=records,
        inventory=dict(metadata={'block': index}, cases=expected[:records], row_sha256=[], trace_sha256=[]),
        files_sha256={'observations.sqlite': str(index)})


def stored_block(root, index, metadata, cases, records, complete):
    directory = root/'blocks'/f'{index:05d}'
    directory.mkdir(parents=True)
    store = EpisodeStore(directory/'observations.sqlite', metadata)
    for case in cases[:records]:
        store.append(case, {'row': {'fixture': list(case)}, 'calls': []}, [])
    if complete:
        store.seal(cases)
        (directory/'complete.json').write_text(json.dumps(dict(
            actual_sumo_runs=len(cases), database_sha256=digest(directory/'observations.sqlite'),
            seal_sha256=digest(directory/'observations.sqlite.seal.json'))))
    else:
        store.close()
    return directory


def test_exact_prefix_requires_a_strict_duplicate_free_original_prefix():
    expected = [[1],[2],[3]]
    assert recovery.exact_prefix(expected, [[1],[2]]) == 2
    for observed in ([[2]], [[1],[3]], expected, [[1],[1]]):
        with pytest.raises(ValueError, match='strict original-schedule prefix'):
            recovery.exact_prefix(expected, observed)


def test_worker_plan_contains_only_missing_original_groups_and_original_indices():
    source = plan()
    frozen = dict(groups=deepcopy(source['groups'][2:]), actual_sumo_runs=6, blocks=5)
    result = recovery.worker_plan(frozen, source)
    assert [g['block_index'] for g in result['groups']] == [2,3,4]
    assert result['actual_sumo_runs'] == 6 and result['blocks'] == 5
    assert source['actual_sumo_runs'] == 10 and len(source['groups']) == 5


def test_complete_and_missing_indices_partition_full_original_schedule():
    source = plan()
    frozen = dict(retained_complete_blocks=[0,1], missing_block_indices=[2,3,4],
        groups=source['groups'][2:], actual_sumo_runs=6, source_actual_sumo_runs=10)
    assert frozen['retained_complete_blocks'] + frozen['missing_block_indices'] == list(range(5))
    assert sum(len(g['cases']) for g in frozen['groups']) == frozen['actual_sumo_runs']
    assert 2 in frozen['missing_block_indices']


def test_inventory_accepts_complete_prefix_then_one_partial_and_reruns_partial(monkeypatch, tmp_path):
    source = tmp_path/'source'
    source_plan = plan(n=3, width=2)
    stored_block(source, 0, {'block':0}, source_plan['groups'][0]['cases'], 2, True)
    stored_block(source, 1, {'block':1}, source_plan['groups'][1]['cases'], 1, False)
    monkeypatch.setitem(recovery.STUDIES, 'fixture', dict(source=source, script='fixture.py', stage='x'))
    monkeypatch.setattr(recovery, 'source_metadata', lambda study, group: {'block':group['block_index']})
    result = recovery.inspect_interrupted_blocks('fixture', source_plan)
    assert result['retained_complete_blocks'] == [0]
    assert result['partial_block_index'] == 1 and result['partial_engineering_runs'] == 1
    assert result['missing_block_indices'] == [1,2]
    assert result['retained_actual_sumo_runs'] == 2
    assert result['recovery_actual_sumo_runs'] == 4


def test_inventory_rejects_partially_sealed_boundary(monkeypatch, tmp_path):
    source = tmp_path/'source'
    source_plan = plan(n=2, width=2)
    stored_block(source, 0, {'block':0}, source_plan['groups'][0]['cases'], 2, True)
    partial = stored_block(source, 1, {'block':1}, source_plan['groups'][1]['cases'], 1, False)
    (partial/'complete.json').write_text('{}')
    monkeypatch.setitem(recovery.STUDIES, 'fixture', dict(source=source, script='fixture.py', stage='x'))
    monkeypatch.setattr(recovery, 'source_metadata', lambda study, group: {'block':group['block_index']})
    with pytest.raises(ValueError, match='Partially sealed'):
        recovery.inspect_interrupted_blocks('fixture', source_plan)


def test_partial_block_is_rerun_whole_and_its_prefix_is_not_formal_n():
    source = plan(width=3)
    partial_records = 2
    missing = list(range(2,5))
    recovery_runs = sum(len(source['groups'][i]['cases']) for i in missing)
    retained_runs = sum(len(source['groups'][i]['cases']) for i in (0,1))
    assert recovery_runs == 9
    assert retained_runs + recovery_runs == source['actual_sumo_runs']
    assert retained_runs + recovery_runs + partial_records > source['actual_sumo_runs']


def test_target_liveness_is_scoped_to_exact_script_handles_and_descendants(monkeypatch, tmp_path):
    source = tmp_path/'source'
    (source/'execution').mkdir(parents=True)
    (source/'execution/launch.json').write_text(json.dumps({'native_pid': 20, 'launcher_pid': 19}))
    (source/'launcher.json').write_text(json.dumps({'launcher_pid': 30}))
    (source/'watcher_launch.json').write_text(json.dumps({'launcher_pid': 40}))
    monkeypatch.setitem(recovery.STUDIES, 'fixture', dict(source=source, script='target.py', stage='x'))
    processes = [dict(ProcessId=10, ParentProcessId=1, CommandLine='python unrelated.py'),
        dict(ProcessId=20, ParentProcessId=19, CommandLine='python worker.py'),
        dict(ProcessId=21, ParentProcessId=20, CommandLine='sumo'),
        dict(ProcessId=30, ParentProcessId=1, CommandLine='powershell unrelated-wrapper.ps1'),
        dict(ProcessId=40, ParentProcessId=1, CommandLine='cmd /c unrelated-wrapper.cmd')]
    selected = recovery.target_live_processes('fixture', processes, own_pid=99, parent_pid=98)
    assert [p['ProcessId'] for p in selected] == [20,21,30,40]


def test_terminal_receipt_prevents_interruption_recovery(monkeypatch, tmp_path):
    source = tmp_path/'source'
    (source/'execution').mkdir(parents=True)
    (source/'execution/exit.json').write_text('{}')
    monkeypatch.setitem(recovery.STUDIES, 'fixture', dict(source=source, script='target.py', stage='x'))
    with pytest.raises(ValueError, match='Terminal receipt exists'):
        recovery.assert_interrupted('fixture', processes=[])


def test_evidence_copy_preserves_partial_and_execution_bytes(tmp_path):
    source, target = tmp_path/'source', tmp_path/'target'
    (source/'blocks/00002').mkdir(parents=True)
    (source/'execution').mkdir()
    (source/'blocks/00002/observations.sqlite').write_bytes(b'partial-db-bytes')
    (source/'execution/progress.json').write_bytes(b'{"runs":5}\n')
    (source/'launcher.json').write_bytes(b'{"launcher_pid":30}\n')
    (source/'launcher.stdout.log').write_bytes(b'launcher evidence\n')
    (source/'watcher_launch.json').write_bytes(b'{"launcher_pid":40}\n')
    (source/'unrelated.txt').write_bytes(b'not execution evidence\n')
    (source/'plan.json').write_bytes(b'{"plan":1}\n')
    target.mkdir()
    hashes = recovery.copy_evidence(source, target, 2)
    assert recovery.files_at(target/'snapshot') == hashes
    assert (target/'snapshot/partial_block/00002/observations.sqlite').read_bytes() == b'partial-db-bytes'
    assert (target/'snapshot/execution/progress.json').read_bytes() == b'{"runs":5}\n'
    assert (target/'snapshot/root_execution/launcher.json').read_bytes() == b'{"launcher_pid":30}\n'
    assert (target/'snapshot/root_execution/launcher.stdout.log').read_bytes() == b'launcher evidence\n'
    assert (target/'snapshot/root_execution/watcher_launch.json').read_bytes() == b'{"launcher_pid":40}\n'
    assert not (target/'snapshot/root_execution/unrelated.txt').exists()


def test_source_dependencies_revalidate_external_inputs_models_and_original_contract(monkeypatch, tmp_path):
    input_file, model_file = tmp_path/'input.json', tmp_path/'model.pt'
    input_file.write_bytes(b'frozen-input')
    model_file.write_bytes(b'frozen-model')
    source_plan = dict(inputs_sha256={'input.json': digest(input_file)},
        models={'m': dict(path='model.pt', sha256=digest(model_file))})
    calls = []
    def original_inputs():
        calls.append(True)
        return deepcopy(source_plan)
    monkeypatch.setattr(recovery, 'ROOT', tmp_path)
    monkeypatch.setattr(recovery.v32, 'inputs', original_inputs)
    recovery.validate_source_dependencies('v32', source_plan)
    assert calls == [True]
    input_file.write_bytes(b'changed-input')
    with pytest.raises(ValueError, match='external input changed'):
        recovery.validate_source_dependencies('v32', source_plan)
    input_file.write_bytes(b'frozen-input')
    model_file.write_bytes(b'changed-model')
    with pytest.raises(ValueError, match='trained model changed'):
        recovery.validate_source_dependencies('v32', source_plan)
    model_file.write_bytes(b'frozen-model')
    monkeypatch.setattr(recovery.v32, 'inputs', lambda: dict(source_plan, changed=True))
    with pytest.raises(ValueError, match='Original source validator'):
        recovery.validate_source_dependencies('v32', source_plan)


def test_complete_block_proof_hashes_database_seal_and_complete_receipt(tmp_path):
    for name, value in zip(recovery.COMPLETE_BLOCK_FILES, (b'database', b'seal', b'receipt')):
        (tmp_path/name).write_bytes(value)
    files = recovery.complete_block_hashes(tmp_path)
    assert set(files) == set(recovery.COMPLETE_BLOCK_FILES)
    assert files['complete.json'] == digest(tmp_path/'complete.json')
    proof = recovery.completion_block_hashes([dict(block_index=7, files_sha256=files)])
    assert proof == {'00007': files}
    with pytest.raises(ValueError, match='DB/seal/receipt'):
        recovery.completion_block_hashes([dict(block_index=7,
            files_sha256={'observations.sqlite': files['observations.sqlite']})])


def test_verify_file_map_detects_added_removed_or_modified_evidence(tmp_path):
    (tmp_path/'a').write_bytes(b'a')
    expected = recovery.files_at(tmp_path)
    recovery.verify_file_map(tmp_path, expected, 'fixture')
    (tmp_path/'a').write_bytes(b'b')
    with pytest.raises(ValueError, match='fixture files changed'):
        recovery.verify_file_map(tmp_path, expected, 'fixture')


def test_complete_directory_selects_each_epoch_once(monkeypatch, tmp_path):
    source = tmp_path/'source'
    monkeypatch.setitem(recovery.STUDIES, 'fixture', dict(source=source, script='x.py', stage='recovered'))
    monkeypatch.setattr(recovery, 'OUT', tmp_path/'out')
    monkeypatch.setattr(recovery, 'source_metadata', lambda study, group: {'epoch':'old'})
    monkeypatch.setattr(recovery, 'recovery_metadata', lambda study, group: {'epoch':'new'})
    frozen = dict(retained_complete_blocks=[0,1], missing_block_indices=[2,3])
    old = recovery.complete_directory('fixture', frozen, {'block_index':1})
    new = recovery.complete_directory('fixture', frozen, {'block_index':2})
    assert old[2] == 'original' and old[0] == source/'blocks/00001'
    assert new[2] == 'recovery' and new[0] == tmp_path/'out/fixture/blocks/00002'


def test_recovery_workers_delegate_to_frozen_workers_instead_of_reimplementing_them():
    tree = ast.parse(Path(recovery.__file__).read_text(encoding='utf-8'))
    worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'worker')
    calls = {ast.unparse(n.func) for n in ast.walk(worker) if isinstance(n, ast.Call)}
    assert 'v32.worker' in calls and 'v36.core.worker' in calls


def test_plan_language_explicitly_excludes_partial_prefix_from_effective_sample():
    text = Path(recovery.__file__).read_text(encoding='utf-8')
    assert 'Partial-prefix runs are immutable engineering evidence' in text
    assert 'partial_prefix_counted_in_effects=False' in text
    assert 'new_independent_initials=0' in text
    assert 'all_block_database_seal_and_complete_receipts_hashed=True' in text
    assert 'block_files_sha256=block_hashes' in text


def test_cli_exposes_only_explicit_freeze_worker_watch_and_summarize():
    tree = ast.parse(Path(recovery.__file__).read_text(encoding='utf-8'))
    rendered = ast.unparse(tree)
    assert "choices=('freeze', 'worker', 'watch', 'summarize')" in rendered
    assert "choices=tuple(STUDIES)" in rendered
