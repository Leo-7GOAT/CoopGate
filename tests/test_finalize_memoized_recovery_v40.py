import ast
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import finalize_memoized_recovery_v40 as finalizer


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def identity_tree(tmp_path,monkeypatch):
    monkeypatch.setattr(finalizer,'committed',lambda _:None)
    failed = tmp_path/'failed'
    exit_receipt = dict(exit_code=1,utc='2026-10-24T04:37:25+00:00')
    save(failed/'execution/exit.json',exit_receipt)
    save(failed/'failed_attempt_verification.json',dict(status='failed_before_simulation',
        process_identity=dict(outer_launcher_pid=1,watcher_pid=2,
            worker_launcher_pid=3,native_worker_pid=4),
        execution_evidence_sha256={
            'execution/exit.json':finalizer.digest(failed/'execution/exit.json')}))

    reuse = tmp_path/'reuse'
    save(reuse/'plan.json',dict(study='v32'))
    plan_sha = finalizer.digest(reuse/'plan.json')
    save(reuse/'watcher_launch.json',dict(launcher_pid=3,plan_sha256=plan_sha,
        command='python -X utf8 tools/recover_block_grids_v38.py watch v32'))
    save(reuse/'execution/spawn.json',dict(watcher_pid=5,launcher_pid=6,
        plan_sha256=plan_sha,command=['python','-X','utf8',
            'tools/recover_block_grids_v38.py','worker','v32']))
    save(reuse/'execution/launch.json',dict(watcher_pid=5,launcher_pid=6,
        native_pid=7,parent_pid=6,plan_sha256=plan_sha,
        native_creation_date='/Date(1792818316400)/',
        command_line='python tools/recover_block_grids_v38.py worker v32'))
    save(reuse/'execution/worker.native.json',dict(native_pid=7,parent_pid=6))
    save(reuse/'execution/worker.acknowledged.json',dict(native_pid=7,parent_pid=6,
        launcher_pid=6,native_start_sha256=finalizer.digest(
            reuse/'execution/worker.native.json')))
    processes = [
        dict(ProcessId=3,ParentProcessId=30,CreationDate='/Date(1792818309000)/',
             CommandLine='python tools/recover_block_grids_v38.py watch v32'),
        dict(ProcessId=5,ParentProcessId=3,CreationDate='/Date(1792818309100)/',
             CommandLine='python tools/recover_block_grids_v38.py watch v32'),
        dict(ProcessId=6,ParentProcessId=5,CreationDate='/Date(1792818316200)/',
             CommandLine='python tools/recover_block_grids_v38.py worker v32'),
        dict(ProcessId=7,ParentProcessId=6,CreationDate='/Date(1792818316400)/',
             CommandLine='python tools/recover_block_grids_v38.py worker v32'),
        dict(ProcessId=8,ParentProcessId=7,CreationDate='/Date(1792818316500)/',
             CommandLine='sumo -c generated.sumocfg'),
        dict(ProcessId=9,ParentProcessId=90,CreationDate='/Date(1792818000000)/',
             CommandLine='python unrelated.py'),
    ]
    return failed,reuse,processes


def test_only_later_receipt_bound_foreign_tree_is_filtered(tmp_path,monkeypatch):
    failed,reuse,processes = identity_tree(tmp_path,monkeypatch)
    filtered,audit = finalizer.filter_proven_pid_reuse(processes,
        failed_directory=failed,reuse_directory=reuse,require_committed=False)
    assert [item['ProcessId'] for item in filtered] == [9]
    assert audit['mode'] == 'proven_pid_reuse'
    assert audit['collision_pid'] == 3
    assert audit['excluded_pids'] == [3,5,6,7,8]
    assert audit['foreign_root_created_after_failed_exit']
    assert [item['ProcessId'] for item in audit['excluded_process_identities']] == [3,5,6,7,8]
    assert audit['foreign_root_creation_millis'] > finalizer.creation_millis(
        audit['failed_exit_utc'])


@pytest.mark.parametrize('change',('old_command','old_creation','broken_parent',
                                    'different_collision','changed_plan'))
def test_ambiguous_or_real_old_process_is_never_skipped(tmp_path,monkeypatch,change):
    failed,reuse,processes = identity_tree(tmp_path,monkeypatch)
    if change == 'old_command':
        processes[0]['CommandLine'] = 'python tools/recover_memoized_recovery_v38.py worker'
    elif change == 'old_creation':
        processes[0]['CreationDate'] = '/Date(1792810000000)/'
    elif change == 'broken_parent':
        processes[2]['ParentProcessId'] = 99
    elif change == 'different_collision':
        processes.append(dict(ProcessId=2,ParentProcessId=99,
            CreationDate='/Date(1792818317000)/',CommandLine='python unrelated.py'))
    else:
        receipt = json.loads((reuse/'execution/spawn.json').read_text())
        receipt['plan_sha256'] = 'changed'
        save(reuse/'execution/spawn.json',receipt)
    with pytest.raises(ValueError):
        finalizer.filter_proven_pid_reuse(processes,failed_directory=failed,
            reuse_directory=reuse,require_committed=False)


def test_orphan_old_descendant_remains_visible_without_proof(tmp_path,monkeypatch):
    failed,reuse,_ = identity_tree(tmp_path,monkeypatch)
    orphan = [dict(ProcessId=99,ParentProcessId=3,
        CreationDate='/Date(1792818317000)/',CommandLine='sumo old-tree')]
    filtered,audit = finalizer.filter_proven_pid_reuse(orphan,
        failed_directory=failed,reuse_directory=reuse,require_committed=False)
    assert filtered == orphan
    assert audit['mode'] == 'no_live_collision'
    assert finalizer.descendants(filtered,{3}) == {99}


def test_analysis_router_never_redirects_v39_inputs(tmp_path):
    source,target = tmp_path/'v39',tmp_path/'v40'
    route = finalizer.AnalysisRouter(source,target)
    assert route/'plan.json' == source/'plan.json'
    assert route/'execution/exit.json' == source/'execution/exit.json'
    assert route/'blocks/00074' == source/'blocks/00074'
    assert route/'analysis' == target/'analysis'
    assert route/'analysis/report.json' == target/'analysis/report.json'


def test_finalizer_has_no_collection_route_and_binds_every_proof_layer():
    tree = ast.parse(Path(finalizer.__file__).read_text(encoding='utf-8'))
    choices = [node for node in ast.walk(tree) if isinstance(node,ast.Tuple)
               and all(isinstance(item,ast.Constant) for item in node.elts)
               and {item.value for item in node.elts} == {'freeze','finalize'}]
    assert choices
    assert finalizer.BLOCK_FILES == (
        'observations.sqlite','observations.sqlite.seal.json','complete.json')
    assert set(finalizer.TERMINAL_FILES) == {
        'watcher_launch.json','execution/spawn.json','execution/launch.json',
        'execution/worker.native.json','execution/worker.acknowledged.json',
        'execution/progress.json','execution/collection_complete.json',
        'execution/exit.json','execution/worker.stdout.log','execution/worker.stderr.log',
        'recovery_watcher.stdout.log','recovery_watcher.stderr.log'}
    text = Path(finalizer.__file__).read_text(encoding='utf-8')
    assert 'module.summarize()' in text
    assert "OUT/'finalization_receipt.json'" in text
    assert 'new_sumo_runs=0' in text
    assert 'finalizer_os_exit_receipt_present=False' in text
    assert 'module.worker' not in text and 'module.watch' not in text
