"""Pure-analysis V40 finalizer for the completed V39 memoized recovery.

V39 collected its complete frozen scope and exited successfully.  Its original
summarizer is conservatively blocked because one PID recorded by the earlier
failed V38 attempt has since been reused by a different, frozen V32 recovery.
This tool does not weaken the old-process guard generally.  It removes only a
currently visible process tree whose later creation identity, command lines,
committed foreign plan and startup receipts prove that exact PID reuse.  Every
unproved old root or descendant remains visible to V39's unchanged guard.

The finalizer has no worker or watcher command.  It freezes hashes of the V39
plan, terminal receipts, all 106 recovered blocks and the V38 failure proof,
then routes only V39's analysis writes into a fresh V40 directory while calling
the original V39-configured summarize function.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re

import audit_buffered_scope_v25 as infrastructure
from execute_prospective_v23 import (check_acknowledgment, exclusive_json,
    native_snapshot, runtime_identity, sources_committed, utc)
from profile_phase_runtime_v24 import committed, digest
import recover_memoized_recovery_v38 as core
import recover_memoized_recovery_v39 as retry


ROOT = Path(__file__).resolve().parents[1]
V39 = ROOT/'artifacts/v39/memoized_recovery_recovery_20261024'
FAILED_V38 = ROOT/'artifacts/v38/memoized_recovery_recovery_20261024'
REUSED_BY = ROOT/'artifacts/v38/interrupted_block_grid_recovery/v32'
OUT = ROOT/'artifacts/v40/memoized_recovery_finalization_20261024'
SCRIPT = 'tools/finalize_memoized_recovery_v40.py'
TEST = 'tests/test_finalize_memoized_recovery_v40.py'
SOURCES = (SCRIPT, TEST)
BLOCK_FILES = ('observations.sqlite','observations.sqlite.seal.json','complete.json')
TERMINAL_FILES = ('watcher_launch.json','execution/spawn.json','execution/launch.json',
    'execution/worker.native.json','execution/worker.acknowledged.json',
    'execution/progress.json','execution/collection_complete.json','execution/exit.json',
    'execution/worker.stdout.log','execution/worker.stderr.log',
    'recovery_watcher.stdout.log','recovery_watcher.stderr.log')
REUSE_RECEIPTS = ('watcher_launch.json','execution/spawn.json','execution/launch.json',
    'execution/worker.native.json','execution/worker.acknowledged.json')
REUSE_SCRIPT = 'tools/recover_block_grids_v38.py'
REUSE_SOURCES = (REUSE_SCRIPT,'tests/test_recover_block_grids_v38.py')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def normalized_command(value):
    return ' '.join(str(value or '').replace('\\','/').lower().split())


def creation_millis(value):
    """Normalize PowerShell /Date(ms)/ and ISO CIM creation timestamps."""
    match = re.fullmatch(r'/Date\((\d+)(?:[+-]\d+)?\)/', str(value or ''))
    if match:
        return int(match.group(1))
    text = str(value or '').replace('Z','+00:00')
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise ValueError('A parseable native process creation time is required') from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return round(parsed.timestamp()*1000)


def relative(path):
    path = Path(path)
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def descendants(processes, roots):
    result, frontier = set(), set(roots)
    while frontier:
        children = {int(item['ProcessId']) for item in processes
                    if item.get('ProcessId') is not None
                    and item.get('ParentProcessId') is not None
                    and int(item['ParentProcessId']) in frontier}
        children -= result
        children -= set(roots)
        if not children:
            break
        result.update(children)
        frontier = children
    return result


def reuse_receipt_identity(directory=REUSED_BY, *, require_committed=True):
    """Bind the foreign frozen plan and its immutable startup identity chain."""
    directory = Path(directory)
    plan_path = directory/'plan.json'
    if require_committed:
        committed(plan_path)
    plan_sha = digest(plan_path)
    plan = read(plan_path)
    if plan.get('study') != 'v32':
        raise ValueError('Only the separately frozen V32 recovery may explain this reuse')
    if require_committed:
        for name in REUSE_SOURCES:
            if plan.get('source_sha256',{}).get(name) != digest(ROOT/name):
                raise ValueError('A source of the foreign frozen recovery changed: '+name)
        sources_committed([ROOT/name for name in REUSE_SOURCES])
    watcher = read(directory/'watcher_launch.json')
    spawn = read(directory/'execution/spawn.json')
    launch = read(directory/'execution/launch.json')
    native = read(directory/'execution/worker.native.json')
    acknowledgment = read(directory/'execution/worker.acknowledged.json')
    watch_fragment = normalized_command(f'{REUSE_SCRIPT} watch v32')
    worker_fragment = normalized_command(f'{REUSE_SCRIPT} worker v32')
    if (watcher.get('plan_sha256') != plan_sha
            or spawn.get('plan_sha256') != plan_sha
            or launch.get('plan_sha256') != plan_sha
            or spawn.get('watcher_pid') != launch.get('watcher_pid')
            or spawn.get('launcher_pid') != launch.get('launcher_pid')
            or native.get('native_pid') != launch.get('native_pid')
            or native.get('parent_pid') != launch.get('parent_pid')
            or launch.get('parent_pid') != launch.get('launcher_pid')
            or acknowledgment.get('native_pid') != launch.get('native_pid')
            or acknowledgment.get('parent_pid') != launch.get('parent_pid')
            or acknowledgment.get('launcher_pid') != launch.get('launcher_pid')
            or acknowledgment.get('native_start_sha256') != digest(
                directory/'execution/worker.native.json')
            or watch_fragment not in normalized_command(watcher.get('command'))
            or worker_fragment not in normalized_command(' '.join(spawn.get('command',())))
            or worker_fragment not in normalized_command(launch.get('command_line'))):
        raise ValueError('The foreign frozen study startup chain is inconsistent')
    return dict(path=relative(directory),plan_sha256=plan_sha,
        receipt_sha256={name:digest(directory/name) for name in REUSE_RECEIPTS},
        collision_root_pid=watcher['launcher_pid'],watcher_pid=spawn['watcher_pid'],
        worker_launcher_pid=spawn['launcher_pid'],native_worker_pid=launch['native_pid'],
        native_creation_date=launch['native_creation_date'],
        native_command_line=launch['command_line'],watch_command_fragment=watch_fragment,
        worker_command_fragment=worker_fragment)


def _failed_roots(directory):
    directory = Path(directory)
    verification = directory/'failed_attempt_verification.json'
    committed(verification)
    proof = read(verification)
    exit_path = directory/'execution/exit.json'
    expected_exit = proof.get('execution_evidence_sha256',{}).get('execution/exit.json')
    exit_receipt = read(exit_path)
    if (proof.get('status') != 'failed_before_simulation' or expected_exit != digest(exit_path)
            or exit_receipt.get('exit_code') != 1):
        raise ValueError('The V38 terminal failure proof is not intact')
    identities = proof.get('process_identity',{})
    names = ('outer_launcher_pid','watcher_pid','worker_launcher_pid','native_worker_pid')
    roots = tuple(identities.get(name) for name in names)
    if any(type(value) is not int or value <= 0 for value in roots):
        raise ValueError('The V38 failed-attempt roots are incomplete')
    return roots, exit_receipt, digest(verification), digest(exit_path)


def filter_proven_pid_reuse(processes, *, failed_directory=FAILED_V38,
                            reuse_directory=REUSED_BY, require_committed=True):
    """Remove only the later, receipt-bound foreign tree sharing an old PID.

    No collision means no filtering.  A collision with any root other than the
    proven foreign launcher, an incomplete live foreign chain, an old command,
    or a creation time not strictly after V38's terminal exit is fatal.
    """
    processes = [dict(item) for item in processes]
    for item in processes:
        if (type(item.get('ProcessId')) is not int
                or type(item.get('ParentProcessId')) is not int
                or not isinstance(item.get('CommandLine'),str)
                or item.get('CreationDate') is None):
            raise ValueError('Complete live process identities are required for PID-reuse proof')
    roots, failed_exit, verification_sha, exit_sha = _failed_roots(failed_directory)
    reuse = reuse_receipt_identity(reuse_directory,require_committed=require_committed)
    by_pid = {item['ProcessId']:item for item in processes}
    if len(by_pid) != len(processes):
        raise ValueError('Duplicate live PID identity')
    collisions = set(roots) & set(by_pid)
    if not collisions:
        return processes,dict(mode='no_live_collision',excluded_pids=[],
            failed_verification_sha256=verification_sha,failed_exit_sha256=exit_sha,
            reuse_identity=reuse)
    foreign_root = reuse['collision_root_pid']
    if collisions != {foreign_root}:
        raise ValueError('A recorded V38 root is live without an exact foreign-study proof')
    role_pids = (foreign_root,reuse['watcher_pid'],reuse['worker_launcher_pid'],
                 reuse['native_worker_pid'])
    if any(pid not in by_pid for pid in role_pids):
        raise ValueError('The claimed foreign PID-reuse tree is not fully live and observable')
    root, watcher, launcher, native = (by_pid[pid] for pid in role_pids)
    root_creation = creation_millis(root['CreationDate'])
    if (root_creation <= creation_millis(failed_exit['utc'])
            or reuse['watch_command_fragment'] not in normalized_command(root['CommandLine'])
            or watcher['ParentProcessId'] != foreign_root
            or reuse['watch_command_fragment'] not in normalized_command(watcher['CommandLine'])
            or launcher['ParentProcessId'] != reuse['watcher_pid']
            or reuse['worker_command_fragment'] not in normalized_command(launcher['CommandLine'])
            or native['ParentProcessId'] != reuse['worker_launcher_pid']
            or creation_millis(native['CreationDate']) != creation_millis(
                reuse['native_creation_date'])
            or normalized_command(native['CommandLine']) != normalized_command(
                reuse['native_command_line'])):
        raise ValueError('The colliding PID is not proven to be the later frozen V32 recovery')
    tree = {foreign_root,*descendants(processes,{foreign_root})}
    if not set(role_pids) <= tree:
        raise ValueError('The foreign startup receipts do not describe one live process tree')
    for pid in tree:
        item = by_pid[pid]
        command = normalized_command(item['CommandLine'])
        if (creation_millis(item['CreationDate']) < root_creation
                or 'tools/recover_memoized_recovery_v38.py' in command
                or 'tools/recover_memoized_recovery_v39.py' in command):
            raise ValueError('A visible descendant cannot be attributed to the later foreign tree')
    filtered = [item for item in processes if item['ProcessId'] not in tree]
    observed = [{name:item.get(name) for name in
                 ('ProcessId','ParentProcessId','Name','CreationDate','CommandLine')}
                for item in sorted((by_pid[pid] for pid in tree),
                                   key=lambda value:value['ProcessId'])]
    return filtered,dict(mode='proven_pid_reuse',collision_pid=foreign_root,
        excluded_pids=sorted(tree),foreign_root_created_after_failed_exit=True,
        old_root_pids=list(roots),failed_exit_utc=failed_exit['utc'],
        foreign_root_creation_millis=root_creation,
        excluded_process_identities=observed,
        failed_verification_sha256=verification_sha,failed_exit_sha256=exit_sha,
        reuse_identity=reuse)


@contextmanager
def configured_safe_core(processes=None, *, route=None):
    """Configure V39 and replace only its failed-attempt liveness observation."""
    saved = {name:getattr(core,name) for name in
             ('OUT','SCRIPT','TEST','SOURCES','FAILED_ATTEMPT','RECOVERY_SEGMENT')}
    original_failed = core.failed_attempt_identity
    module = retry.configure()
    processes_were_supplied = processes is not None
    snapshot = native_snapshot() if processes is None else [dict(item) for item in processes]
    filtered,audit = filter_proven_pid_reuse(snapshot)

    def guarded(directory=None,processes=None):
        if processes is None and processes_were_supplied:
            visible = filtered
        else:
            current = native_snapshot() if processes is None else processes
            visible,_ = filter_proven_pid_reuse(current)
        return original_failed(directory,processes=visible)

    module.failed_attempt_identity = guarded
    if route is not None:
        module.OUT = route
    try:
        yield module,audit
    finally:
        module.failed_attempt_identity = original_failed
        for name,value in saved.items():
            setattr(module,name,value)


def terminal_manifest(processes=None):
    """Validate complete V39 identities without decoding episode outcomes."""
    with configured_safe_core(processes) as (module,audit):
        plan = module.inputs()
        plan_sha = digest(V39/'plan.json')
        execution = V39/'execution'
        receipt,launch,done = [read(execution/name) for name in
                               ('exit.json','launch.json','collection_complete.json')]
        infrastructure.validate_completion(plan,plan_sha,receipt,launch,done)
        check_acknowledgment(execution,'worker',launch['launcher_pid'],launch['native_pid'])
        progress = read(execution/'progress.json')
        if (progress.get('completed_runs') != plan['actual_sumo_runs']
                or progress.get('expected_runs') != plan['actual_sumo_runs']
                or progress.get('completed_blocks') != plan['blocks']
                or progress.get('expected_blocks') != plan['blocks']
                or progress.get('aggregate_completed_blocks') != plan['aggregate_blocks']
                or progress.get('aggregate_expected_blocks') != plan['aggregate_blocks']):
            raise ValueError('V39 final durable progress is not the complete frozen scope')
        expected_names = [f"{group['block_index']:05d}" for group in plan['missing_groups']]
        actual_names = sorted(path.name for path in (V39/'blocks').iterdir() if path.is_dir())
        if actual_names != expected_names or len(expected_names) != 106:
            raise ValueError('V39 must contain exactly the 106 frozen recovery blocks')
        blocks = {}
        for group in plan['missing_groups']:
            name = f"{group['block_index']:05d}"
            directory = V39/'blocks'/name
            identity = module.block_identity(directory,module.recovery_metadata(group),
                                             group['executions'],complete=True)
            if set(identity['files']) != set(BLOCK_FILES):
                raise ValueError('A V39 block contains an unexpected or missing file')
            blocks[name] = {item:identity['files'][item] for item in BLOCK_FILES}
        if module.failed_attempt_identity() != plan['previous_failed_attempt']:
            raise ValueError('The V38 failure proof no longer matches the V39 plan')
        receipts = {name:digest(V39/name) for name in TERMINAL_FILES}
        return dict(v39_plan_sha256=plan_sha,v39_terminal_receipts_sha256=receipts,
            v39_recovered_block_files_sha256=blocks,recovered_blocks=106,
            recovered_actual_sumo_runs=212,aggregate_blocks=180,
            aggregate_actual_sumo_runs=360,v38_failed_attempt=plan['previous_failed_attempt'],
            reuse_identity=audit['reuse_identity']),audit


def freeze():
    if OUT.exists():
        raise FileExistsError('Preserve the original V40 finalization attempt')
    source_sha = {name:digest(ROOT/name) for name in SOURCES}
    sources_committed([ROOT/name for name in SOURCES])
    manifest,audit = terminal_manifest()
    OUT.mkdir(parents=True,exist_ok=False)
    exclusive_json(OUT/'plan.json',dict(schema='coopgate.memoized_recovery_finalizer.v40',
        utc=utc(),runtime=runtime_identity(),source_sha256=source_sha,
        source=relative(V39),analysis_output=relative(OUT/'analysis'),
        pure_analysis=True,new_sumo_runs=0,original_v39_plan_unchanged=True,
        original_v39_summarize_mechanics=True,manifest=manifest,
        pid_reuse_freeze_observation=audit,
        rule=('Exclude only a later-created live tree whose committed foreign plan, '
              'startup receipts, commands and creation identities prove PID reuse; '
              'leave every unproved V38 root or descendant visible to the original guard.')))
    print(json.dumps(dict(plan_sha256=digest(OUT/'plan.json'),blocks=106,
        actual_sumo_runs=212,new_sumo_runs=0)),flush=True)


def inputs():
    committed(OUT/'plan.json')
    plan = read(OUT/'plan.json')
    if (plan.get('schema') != 'coopgate.memoized_recovery_finalizer.v40'
            or plan.get('runtime') != runtime_identity()
            or plan.get('pure_analysis') is not True
            or plan.get('new_sumo_runs') != 0):
        raise ValueError('The V40 pure-analysis protocol changed')
    for name,expected in plan.get('source_sha256',{}).items():
        if digest(ROOT/name) != expected:
            raise ValueError('A V40 finalizer source changed: '+name)
    if set(plan.get('source_sha256',{})) != set(SOURCES):
        raise ValueError('The V40 finalizer source family is incomplete')
    sources_committed([ROOT/name for name in SOURCES])
    manifest,audit = terminal_manifest()
    if manifest != plan.get('manifest'):
        raise ValueError('V39 terminal evidence changed after V40 freeze')
    return plan,audit


class AnalysisRouter:
    """Read every source path from V39 and route only analysis writes to V40."""
    def __init__(self,source=V39,target=OUT):
        self.source,self.target = Path(source),Path(target)

    def __truediv__(self,name):
        path = Path(name)
        base = self.target if path.parts and path.parts[0] == 'analysis' else self.source
        return base.joinpath(*path.parts)


def finalize():
    plan,_ = inputs()
    if (V39/'analysis').exists():
        raise FileExistsError('Do not replace or reinterpret an existing V39 analysis')
    if (OUT/'analysis').exists() or (OUT/'finalization_receipt.json').exists():
        raise FileExistsError('Preserve the original V40 finalization output')
    route = AnalysisRouter()
    with configured_safe_core(route=route) as (module,audit):
        module.summarize()
    current,_ = inputs()
    if current != plan:
        raise ValueError('V40 or V39 evidence changed during finalization')
    analysis = OUT/'analysis'
    output_hashes = {name:digest(analysis/name) for name in
                     ('report.json','completion_verification.json','sha256.json')}
    completion = read(analysis/'completion_verification.json')
    if (completion.get('plan_sha256') != plan['manifest']['v39_plan_sha256']
            or completion.get('exit_code') != 0
            or completion.get('recovered_blocks') != 106
            or completion.get('aggregate_blocks') != 180
            or completion.get('actual_sumo_runs') != 360):
        raise ValueError('Original V39 summarize mechanics did not prove the full scope')
    exclusive_json(OUT/'finalization_receipt.json',dict(
        schema='coopgate.memoized_recovery_finalization.v40',utc=utc(),
        finalizer_pid=os.getpid(),v40_plan_sha256=digest(OUT/'plan.json'),
        v39_plan_sha256=plan['manifest']['v39_plan_sha256'],
        v39_terminal_receipts_sha256=plan['manifest']['v39_terminal_receipts_sha256'],
        v39_recovered_block_files_sha256=plan['manifest']['v39_recovered_block_files_sha256'],
        v38_failed_attempt=plan['manifest']['v38_failed_attempt'],
        pid_reuse_observation=audit,analysis_sha256=output_hashes,
        original_v39_summarize_mechanics=True,pure_analysis=True,new_sumo_runs=0,
        v39_worker_exit_code=0,summarize_returned_without_exception=True,
        finalizer_os_exit_receipt_present=False,full_180_pair_coverage=True))
    print(json.dumps(dict(finalized=True,pairs=180,new_sumo_runs=0,
        receipt_sha256=digest(OUT/'finalization_receipt.json'))),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('freeze','finalize'))
    globals()[parser.parse_args().command]()
