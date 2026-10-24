"""Explicit additive recovery for interrupted V32 and V36 sealed-block grids.

The interrupted directories are immutable evidence.  A freeze records every
complete block plus the one partial next block, copies the partial bytes and
execution evidence, and fixes the missing original block indices.  Recovery
reruns the entire partial block and every later block in a separate directory
through the unchanged frozen worker functions.  Partial-prefix executions are
engineering evidence only and never contribute an additional statistical N.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.episode_store_v21 import inventory, observations
from coopgate.physical_comparison_analysis_v27 import equal_fields
from execute_prospective_v23 import (acknowledge_child, check_acknowledgment,
    exclusive_json, native_snapshot, runtime_identity, sources_committed, utc)
from profile_phase_runtime_v24 import committed, digest

import audit_buffered_scope_v25 as infrastructure
import extend_online_actuator_v32 as v32
import extend_recovery_gate_v36 as v36


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/v38/interrupted_block_grid_recovery'
SCRIPT = 'tools/recover_block_grids_v38.py'
TEST = 'tests/test_recover_block_grids_v38.py'
STUDIES = {
    'v32': dict(source=v32.OUT, script=Path(v32.SCRIPT).name,
                stage='full_online_grid_v38_recovery'),
    'v36': dict(source=v36.OUT, script=Path(v36.SCRIPT).name,
                stage='full_recovery_gate_grid_v38_recovery'),
}
TERMINAL_RECEIPTS = ('execution/exit.json', 'execution/collection_complete.json')
HANDLE_RECEIPTS = ('execution/launch.json', 'execution/spawn.json',
                   'launcher.json', 'watcher_launch.json')
ROOT_EXECUTION_EVIDENCE_PREFIXES = ('launcher', 'watcher', 'launch_verification')
COMPLETE_BLOCK_FILES = ('observations.sqlite', 'observations.sqlite.seal.json', 'complete.json')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def files_at(directory):
    directory = Path(directory)
    return {p.relative_to(directory).as_posix(): digest(p)
            for p in sorted(directory.rglob('*')) if p.is_file()}


def exact_prefix(expected, observed):
    expected, observed = list(map(list, expected)), list(map(list, observed))
    if len(observed) >= len(expected) or observed != expected[:len(observed)]:
        raise ValueError('Interrupted block is not a strict original-schedule prefix')
    return len(observed)


def known_handle_pids(source):
    result = set()
    for relative in HANDLE_RECEIPTS:
        path = source/relative
        if not path.exists():
            continue
        value = read(path)
        for name in ('native_pid', 'parent_pid', 'launcher_pid', 'watcher_pid'):
            if type(value.get(name)) is int:
                result.add(value[name])
    return result


def target_live_processes(study, processes, own_pid=None, parent_pid=None):
    """Return only processes tied to the interrupted attempt, including children."""
    if study not in STUDIES:
        raise ValueError('Unknown interrupted block grid')
    own_pid = os.getpid() if own_pid is None else own_pid
    parent_pid = os.getppid() if parent_pid is None else parent_pid
    source, script = STUDIES[study]['source'], STUDIES[study]['script'].lower()
    handles = known_handle_pids(source)
    selected = {p['ProcessId'] for p in processes
                if p['ProcessId'] not in (own_pid, parent_pid)
                and (p['ProcessId'] in handles or script in (p.get('CommandLine') or '').lower())}
    while True:
        children = {p['ProcessId'] for p in processes if p['ParentProcessId'] in selected}
        if children <= selected:
            break
        selected |= children
    return [p for p in processes if p['ProcessId'] in selected]


def assert_interrupted(study, processes=None):
    config = STUDIES[study]
    source = config['source']
    present = [name for name in TERMINAL_RECEIPTS if (source/name).exists()]
    if present:
        raise ValueError('Terminal receipt exists; inspect completion instead of recovering: '+','.join(present))
    processes = native_snapshot() if processes is None else processes
    live = target_live_processes(study, processes)
    if live:
        raise ValueError('An original target handle or descendant is still live')
    return dict(observed_utc=utc(), original_target_handles_absent=True,
                original_terminal_receipts_absent=True,
                known_original_handle_pids=sorted(known_handle_pids(source)))


def source_metadata(study, group):
    if study == 'v32':
        return v32.metadata(group)
    if study == 'v36':
        return v36.metadata(group)
    raise ValueError('Unknown interrupted block grid')


def validate_complete_receipt(study, group, directory):
    receipt = read(directory/'complete.json')
    if (receipt.get('database_sha256') != digest(directory/'observations.sqlite')
            or receipt.get('seal_sha256') != digest(directory/'observations.sqlite.seal.json')
            or receipt.get('actual_sumo_runs') != len(group['cases'])):
        raise ValueError('Complete-block receipt or hashes differ')
    if study == 'v36' and len(receipt.get('reviews', [])) != len(group['cases']):
        raise ValueError('V36 complete block lacks every collection review')
    return receipt


def inspect_interrupted_blocks(study, source_plan):
    """Inspect identities only; never decompress rows or calculate effects."""
    source = STUDIES[study]['source']
    groups = source_plan['groups']
    directories = sorted((source/'blocks').iterdir(), key=lambda p: p.name)
    if (not directories or any(not p.is_dir() or not p.name.isdigit() for p in directories)
            or [int(p.name) for p in directories] != list(range(len(directories)))
            or len(directories) > len(groups)):
        raise ValueError('Original block directories are not one contiguous prefix')
    entries, partial = [], []
    for directory in directories:
        index = int(directory.name)
        group = groups[index]
        database = directory/'observations.sqlite'
        if not database.exists():
            raise ValueError('Interrupted block has no durable episode database')
        state = inventory(database)
        if state['metadata'] != source_metadata(study, group):
            raise ValueError('Original block metadata differs from the frozen plan')
        seal, complete = directory/'observations.sqlite.seal.json', directory/'complete.json'
        markers = seal.exists(), complete.exists()
        if all(markers):
            if state['cases'] != [list(k) for k in group['cases']]:
                raise ValueError('Complete block differs from its frozen schedule')
            seal_value = read(seal)
            if (seal_value.get('database_sha256') != digest(database)
                    or {k: seal_value[k] for k in ('metadata','cases','row_sha256','trace_sha256')} != state):
                raise ValueError('Complete database seal differs from inventory')
            validate_complete_receipt(study, group, directory)
            status = 'complete'
        else:
            if any(markers):
                raise ValueError('Partially sealed block cannot be recovered implicitly')
            records = exact_prefix(group['cases'], state['cases'])
            partial.append(index)
            status = 'partial'
        entries.append(dict(block_index=index, status=status, records=len(state['cases']),
            inventory=state, files_sha256=files_at(directory)))
    if len(partial) != 1 or partial[0] != len(directories)-1:
        raise ValueError('Exactly one final partial next block is required')
    retained = [e['block_index'] for e in entries if e['status'] == 'complete']
    if retained != list(range(partial[0])):
        raise ValueError('Only a complete prefix followed by one partial block is recoverable')
    missing = list(range(partial[0], len(groups)))
    return dict(entries=entries, retained_complete_blocks=retained,
        partial_block_index=partial[0], partial_engineering_runs=entries[-1]['records'],
        missing_block_indices=missing,
        retained_actual_sumo_runs=sum(len(groups[i]['cases']) for i in retained),
        recovery_actual_sumo_runs=sum(len(groups[i]['cases']) for i in missing))


def copy_evidence(source, target, partial_index):
    snapshot = target/'snapshot'
    partial_source = source/'blocks'/f'{partial_index:05d}'
    partial_target = snapshot/'partial_block'/partial_source.name
    execution_source, execution_target = source/'execution', snapshot/'execution'
    partial_target.mkdir(parents=True)
    for item in sorted(partial_source.iterdir()):
        if item.is_file():
            shutil.copy2(item, partial_target/item.name)
    if execution_source.exists():
        execution_target.mkdir(parents=True)
        for item in sorted(execution_source.iterdir()):
            if item.is_file():
                shutil.copy2(item, execution_target/item.name)
    root_execution = [item for item in sorted(source.iterdir()) if item.is_file()
                      and item.name.startswith(ROOT_EXECUTION_EVIDENCE_PREFIXES)]
    if root_execution:
        root_target = snapshot/'root_execution'
        root_target.mkdir(parents=True)
        for item in root_execution:
            shutil.copy2(item, root_target/item.name)
    shutil.copy2(source/'plan.json', snapshot/'source_plan.json')
    return files_at(snapshot)


def validate_source_dependencies(study, source_plan):
    """Revalidate every external input/model byte and the original study contract."""
    inputs_sha256 = source_plan.get('inputs_sha256')
    models = source_plan.get('models')
    if not isinstance(inputs_sha256, dict) or not inputs_sha256:
        raise ValueError('Frozen source plan lacks external input hashes')
    if not isinstance(models, dict) or not models:
        raise ValueError('Frozen source plan lacks trained model hashes')
    for name, expected in inputs_sha256.items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen source external input changed: '+name)
    for label, item in models.items():
        if (not isinstance(item, dict) or not isinstance(item.get('path'), str)
                or not isinstance(item.get('sha256'), str)
                or digest(ROOT/item['path']) != item['sha256']):
            raise ValueError('Frozen source trained model changed: '+str(label))
    validated = v32.inputs() if study == 'v32' else v36.inputs()
    if validated != source_plan:
        raise ValueError('Original source validator no longer returns the frozen source plan')


def freeze(study):
    if study not in STUDIES:
        raise ValueError('Unknown interrupted block grid')
    target, source = OUT/study, STUDIES[study]['source']
    if target.exists():
        raise FileExistsError('Preserve the first explicit recovery protocol')
    # The original validator checks every frozen dependency, model and runtime.
    source_plan = v32.inputs() if study == 'v32' else v36.inputs()
    liveness = assert_interrupted(study)
    prefix = inspect_interrupted_blocks(study, source_plan)
    sources = dict(source_plan['source_sha256'])
    sources.update({name: digest(ROOT/name) for name in (SCRIPT, TEST)})
    for name, expected in sources.items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen recovery dependency changed: '+name)
    sources_committed([ROOT/name for name in sources])
    source_files = files_at(source)
    target.mkdir(parents=True)
    snapshot = copy_evidence(source, target, prefix['partial_block_index'])
    if files_at(source) != source_files:
        raise ValueError('Interrupted source changed while evidence was copied')
    groups = [deepcopy(source_plan['groups'][i]) for i in prefix['missing_block_indices']]
    plan = dict(utc=utc(), study=study, runtime=source_plan['runtime'], source_sha256=sources,
        source_inputs_sha256=deepcopy(source_plan['inputs_sha256']),
        source_models=deepcopy(source_plan['models']),
        source_directory=source.relative_to(ROOT).as_posix(),
        source_plan_sha256=digest(source/'plan.json'), source_files_sha256=source_files,
        snapshot_sha256=snapshot, source_blocks=source_plan['blocks'], blocks=source_plan['blocks'],
        source_actual_sumo_runs=source_plan['actual_sumo_runs'],
        actual_sumo_runs=prefix['recovery_actual_sumo_runs'], groups=groups,
        retained_complete_blocks=prefix['retained_complete_blocks'],
        missing_block_indices=prefix['missing_block_indices'],
        partial_block_index=prefix['partial_block_index'],
        partial_engineering_runs=prefix['partial_engineering_runs'], block_inventory=prefix['entries'],
        retained_actual_sumo_runs=prefix['retained_actual_sumo_runs'], **liveness,
        execution='Run only the missing original block indices in this additive directory through the unchanged frozen worker function. The entire partial block is rerun.',
        aggregation='Use each retained original complete block exactly once and each recovered missing block exactly once. Require exact schedules, hashes, native zero exit and full independent reviews/replays before reporting.',
        statistical_scope='Partial-prefix runs are immutable engineering evidence, excluded from every efficacy denominator and independent-sample count. Recovery adds no independent initial.',
        new_independent_initials=0, new_risk_certificate=False,
        original_exit_status='unknown', partial_effectiveness_outcomes_read=False,
        original_source_mutated=False, original_thresholds_models_and_schedules_unchanged=True)
    exclusive_json(target/'plan.json', plan)
    print(json.dumps(dict(study=study, retained_blocks=len(plan['retained_complete_blocks']),
        missing_blocks=len(plan['missing_block_indices']), partial_engineering_runs=plan['partial_engineering_runs'],
        recovery_actual_sumo_runs=plan['actual_sumo_runs'], plan_sha256=digest(target/'plan.json'))),flush=True)


def verify_file_map(root, expected, label):
    actual = files_at(root)
    if actual != expected:
        raise ValueError(label+' files changed after the recovery freeze')


def inputs(study):
    if study not in STUDIES:
        raise ValueError('Unknown interrupted block grid')
    target, source = OUT/study, STUDIES[study]['source']
    committed(target/'plan.json')
    plan = read(target/'plan.json')
    if (plan['study'] != study or plan['source_directory'] != source.relative_to(ROOT).as_posix()
            or plan['source_plan_sha256'] != digest(source/'plan.json')
            or plan['runtime'] != runtime_identity()):
        raise ValueError('Recovery identity, source plan or runtime changed')
    for name, expected in plan['source_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen recovery source changed: '+name)
    sources_committed([ROOT/name for name in plan['source_sha256']])
    verify_file_map(source, plan['source_files_sha256'], 'Interrupted source')
    verify_file_map(target/'snapshot', plan['snapshot_sha256'], 'Evidence snapshot')
    assert_interrupted(study)
    source_plan = read(source/'plan.json')
    if (plan['source_inputs_sha256'] != source_plan.get('inputs_sha256')
            or plan['source_models'] != source_plan.get('models')):
        raise ValueError('Frozen source input/model identities differ from the recovery plan')
    validate_source_dependencies(study, source_plan)
    if ([g['block_index'] for g in plan['groups']] != plan['missing_block_indices']
            or [source_plan['groups'][i] for i in plan['missing_block_indices']] != plan['groups']
            or plan['retained_complete_blocks'] + plan['missing_block_indices'] != list(range(source_plan['blocks']))
            or sum(len(g['cases']) for g in plan['groups']) != plan['actual_sumo_runs']
            or plan['source_actual_sumo_runs'] != source_plan['actual_sumo_runs']):
        raise ValueError('Recovery no longer covers the exact missing original suffix')
    return plan, source_plan


def recovery_metadata(study, group):
    return dict(plan_sha256=digest(OUT/study/'plan.json'), block_index=group['block_index'],
                stage=STUDIES[study]['stage'])


def worker_plan(plan, source_plan):
    result = deepcopy(source_plan)
    result.update(groups=deepcopy(plan['groups']), actual_sumo_runs=plan['actual_sumo_runs'],
                  blocks=plan['blocks'])
    return result


def worker(study):
    plan, source_plan = inputs(study)
    adapted = worker_plan(plan, source_plan)
    if study == 'v32':
        old = v32.OUT, v32.inputs, v32.metadata
        try:
            v32.OUT = OUT/study
            v32.inputs = lambda: adapted
            v32.metadata = lambda group: recovery_metadata(study, group)
            return v32.worker()
        finally:
            v32.OUT, v32.inputs, v32.metadata = old
    old = v36.core.OUT, v36.core.inputs, v36.core.metadata
    try:
        v36.core.OUT = OUT/study
        v36.core.inputs = lambda: adapted
        v36.core.metadata = lambda group: recovery_metadata(study, group)
        return v36.core.worker()
    finally:
        v36.core.OUT, v36.core.inputs, v36.core.metadata = old


def watch(study):
    plan, _ = inputs(study)
    target, execution = OUT/study, OUT/study/'execution'
    execution.mkdir(exist_ok=False)
    with (execution/'worker.stdout.log').open('x') as stdout, (execution/'worker.stderr.log').open('x') as stderr:
        command = [sys.executable, '-X', 'utf8', SCRIPT, 'worker', study]
        process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        exclusive_json(execution/'spawn.json', dict(watcher_pid=os.getpid(), launcher_pid=process.pid,
            command=command, plan_sha256=digest(target/'plan.json'), utc=utc()))
        try:
            native = acknowledge_child(process, execution, 'worker')
            exclusive_json(execution/'launch.json', dict(**native, watcher_pid=os.getpid(),
                plan_sha256=digest(target/'plan.json'), utc=utc()))
        except Exception as error:
            exclusive_json(execution/'observation_error.json', dict(type=type(error).__name__,
                message=str(error), launcher_pid=process.pid, utc=utc()))
            code = process.wait()
            exclusive_json(execution/'exit.json', dict(launcher_pid=process.pid, exit_code=code,
                plan_sha256=digest(target/'plan.json'), utc=utc()))
            raise
        code = process.wait()
        exclusive_json(execution/'exit.json', dict(**native, exit_code=code,
            plan_sha256=digest(target/'plan.json'), utc=utc()))
    if code:
        raise SystemExit(code)


def validate_recovery_exit(study, plan):
    target, execution = OUT/study, OUT/study/'execution'
    receipt, launch, done = [read(execution/name) for name in
        ('exit.json','launch.json','collection_complete.json')]
    infrastructure.validate_completion(plan, digest(target/'plan.json'), receipt, launch, done)
    check_acknowledgment(execution, 'worker', launch['launcher_pid'], launch['native_pid'])
    actual = sorted(int(p.name) for p in (target/'blocks').iterdir() if p.is_dir() and p.name.isdigit())
    if actual != plan['missing_block_indices']:
        raise ValueError('Recovered block directories differ from the exact missing suffix')
    return receipt, launch, done


def complete_directory(study, plan, group):
    index = group['block_index']
    if index in plan['retained_complete_blocks']:
        return STUDIES[study]['source']/'blocks'/f'{index:05d}', source_metadata(study, group), 'original'
    if index in plan['missing_block_indices']:
        return OUT/study/'blocks'/f'{index:05d}', recovery_metadata(study, group), 'recovery'
    raise ValueError('Block is neither retained nor scheduled for recovery')


def verify_complete_block(study, group, directory):
    proof = validate_complete_receipt(study, group, directory)
    return proof


def complete_block_hashes(directory):
    return {name: digest(directory/name) for name in COMPLETE_BLOCK_FILES}


def completion_block_hashes(provenance):
    result = {}
    required = set(COMPLETE_BLOCK_FILES)
    for item in provenance:
        key = f"{item['block_index']:05d}"
        files = item.get('files_sha256')
        if key in result or not isinstance(files, dict) or set(files) != required:
            raise ValueError('Completion proof requires one DB/seal/receipt hash set per block')
        result[key] = files
    return result


def summarize_v32(plan, source_plan):
    old, base, cases, provenance = v32.preflight.archived(), v32.baselines(), [], []
    for group in source_plan['groups']:
        directory, metadata, epoch = complete_directory('v32', plan, group)
        verify_complete_block('v32', group, directory)
        count = 0
        for key, record, trace in observations(directory/'observations.sqlite', metadata, group['cases']):
            baseline, previous = base[(*key[:-2],'baseline')], old[key[:-1]][0]
            outcome = v32.verify(record, trace, key, baseline, previous)
            cases.append(dict(case=list(key), **record, **outcome, baseline_team_cost=baseline['team_cost'],
                phase05_outcome=v32.relative_outcome(previous, baseline, previous['learned_action_steps']>0)))
            count += 1
        files = complete_block_hashes(directory)
        provenance.append(dict(block_index=group['block_index'], epoch=epoch,
            actual_sumo_runs=count, database_sha256=files['observations.sqlite'],
            files_sha256=files))
    if len(cases) != source_plan['actual_sumo_runs']:
        raise ValueError('Every original V32 case is required exactly once')
    cells = []
    for scene in source_plan['spec']['scenarios']:
        for plant in source_plan['spec']['plants']:
            for model in v32.MODELS:
                for kind in v32.KINDS:
                    selected = [r for r in cases if r['case'][1] == scene and r['case'][4] == plant
                        and r['case'][5] == 'phase05_'+model and r['case'][-1] == kind]
                    cells.append(dict(scene=scene, plant=plant, model=model, kind=kind,
                                      **v32.summarize_cell(selected)))
    plant_cells = [dict(plant=plant, kind=kind, **v32.summarize_cell([r for r in cases
        if r['case'][4] == plant and r['case'][-1] == kind]))
        for plant in source_plan['spec']['plants'] for kind in v32.KINDS]
    return dict(actual_sumo_runs=len(cases), cells=cells, plant_cells=plant_cells, cases=cases,
        block_provenance=provenance, all_physical_and_input_reviews_pass=True,
        scope=source_plan['scope'], failure_rule=source_plan['failure_rule'])


def summarize_v36(plan, source_plan):
    torch.set_num_threads(1)
    core = v36.core
    models = {s: load_coupled(ROOT/m['path'])[0] for s,m in source_plan['models'].items()}
    old, cases, provenance, total = core.archived_recovery(), [], [], 0
    for group in source_plan['groups']:
        directory, metadata, epoch = complete_directory('v36', plan, group)
        proof = verify_complete_block('v36', group, directory)
        records, count = {}, 0
        for index, (key, record, trace) in enumerate(observations(
                directory/'observations.sqlite', metadata, group['cases'])):
            row = record['row']
            model = None if key[-1] == 'none' else models[key[-1]]
            equal_fields(row, old[tuple(key[:5])][0], core.recovery.PAIR_FIELDS)
            infrastructure.check_calls(row, trace, record['calls'])
            review = dict(case=list(key), **core.review_joint_trace(row, trace, model, replay=True))
            if review != proof['reviews'][index]:
                raise ValueError('Full V36 gate replay differs from collection review')
            if key[-2] == 'baseline':
                core.match_archived_baseline(row, trace, old[tuple(key[:5])])
            records[key[-2:]] = dict(case=list(key), row=row, calls=record['calls'], review=review)
            total += 1
            count += 1
        baseline = records[('baseline','none')]['row']
        for record in records.values():
            equal_fields(record['row'], baseline, core.recovery.PAIR_FIELDS)
            cases.append(dict(**record, baseline=baseline,
                              **core.relative_outcome(record['row'], baseline)))
        files = complete_block_hashes(directory)
        provenance.append(dict(block_index=group['block_index'], epoch=epoch,
            actual_sumo_runs=count, database_sha256=files['observations.sqlite'],
            files_sha256=files))
    if total != source_plan['actual_sumo_runs']:
        raise ValueError('Every original V36 case is required exactly once')
    cells = []
    for rule in core.RULES:
        for model in (('none',) if rule == 'baseline' else (None,*core.MODELS)):
            for scene in (None,*core.recovery.SCENES):
                for plant in (None,*core.recovery.PROFILES):
                    selected = [p for p in cases if p['row']['rule'] == rule
                        and (model is None or p['row']['model'] == model)
                        and (scene is None or p['row']['scenario'] == scene)
                        and (plant is None or p['row']['plant_name'] == plant)]
                    if selected:
                        cells.append(dict(rule=rule, model=model, scene=scene, plant=plant,
                                          **core.cell_summary(selected)))
    return dict(actual_sumo_runs=total, cases=cases, cells=cells, block_provenance=provenance,
        all_physical_and_full_gate_replays_match=True, old_baseline_failure_conditions=10,
        scope=source_plan['statistical_scope'], prior_adapter_scope=source_plan['prior_adapters'])


def summarize(study):
    plan, source_plan = inputs(study)
    receipt, launch, done = validate_recovery_exit(study, plan)
    report = summarize_v32(plan, source_plan) if study == 'v32' else summarize_v36(plan, source_plan)
    report.update(recovery_plan_sha256=digest(OUT/study/'plan.json'),
        source_plan_sha256=plan['source_plan_sha256'], recovery_exit_sha256=digest(OUT/study/'execution/exit.json'),
        retained_original_actual_sumo_runs=plan['retained_actual_sumo_runs'],
        recovered_actual_sumo_runs=plan['actual_sumo_runs'],
        discarded_partial_engineering_runs=plan['partial_engineering_runs'],
        analytic_independent_initials=source_plan.get('old_independent_initials',
            source_plan.get('old_unique_initials', 160)), new_independent_initials=0,
        partial_prefix_counted_in_effects=False, new_risk_certificate=False,
        original_exit_status='unknown', recovery_native_exit_code=receipt['exit_code'])
    target = OUT/study/'analysis'
    target.mkdir(exist_ok=False)
    exclusive_json(target/'report.json', report)
    exclusive_json(target/'sha256.json', dict(report=digest(target/'report.json')))
    block_hashes = completion_block_hashes(report['block_provenance'])
    database_hashes = {name: files['observations.sqlite'] for name, files in block_hashes.items()}
    exclusive_json(OUT/study/'completion_verification.json', dict(study=study,
        plan_sha256=digest(OUT/study/'plan.json'), report_sha256=digest(target/'report.json'),
        exit_sha256=digest(OUT/study/'execution/exit.json'), launch_sha256=digest(OUT/study/'execution/launch.json'),
        collection_complete_sha256=digest(OUT/study/'execution/collection_complete.json'),
        recovery_exit_code=receipt['exit_code'], native_pid=launch['native_pid'],
        exact_full_schedule=True, all_blocks_hashed=True,
        all_block_database_seal_and_complete_receipts_hashed=True,
        independent_reviews_replayed=True, partial_prefix_counted_in_effects=False,
        block_files_sha256=block_hashes, block_database_sha256=database_hashes,
        actual_sumo_runs=report['actual_sumo_runs'], new_independent_initials=0, utc=utc()))
    print(json.dumps(dict(study=study, actual_sumo_runs=report['actual_sumo_runs'],
        recovered_actual_sumo_runs=plan['actual_sumo_runs'], all_reviews_pass=True,
        report_sha256=digest(target/'report.json'))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze','worker','watch','summarize'))
    parser.add_argument('study', choices=tuple(STUDIES))
    args = parser.parse_args()
    globals()[args.command](args.study)
