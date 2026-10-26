"""Immutable inventory of the fourth prospective-calibration interruption.

This utility reads execution identity and storage structure only. It never
loads calibration effects or runs inference. Every source byte is archived in
bounded ZIP parts after the acknowledged V29 handles are proved absent.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile

from coopgate.episode_store_v21 import inventory
import execute_prospective_v23 as original
import recover_prospective_v24 as mechanics


ROOT = original.ROOT
CAL = ROOT/'artifacts/v38/prospective_calibration_recovery_20261024'
OUT = ROOT/'artifacts/recovery_20261026/calibration_interruption'
PART_LIMIT = 60*1024*1024
SCRIPT = 'tools/snapshot_interruption_20261026.py'
RECOVERY_SCRIPT = 'tools/recover_prospective_20261026.py'
TEST = 'tests/test_recovery_20261026.py'
SOURCES = (SCRIPT, RECOVERY_SCRIPT, TEST)
TARGET_COMMANDS = ('tools/recover_prospective_v38.py',
                   'tools/snapshot_interruption_20261026.py',
                   'tools/recover_prospective_20261026.py')
TERMINAL_RECEIPTS = ('supervisor_exit.json', 'completed.json',
                     'collect_exits.json', 'review_exits.json')


def files_at(directory):
    directory = Path(directory)
    return {p.relative_to(directory).as_posix():original.digest(p)
            for p in sorted(directory.rglob('*')) if p.is_file()}


def parts_for(files, limit=PART_LIMIT):
    result, current, size = [], [], 0
    for path in files:
        length = path.stat().st_size
        if length > limit:
            raise ValueError('Single file exceeds declared archive-part budget')
        if current and size+length > limit:
            result.append(current)
            current, size = [], 0
        current.append(path)
        size += length
    if current:
        result.append(current)
    return result


def _pids(value, key=''):
    """Extract only explicitly labelled process identities from receipts."""
    result = set()
    if isinstance(value, dict):
        for name, item in value.items():
            result |= _pids(item, name)
    elif isinstance(value, list):
        if key.endswith('_pids'):
            result |= {item for item in value if type(item) is int and item > 0}
        else:
            for item in value:
                result |= _pids(item, key)
    elif key.endswith('_pid') and type(value) is int and value > 0:
        result.add(value)
    return result


def declared_process_ids():
    execution = CAL/'calibration/execution'
    paths = [CAL/'launcher.json', execution/'state.json', execution/'watcher_launch.json',
             execution/'supervisor.native.json', execution/'supervisor.acknowledged.json',
             execution/'collect_launch.json']
    paths += sorted(execution.glob('collect_*.native.json'))
    paths += sorted(execution.glob('collect_*.acknowledged.json'))
    result = set()
    for path in paths:
        if path.exists():
            result |= _pids(original.read(path))
    return result


def assert_stopped(processes=None, own_pid=None, parent_pid=None):
    processes = original.native_snapshot() if processes is None else list(processes)
    own_pid = os.getpid() if own_pid is None else own_pid
    parent_pid = os.getppid() if parent_pid is None else parent_pid
    excluded = {own_pid, parent_pid}
    known = declared_process_ids()
    live = []
    for process in processes:
        command = (process.get('CommandLine') or '').replace('\\','/').lower()
        if process['ProcessId'] not in excluded and (process['ProcessId'] in known
                or any(name.lower() in command for name in TARGET_COMMANDS)):
            live.append(process)
    if live:
        raise ValueError('A target V29/V38 execution is still live; refuse interruption snapshot')
    return dict(observed_utc=original.utc(), declared_process_ids=sorted(known),
                target_execution_processes_absent=True,
                unrelated_python_sumo_processes=len(processes)-len(live))


def terminal_state():
    execution = CAL/'calibration/execution'
    present = [name for name in TERMINAL_RECEIPTS if (execution/name).exists()]
    if present:
        raise ValueError('Terminal receipt exists; inspect completion instead of declaring an unknown exit')
    state = original.read(execution/'state.json')
    plan = original.read(CAL/'frozen/plan.json')
    plan_sha = original.digest(CAL/'frozen/plan.json')
    recovery_sha = original.digest(CAL/'recovery/plan.json')
    shards = plan['stages']['calibration']['shards']
    launch = original.read(execution/'collect_launch.json')
    watch = original.read(execution/'watcher_launch.json')
    if (state.get('stage') != 'calibration' or state.get('phase') != 'collect'
            or shards != 3 or len(state.get('exit_codes',())) != shards
            or any(code is not None for code in state['exit_codes'])
            or state.get('plan_sha256') != plan_sha
            or state.get('recovery_plan_sha256') != recovery_sha
            or state.get('worker_pids') != launch.get('worker_pids')
            or state.get('worker_native_pids') != launch.get('worker_native_pids')
            or len(state['worker_pids']) != shards or len(set(state['worker_pids'])) != shards
            or len(state['worker_native_pids']) != shards
            or len(set(state['worker_native_pids'])) != shards
            or launch.get('stage') != 'calibration' or launch.get('plan_sha256') != plan_sha
            or launch.get('recovery_plan_sha256') != recovery_sha
            or launch.get('supervisor_pid') != watch.get('supervisor_native_pid')
            or watch.get('stage') != 'calibration' or watch.get('plan_sha256') != plan_sha
            or watch.get('recovery_plan_sha256') != recovery_sha
            or len(launch.get('native_identities',())) != shards):
        raise ValueError('Last state is not the observed unterminated collection state')
    for shard,(launcher,native,identity) in enumerate(zip(
            state['worker_pids'],state['worker_native_pids'],launch['native_identities'])):
        if identity.get('launcher_pid') != launcher or identity.get('native_pid') != native:
            raise ValueError('Collect launch identity differs from the unterminated state')
        original.check_acknowledgment(execution,f'collect_{shard}',launcher,native)
    original.check_acknowledgment(execution,'supervisor',watch['supervisor_pid'],
                                  watch['supervisor_native_pid'])
    return state


def scientific_identity():
    frozen = original.read(CAL/'frozen/plan.json')
    recovery = original.read(CAL/'recovery/plan.json')
    original.verify_manifest(CAL/'recovery')
    if (recovery['original_plan_sha256'] != original.digest(CAL/'frozen/plan.json')
            or recovery['original_exit_status'] != 'unknown'
            or recovery['initial_count'] != frozen['stages']['calibration']['n_initials']
            or recovery['confirmation_initial_count'] != frozen['stages']['confirmation']['n_initials']
            or recovery['original_attempt'] != frozen['attempt']
            or recovery['allocated_error'] != frozen['allocated_error']
            or not recovery['schedule_models_thresholds_unchanged']
            or not recovery['statistical_sample_size_unchanged']
            or recovery['partial_effectiveness_outcomes_read']):
        raise ValueError('Parent recovery changed the registered scientific identity')
    for name, expected in recovery['source_sha256'].items():
        if original.digest(ROOT/name) != expected:
            raise ValueError('Frozen parent source changed: '+name)
    return frozen, recovery


def merge_source_hashes(*groups):
    result = {}
    for group in groups:
        for name, expected in group.items():
            if name in result and result[name] != expected:
                raise ValueError('Frozen plans disagree about source identity: '+name)
            result[name] = expected
    return result


def checked_blocks():
    plan = original.read(CAL/'frozen/plan.json')
    spec = plan['stages']['calibration']
    directory = CAL/'calibration'
    entries = []
    for block in sorted((directory/'blocks').iterdir()):
        index = int(block.name)
        if not 0 <= index < spec['n_initials']:
            raise ValueError('Block index is outside the frozen calibration schedule')
        database = block/'observations.sqlite'
        state = inventory(database)
        expected = original.schedule(spec,index)
        metadata = original.block_metadata(CAL,'calibration',index,None)
        markers = [(block/name).exists() for name in
                   ('observations.sqlite.seal.json','complete.json','sha256.json')]
        complete = all(markers)
        if state['metadata'] != metadata:
            raise ValueError('Interrupted block metadata changed')
        if complete:
            original.verify_manifest(block)
            seal = original.read(block/'observations.sqlite.seal.json')
            if (state['cases'] != [list(case) for case in expected]
                    or seal['database_sha256'] != original.digest(database)):
                raise ValueError('Completed block is inconsistent')
        else:
            if any(markers):
                raise ValueError('Partially sealed block requires a separate recovery decision')
            mechanics.prefix_records(database,metadata,expected,original.digest(database))
            if (block/'decision.json').exists() != (len(state['cases']) >= 10):
                raise ValueError('Prediction receipt and committed prefix disagree')
        entries.append(dict(initial_index=index,records=len(state['cases']),complete=complete,
            inventory=state,database_sha256=original.digest(database),
            decision_sha256=original.digest(block/'decision.json') if (block/'decision.json').exists() else None))
    if not entries:
        raise ValueError('No interrupted calibration records found')
    for shard in range(spec['shards']):
        indices = [entry['initial_index'] for entry in entries
                   if entry['initial_index'] % spec['shards'] == shard]
        if indices != original.assigned(spec,shard)[:len(indices)]:
            raise ValueError('Interrupted records are not a contiguous per-shard prefix')
    return directory, plan, entries


def verify_snapshot(require_committed=True):
    target = OUT/'calibration'
    if require_committed:
        original.require_committed([target/'inventory.json'])
    evidence = original.read(target/'inventory.json')
    directory = ROOT/evidence['source_directory']
    if (evidence['source_plan_sha256'] != original.digest(CAL/'frozen/plan.json')
            or evidence['parent_recovery_plan_sha256'] != original.digest(CAL/'recovery/plan.json')
            or evidence['source_exit_receipt_present']
            or evidence['original_exit_status'] != 'unknown'
            or evidence['partial_effectiveness_outcomes_read']):
        raise ValueError('Snapshot provenance or no-outcome contract changed')
    if files_at(directory) != evidence['original_files']:
        raise ValueError('Interrupted source files changed after preservation')
    seen = {}
    for part in evidence['archive_parts']:
        path = target/part['name']
        if require_committed:
            original.require_committed([path])
        if original.digest(path) != part['sha256']:
            raise ValueError('Archive part changed')
        with zipfile.ZipFile(path) as archive:
            if archive.namelist() != part['files']:
                raise ValueError('Archive ordering or membership changed')
            for name in part['files']:
                if name in seen:
                    raise ValueError('Duplicate file across archive parts')
                seen[name] = hashlib.sha256(archive.read(name)).hexdigest()
    if seen != evidence['original_files']:
        raise ValueError('Archive contents differ from every preserved source byte')
    for name, expected in evidence['source_sha256'].items():
        if original.digest(ROOT/name) != expected:
            raise ValueError('Frozen snapshot source changed: '+name)
    return evidence


def snapshot():
    presence = assert_stopped()
    last_state = terminal_state()
    plan, recovery = scientific_identity()
    original.sources_committed([ROOT/name for name in SOURCES])
    directory, checked_plan, entries = checked_blocks()
    if checked_plan != plan:
        raise ValueError('Block schedule uses a different frozen plan')
    source_hashes = merge_source_hashes(plan['source_sha256'],recovery['source_sha256'],
        {name:original.digest(ROOT/name) for name in SOURCES})
    for name, expected in source_hashes.items():
        if original.digest(ROOT/name) != expected:
            raise ValueError('Frozen source changed: '+name)
    original_files = files_at(directory)
    target = OUT/'calibration'
    target.mkdir(parents=True,exist_ok=False)
    archives = []
    for index, paths in enumerate(parts_for([directory/name for name in original_files])):
        name = f'original_{index:03d}.zip'
        path = target/name
        with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for source in paths:
                archive.write(source,source.relative_to(directory).as_posix())
        archives.append(dict(name=name,sha256=original.digest(path),bytes=path.stat().st_size,
            files=[source.relative_to(directory).as_posix() for source in paths]))
    retained = sum(entry['records'] for entry in entries)
    evidence = dict(utc=original.utc(),kind='calibration',interruption_generation=4,**presence,
        source_directory=directory.relative_to(ROOT).as_posix(),original_files=original_files,
        archive_parts=archives,blocks=entries,retained_roles=retained,
        fully_completed_blocks=sum(entry['complete'] for entry in entries),
        partial_blocks=sum(not entry['complete'] for entry in entries),source_sha256=source_hashes,
        source_plan_sha256=original.digest(CAL/'frozen/plan.json'),
        parent_recovery_plan_sha256=original.digest(CAL/'recovery/plan.json'),
        original_exit_status='unknown',source_exit_receipt_present=False,last_progress=last_state,
        initial_count=plan['stages']['calibration']['n_initials'],
        confirmation_initial_count=plan['stages']['confirmation']['n_initials'],
        original_attempt=plan['attempt'],allocated_error=plan['allocated_error'],
        schedule_models_thresholds_unchanged=True,statistical_sample_size_unchanged=True,
        partial_effectiveness_outcomes_read=False,new_independent_initials=0,
        recovery_rule='Replay every partial boundary exactly, then copy exact SQLite blobs and execute only the missing frozen suffix.',
        publication_rule='No calibration effect, candidate selection or risk claim is read before complete execution, review and actual zero exits.')
    if retained >= plan['stages']['calibration']['total_sumo_runs']:
        raise ValueError('An interruption snapshot requires a genuinely incomplete schedule')
    original.exclusive_json(target/'inventory.json',evidence)
    verify_snapshot(require_committed=False)
    if files_at(directory) != original_files:
        raise ValueError('Interrupted source changed during archive')
    assert_stopped()
    print(json.dumps(dict(complete=evidence['fully_completed_blocks'],partial=evidence['partial_blocks'],
        retained_roles=retained,archive_parts=len(archives))),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    snapshot()
