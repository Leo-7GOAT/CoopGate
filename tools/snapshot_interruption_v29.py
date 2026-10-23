"""Read-only inventories and bounded archive parts for the second interruption."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile

from coopgate.episode_store_v21 import inventory
import audit_buffered_scope_v25 as buffered
import execute_prospective_v23 as original
import recover_prospective_v24 as prior

ROOT = original.ROOT
OUT = ROOT/'artifacts/v29/interruption_20261023'
CAL = ROOT/'artifacts/v24/prospective_calibration_recovery_20261022'
BUFFER = ROOT/'artifacts/v25/buffered_complete_scope'
PART_LIMIT = 60*1024*1024


def files_at(directory):
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


def assert_stopped():
    processes = original.native_snapshot()
    others = [p for p in processes if p['ProcessId'] not in (os.getpid(), os.getppid())]
    original.require_project_stopped(others)
    names = ('recover_prospective_v24.py', 'audit_buffered_scope_v25.py',
             'recover_prospective_v29.py', 'recover_buffered_v29.py')
    if any(any(name in (p.get('CommandLine') or '') for name in names) for p in others):
        raise ValueError('A target execution is still live; an expired observation is not an interruption')
    return dict(observed_utc=original.utc(), target_execution_processes_absent=True)


def checked_blocks(kind):
    directory = CAL/'calibration' if kind == 'calibration' else BUFFER
    plan = original.read(CAL/'frozen/plan.json' if kind == 'calibration' else BUFFER/'plan.json')
    entries = []
    for block in sorted((directory/'blocks').iterdir()):
        index = int(block.name)
        db = block/'observations.sqlite'
        state = inventory(db)
        if kind == 'calibration':
            spec = plan['stages']['calibration']
            expected = original.schedule(spec, index)
            metadata = original.block_metadata(CAL, 'calibration', index, None)
        else:
            group = plan['groups'][index]
            expected = buffered.schedule(group)
            metadata = buffered.metadata(group)
        markers = [(block/name).exists() for name in ('observations.sqlite.seal.json', 'complete.json')]
        complete = all(markers)
        if state['metadata'] != metadata:
            raise ValueError('Original block metadata changed')
        if complete:
            if state['cases'] != [list(c) for c in expected]:
                raise ValueError('Completed block has the wrong schedule')
            seal = original.read(block/'observations.sqlite.seal.json')
            if seal['database_sha256'] != original.digest(db):
                raise ValueError('Completed database seal differs')
            if kind == 'calibration':
                original.verify_manifest(block)
            else:
                receipt = original.read(block/'complete.json')
                if (receipt['database_sha256'] != original.digest(db)
                        or receipt['seal_sha256'] != original.digest(block/'observations.sqlite.seal.json')
                        or len(receipt['paired_cases']) != len(group['cases'])):
                    raise ValueError('Completed paired audit is inconsistent')
        else:
            if any(markers) or (block/'sha256.json').exists():
                raise ValueError('Partially sealed block requires a separate recovery decision')
            prior.prefix_records(db, metadata, expected, original.digest(db))
            if kind == 'calibration' and (block/'decision.json').exists() != (len(state['cases']) >= 10):
                raise ValueError('Prediction receipt and committed prefix disagree')
        entries.append(dict(initial_index=index, records=len(state['cases']), complete=complete,
            inventory=state, database_sha256=original.digest(db),
            decision_sha256=original.digest(block/'decision.json') if (block/'decision.json').exists() else None))
    if kind == 'calibration':
        for shard in range(spec['shards']):
            selected = [e['initial_index'] for e in entries if e['initial_index'] % spec['shards'] == shard]
            if selected != original.assigned(spec, shard)[:len(selected)]:
                raise ValueError('Calibration records are not contiguous within each shard')
    elif [e['initial_index'] for e in entries] != list(range(len(entries))):
        raise ValueError('Buffered records are not a contiguous prefix')
    return directory, plan, entries


def verify_snapshot(kind, require_committed=True):
    target = OUT/kind
    if require_committed:
        original.require_committed([target/'inventory.json'])
    evidence = original.read(target/'inventory.json')
    directory = ROOT/evidence['source_directory']
    if files_at(directory) != evidence['original_files']:
        raise ValueError('Interrupted source files changed after preservation')
    seen = {}
    for part in evidence['archive_parts']:
        path = target/part['name']
        if require_committed:
            original.require_committed([path])
        if original.digest(path) != part['sha256']:
            raise ValueError('An archive part changed')
        with zipfile.ZipFile(path) as archive:
            if archive.namelist() != part['files']:
                raise ValueError('Archive file ordering or membership changed')
            for name in part['files']:
                if name in seen:
                    raise ValueError('Duplicate file across archive parts')
                seen[name] = hashlib.sha256(archive.read(name)).hexdigest()
    if seen != evidence['original_files']:
        raise ValueError('Archive contents differ from every preserved source byte')
    return evidence


def snapshot(kind):
    presence = assert_stopped()
    if kind == 'calibration':
        prior.inputs()
        sources = {**original.read(CAL/'frozen/plan.json')['source_sha256'],
                   **original.read(CAL/'recovery/plan.json')['source_sha256']}
    else:
        sources = buffered.inputs()['source_sha256']
    for name, value in sources.items():
        if original.digest(ROOT/name) != value:
            raise ValueError('Frozen source changed: '+name)
    directory, plan, entries = checked_blocks(kind)
    original_files = files_at(directory)
    target = OUT/kind
    target.mkdir(parents=True, exist_ok=False)
    archives = []
    paths = [directory/name for name in original_files]
    for index, paths_in_part in enumerate(parts_for(paths)):
        name = f'original_{index:03d}.zip'
        path = target/name
        with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for file in paths_in_part:
                archive.write(file, file.relative_to(directory).as_posix())
        archives.append(dict(name=name, sha256=original.digest(path), bytes=path.stat().st_size,
            files=[p.relative_to(directory).as_posix() for p in paths_in_part]))
    execution = directory/'execution'
    evidence = dict(utc=original.utc(), kind=kind, **presence,
        source_directory=directory.relative_to(ROOT).as_posix(), original_files=original_files,
        archive_parts=archives, blocks=entries, retained_roles=sum(e['records'] for e in entries),
        fully_completed_blocks=sum(e['complete'] for e in entries),
        partial_blocks=sum(not e['complete'] for e in entries), source_sha256=sources,
        original_exit_status='unknown', partial_effectiveness_outcomes_read=False,
        source_plan_sha256=original.digest(CAL/'frozen/plan.json' if kind == 'calibration' else BUFFER/'plan.json'),
        source_exit_receipt_present=any((execution/n).exists() for n in ('exit.json','supervisor_exit.json','completed.json')),
        last_progress=original.read(execution/('state.json' if kind == 'calibration' else 'progress.json')),
        previous_goal_turn='progress', previous_turn_evidence='Published complete V28 diagnostic observations, figures and verification at commit 047ecc15.')
    if evidence['source_exit_receipt_present']:
        raise ValueError('Inspect terminal receipts instead of treating this execution as an unknown exit')
    original.exclusive_json(target/'inventory.json', evidence)
    verify_snapshot(kind, require_committed=False)
    assert_stopped()
    print(json.dumps(dict(kind=kind, complete=evidence['fully_completed_blocks'], partial=evidence['partial_blocks'],
                         retained_roles=evidence['retained_roles'], archive_parts=len(archives))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=('calibration', 'buffered'))
    snapshot(parser.parse_args().kind)
