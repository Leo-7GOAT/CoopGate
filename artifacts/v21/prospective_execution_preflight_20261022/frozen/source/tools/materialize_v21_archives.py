"""Recreate frozen V21 raw files from Git-tracked archives without overwriting.

Fresh clones contain compact immutable archives rather than tens of thousands
of loose trace files. This command restores exact bytes at their original paths.
It does not run experiments, alter results, or manufacture process exit records.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import uuid
import zipfile


ROOT = Path(__file__).resolve().parents[1]
FIRST = ROOT / 'artifacts/v21/interrupted_recovery_20261021/frozen'
SECOND = ROOT / 'artifacts/v21/interrupted_recovery_20261022/frozen'


def checksum(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def confined(root, relative):
    name = PurePosixPath(relative)
    if (name.is_absolute() or not name.parts or any(p in ('..', '.') for p in name.parts)
            or '\\' in relative or ':' in relative):
        raise ValueError('Archive paths must remain relative to the declared root')
    target = root.joinpath(*name.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError('Archive path resolves outside the declared root')
    return target


def verify_manifest(directory):
    manifest = read(directory/'sha256.json')
    for name, expected in manifest.items():
        if checksum(confined(directory, name)) != expected:
            raise ValueError('Frozen archive/source manifest differs: '+name)
    return manifest


def existing_state(root, members):
    state = {}
    for name, expected in members.items():
        path = confined(root, name)
        if path.exists():
            if not path.is_file() or checksum(path) != expected:
                raise ValueError('Refusing to replace an existing different file: '+str(path))
            state[name] = True
        else:
            state[name] = False
    return state


def install_stream(root, name, stream, expected):
    path = confined(root, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent/('.materialize-'+uuid.uuid4().hex+'.partial')
    try:
        with temporary.open('xb') as handle:
            shutil.copyfileobj(stream, handle, length=1024*1024)
        if checksum(temporary) != expected:
            raise ValueError('Extracted content differs from frozen checksum')
        # Hard-link creation is exclusive: a concurrently created target is not
        # overwritten, unlike rename/replace. The final bytes are complete.
        os.link(temporary, path)
    finally:
        if temporary.exists():
            if not temporary.resolve().is_relative_to(root.resolve()):
                raise ValueError('Temporary extraction path escaped the root')
            temporary.unlink()


def restore_archive(archive, destination, members, *, check_only=False):
    destination = Path(destination)
    with zipfile.ZipFile(archive) as stored:
        entries = stored.infolist()
        names = [entry.filename for entry in entries]
        if len(set(names)) != len(names) or set(names) != set(members):
            raise ValueError('Archive members differ from the exact frozen file set')
        for entry in entries:
            confined(destination, entry.filename)
            if entry.is_dir() or ((entry.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError('Only ordinary archived files are supported')
        state = existing_state(destination, members)
        missing = [name for name in names if not state[name]]
        if not check_only:
            for name in missing:
                with stored.open(name) as stream:
                    install_stream(destination, name, stream, members[name])
        return dict(existing=len(names)-len(missing), missing=len(missing), restored=0 if check_only else len(missing))


def plan():
    verify_manifest(FIRST)
    second_manifest = verify_manifest(SECOND)
    first, second = read(FIRST/'plan.json'), read(SECOND/'plan.json')
    if checksum(FIRST/'plan.json') != second['previous_recovery_plan_sha256']:
        raise ValueError('Continuation refers to a different original archive plan')
    archives, copies = [], []
    for item in first['studies'].values():
        for shard, entry in item['shards'].items():
            source = confined(FIRST, entry['archive'])
            if checksum(source) != entry['archive_sha256']:
                raise ValueError('Original shard archive differs')
            destination = confined(ROOT, item['directory']+'/shard_'+shard)
            archives.append((source, destination, entry['original_files']))
    for shard in range(3):
        prefix = f'artifacts/v21/interrupted_recovery_20261021/suffix/shard_{shard}/'
        members = {name[len(prefix):]:value for name,value in second['previous_files'].items() if name.startswith(prefix)}
        source = SECOND/f'previous_suffix_{shard}.zip'
        if source.name not in second_manifest:
            raise ValueError('Continuation archive is not bound to its manifest')
        archives.append((source, confined(ROOT, prefix.rstrip('/')), members))
    prefix = 'artifacts/v21/interrupted_recovery_20261021/execution/'
    for name, expected in second['previous_files'].items():
        if name.startswith(prefix):
            source = SECOND/'previous_execution'/name[len(prefix):]
            if checksum(source) != expected:
                raise ValueError('Preserved execution record differs')
            copies.append((source, name, expected))
        elif '/suffix/shard_' not in name:
            raise ValueError('Unexpected previous file role')
    return archives, copies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true', help='Report missing files without restoring them')
    args = parser.parse_args()
    archives, copies = plan()
    # Check every existing target before creating anything. Existing modified
    # observations are never silently repaired or replaced.
    for _, destination, members in archives:
        existing_state(destination, members)
    copy_state = existing_state(ROOT, {name:value for _,name,value in copies})
    counts = dict(existing=0, missing=0, restored=0)
    for source, destination, members in archives:
        part = restore_archive(source, destination, members, check_only=args.check_only)
        for key in counts:
            counts[key] += part[key]
    for source, name, expected in copies:
        counts['existing' if copy_state[name] else 'missing'] += 1
        if not copy_state[name] and not args.check_only:
            with source.open('rb') as stream:
                install_stream(ROOT, name, stream, expected)
            counts['restored'] += 1
    print(json.dumps(dict(**counts, actual_sumo_runs=0, outcomes_changed=False)), flush=True)


if __name__ == '__main__':
    main()
