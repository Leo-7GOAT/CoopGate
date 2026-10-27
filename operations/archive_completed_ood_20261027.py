"""Archive sealed OOD collection bytes; do not interpret partial review results."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'artifacts/recovery_20261026/ood_recovery'
TARGET = ROOT/'artifacts/recovery_20261027/ood_completed_collection'
LIMIT = 60*1024*1024


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def files():
    return sorted(path for folder in ('blocks','boundary','execution')
                  for path in (SOURCE/folder).rglob('*') if path.is_file())


def main():
    if TARGET.exists():
        raise FileExistsError('Preserve any existing archive prefix')
    receipt = json.loads((SOURCE/'execution/exit.json').read_text(encoding='utf-8'))
    done = json.loads((SOURCE/'execution/collection_complete.json').read_text(encoding='utf-8'))
    manifest = json.loads((SOURCE/'execution/blocks_manifest.json').read_text(encoding='utf-8'))
    if (receipt['exit_code'] != 0 or done['actual_sumo_runs'] != 22464
            or done['blocks'] != 1728 or manifest['blocks'] != 1728
            or len(manifest['entries']) != 1728
            or digest(SOURCE/'execution/collection_complete.json') != receipt['collection_complete_sha256']
            or digest(SOURCE/'execution/blocks_manifest.json') != receipt['blocks_manifest_sha256']):
        raise ValueError('Actual complete original OOD collection required')
    for index, entry in enumerate(manifest['entries']):
        if entry['block_index'] != index:
            raise ValueError('Block schedule is not complete')
        for name,key in (('complete.json','complete_sha256'),
                         ('observations.sqlite','database_sha256'),
                         ('observations.sqlite.seal.json','seal_sha256')):
            if digest(SOURCE/'blocks'/f'{index:05d}'/name) != entry[key]:
                raise ValueError('Sealed collection bytes changed')
    originals = {p.relative_to(SOURCE).as_posix():digest(p) for p in files()}
    parts, current, size = [], [], 0
    for name in originals:
        length = (SOURCE/name).stat().st_size
        if length > LIMIT:
            raise ValueError('Single source file exceeds archive budget')
        if current and size+length > LIMIT:
            parts.append(current)
            current, size = [], 0
        current.append(name)
        size += length
    if current:
        parts.append(current)
    TARGET.mkdir(parents=True)
    archives, restored = [], {}
    for index, members in enumerate(parts):
        path = TARGET/f'collection_{index:03d}.zip'
        with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for name in members:
                archive.write(SOURCE/name,name)
        with zipfile.ZipFile(path) as archive:
            if archive.namelist() != members:
                raise ValueError('Archive membership differs')
            for name in members:
                if name in restored:
                    raise ValueError('Duplicate archived source')
                restored[name] = hashlib.sha256(archive.read(name)).hexdigest()
        archives.append(dict(name=path.name,sha256=digest(path),bytes=path.stat().st_size,files=members))
    if restored != originals or {p.relative_to(SOURCE).as_posix():digest(p) for p in files()} != originals:
        raise ValueError('Archive or source bytes changed')
    value = dict(utc=datetime.now(timezone.utc).isoformat(),
                 source=SOURCE.relative_to(ROOT).as_posix(),
                 actual_collection_exit_code=0, actual_sumo_runs=22464,
                 paired_blocks=1728, independent_initials=192,
                 extra_boundary_replays=1, statistical_sample_size_unchanged=True,
                 original_files_sha256=originals, archive_parts=archives,
                 all_sealed_block_hashes_verified=True,
                 every_archived_file_byte_verified=True,
                 full_scientific_replay_acceptance_claim=False,
                 restore='Extract all parts into the source directory; compare every relative file SHA256. Prior frozen recovery inputs remain at their existing repository paths.')
    with (TARGET/'inventory.json').open('x',encoding='utf-8') as handle:
        json.dump(value,handle,indent=2)
        handle.write('\n')
    print(json.dumps(dict(files=len(originals),parts=len(archives),
                          bytes=sum(p['bytes'] for p in archives),
                          scientific_acceptance_claim=False)),flush=True)


if __name__ == '__main__':
    main()
