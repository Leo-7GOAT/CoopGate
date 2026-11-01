"""Archive named COMPLETED MAPPO development directories, with byte inventory."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
SCOPE=ROOT/'artifacts/mappo_ctde_20261101/development'


def archive(out, names):
    out=Path(out)
    out.mkdir(parents=True, exist_ok=False)
    files=[]
    for name in names:
        directory=(SCOPE/name).resolve()
        if directory.parent != SCOPE.resolve() or not directory.is_dir():
            raise ValueError('Only explicitly named MAPPO development directories permitted')
        if not ((directory/'report.json').exists() or (directory/'complete.json').exists()):
            raise ValueError('Refusing unfinished directory '+name)
        files.extend(p for p in directory.rglob('*') if p.is_file())
    inventory=[]
    target=out/'completed_development.zip'
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(files):
            data=path.read_bytes()
            relative=path.relative_to(ROOT).as_posix()
            bundle.writestr(relative, data)
            inventory.append(dict(path=relative, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    with zipfile.ZipFile(target) as bundle:
        if len(bundle.namelist()) != len(inventory):
            raise ValueError('Archive file count mismatch')
        for item in inventory:
            data=bundle.read(item['path'])
            if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
                raise ValueError('Archive byte verification failed')
    report=dict(completed_directories=names, files=len(inventory), original_bytes=sum(i['bytes'] for i in inventory),
                archive=target.name, archive_bytes=target.stat().st_size,
                archive_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                all_archived_bytes_verified=True, inventory=inventory)
    (out/'inventory.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='inventory'}), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--out', required=True)
    parser.add_argument('names', nargs='+')
    args=parser.parse_args(); archive(args.out, args.names)
