"""Independently extractable, byte-verified ZIP parts for completed studies."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
SCOPE=ROOT/'artifacts/mappo_ctde_20261101'


def archive(out,names):
    if len(names)!=len(set(names)):
        raise ValueError('Duplicate requested directories')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    files=[]
    for name in names:
        if '/' in name or '\\' in name or name in ('.','..'):
            raise ValueError('Explicit direct directory names only')
        directory=(SCOPE/name if name.endswith('_execution') else SCOPE/'development'/name).resolve()
        parent=SCOPE.resolve() if name.endswith('_execution') else (SCOPE/'development').resolve()
        if directory.parent!=parent or not directory.is_dir():
            raise ValueError('Outside the authorized completed MAPPO scope')
        if not any((directory/n).exists() for n in ('complete.json','completion.json','report.json')):
            raise ValueError('Refuse unfinished directory '+name)
        if (directory/'policy.pt').exists() and not (directory/'audit.json').exists():
            raise ValueError('PPO training audit required before delivery')
        if (directory/'protocol.json').exists():
            protocol=json.loads((directory/'protocol.json').read_text())
            if protocol.get('backend')=='sumo' and not (directory/'verification.json').exists():
                raise ValueError('Actual SUMO replay required before complete delivery')
        for path in directory.rglob('*'):
            if path.is_file():
                if not path.resolve().is_relative_to(directory):
                    raise ValueError('Escaped source path')
                files.append(path)
    files=sorted(files)
    parts=[];inventory=[];bundle=None;part=None;members=[]

    def close_part():
        nonlocal bundle
        if bundle is None:
            return
        bundle.close()
        data=part.read_bytes()
        with zipfile.ZipFile(part) as opened:
            if len(opened.namelist())!=len(members):
                raise ValueError('Part member count mismatch')
            for item in members:
                restored=opened.read(item['path'])
                if len(restored)!=item['bytes'] or hashlib.sha256(restored).hexdigest()!=item['sha256']:
                    raise ValueError('Archive byte verification failed')
        if len(data)>48*1024*1024:
            raise ValueError('Unexpected oversized ZIP part; original prefix retained')
        parts.append(dict(name=part.name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),files=len(members)))
        bundle=None

    for path in files:
        if bundle is None:
            part=out/f'completed_part_{len(parts)+1:03d}.zip'
            bundle=zipfile.ZipFile(part,'x',zipfile.ZIP_DEFLATED,compresslevel=6)
            members=[]
        data=path.read_bytes();name=path.relative_to(ROOT).as_posix()
        item=dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),part=part.name)
        bundle.writestr(name,data);members.append(item);inventory.append(item)
        if bundle.fp.tell()>=24*1024*1024:
            close_part()
    close_part()
    report=dict(completed_directories=names,files=len(inventory),original_bytes=sum(x['bytes'] for x in inventory),
        archive_bytes=sum(p['bytes'] for p in parts),parts=parts,all_archived_bytes_verified=True,
        restore='Extract every ZIP part into the repository root. Each part is a standalone ZIP; no concatenation needed.',
        inventory=inventory)
    (out/'inventory.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='inventory'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True)
    parser.add_argument('names',nargs='+')
    args=parser.parse_args();archive(args.out,args.names)
