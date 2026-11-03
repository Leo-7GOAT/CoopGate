# Derived exact-byte archiver sha256=cf4f13cd3e48ddfa23b5c91787dc1f4b7e55fe09ae43707efdfa9fc50a66ca2d
"""Independently extractable, byte-verified ZIP parts for completed studies."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
SCOPE=ROOT/'artifacts/mappo_ctde_20261101'


def archive(out,names):
    expected={'joint_admissibility_development_20261103',
        'joint_admissibility_audit_recovery_20261103',
        'switching_reserve_smoke_20261103','switching_reserve_development_20261103'}
    if set(names)!=expected or len(names)!=4:raise ValueError('Explicit complete development delivery only')
    for name in names:
        directory=SCOPE/'development'/name
        owner=json.loads((directory/'owner_native_exit.json').read_bytes())
        expected_code=1 if name=='joint_admissibility_development_20261103' else 0
        if owner['actual_native_exit_code']!=expected_code:raise ValueError('Original failure and actual recovery exits required')
        if expected_code==0 and owner['report_sha256']!=hashlib.sha256((directory/'report.json').read_bytes()).hexdigest():
            raise ValueError('Actual report completion binding differs')
    recovery=json.loads((SCOPE/'development/joint_admissibility_audit_recovery_20261103/report.json').read_bytes())
    reserve=json.loads((SCOPE/'development/switching_reserve_development_20261103/report.json').read_bytes())
    if (recovery['actual_sumo_executions']!=700 or recovery['new_recovery_actual_sumo_executions']!=56
            or reserve['additional_actual_sumo_executions']!=297 or reserve['effective_paired_executions']!=1000):
        raise ValueError('All original retained and missing executions required')
    if len(names)!=len(set(names)):
        raise ValueError('Duplicate requested directories')
    out=Path(out)
    if out.exists():raise FileExistsError('Retain existing delivery')
    files=[]
    for name in names:
        if '/' in name or '\\' in name or name in ('.','..'):
            raise ValueError('Explicit direct directory names only')
        directory=(SCOPE/'development'/name).resolve()
        if directory.parent!=(SCOPE/'development').resolve() or not directory.is_dir():
            raise ValueError('Outside the explicitly authorized completed development scope')
        for path in directory.rglob('*'):
            if path.is_file():
                if not path.resolve().is_relative_to(directory):
                    raise ValueError('Escaped source path')
                files.append(path)
    files=sorted(files)
    out.mkdir(parents=True,exist_ok=False)
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
