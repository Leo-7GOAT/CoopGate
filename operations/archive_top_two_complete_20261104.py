"""Archive the fully completed original attempt6 scope, preserving every byte.

Each independently extractable ZIP is read back against the input hash. This
does not simulate, choose observations or recompute a scientific test.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from mappo_completed_top_two_attempt6 import complete_evidence

BASE=ROOT/'artifacts/mappo_ctde_20261101'
STUDY=BASE/'formal_mappo_attempt6_top_two_recovery_20261104'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive(out):
    complete_evidence(STUDY)
    out=Path(out).resolve()
    if out.parent!=BASE.resolve() or out.exists():
        raise ValueError('Use a new direct project artifact directory')
    files=sorted(p for p in STUDY.rglob('*') if p.is_file())
    if any(not p.resolve().is_relative_to(STUDY.resolve()) for p in files):
        raise ValueError('Escaped original study directory')
    out.mkdir(exist_ok=False)
    inventory=[];parts=[];bundle=None;members=[];part=None

    def close_part():
        nonlocal bundle
        if bundle is None:return
        bundle.close()
        with part.open('rb+') as f:os.fsync(f.fileno())
        with zipfile.ZipFile(part) as opened:
            if len(opened.namelist())!=len(members):
                raise ValueError('Archive membership differs')
            for item in members:
                data=opened.read(item['path'])
                if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:
                    raise ValueError('Archive readback differs')
        if part.stat().st_size>48*1024*1024:
            raise ValueError('Oversized part retained for inspection')
        parts.append(dict(name=part.name,bytes=part.stat().st_size,sha256=sha(part),files=len(members)))
        bundle=None

    for path in files:
        if bundle is None:
            part=out/f'completed_part_{len(parts)+1:03d}.zip'
            bundle=zipfile.ZipFile(part,'x',zipfile.ZIP_DEFLATED,compresslevel=6)
            members=[]
        data=path.read_bytes();name=path.relative_to(ROOT).as_posix()
        item=dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),part=part.name)
        bundle.writestr(name,data);members.append(item);inventory.append(item)
        if bundle.fp.tell()>=24*1024*1024:close_part()
    close_part()
    # Every source must remain unchanged for the entire archive operation.
    if any(sha(ROOT/item['path'])!=item['sha256'] for item in inventory):
        raise ValueError('Completed study changed during archival')
    report=dict(completed=True,study=STUDY.relative_to(ROOT).as_posix(),
        files=len(inventory),original_bytes=sum(v['bytes'] for v in inventory),
        archive_bytes=sum(v['bytes'] for v in parts),parts=parts,
        all_archived_bytes_verified=True,full_effective_schedule=59392,
        retained_actual_records=642,new_actual_sumo_executions=58750,
        independent_initials=7168,original_interrupted_owner_exit_code=None,
        unknown_lost_inflight_executions=True,new_error_allocated=False,
        certificate_sha256=sha(STUDY/'analysis/certificate.json'),
        source_sha256=sha(Path(__file__)),
        restore='Extract all standalone ZIP parts into the repository root; do not concatenate.',
        inventory=inventory)
    with (out/'inventory.json').open('x',encoding='utf-8') as f:
        json.dump(report,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps({k:v for k,v in report.items() if k!='inventory'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True)
    archive(parser.parse_args().out)
