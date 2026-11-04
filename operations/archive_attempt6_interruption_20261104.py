"""Archive immutable interruption evidence and recovery registration, not live runs."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import sys
import zipfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'tools'))
import recover_mappo_top_two_attempt6 as r
OUT = r.original.BASE / 'completed_attempt6_interruption_registration_archive_20261104'


def run(out=None):
    out = OUT if out is None else Path(out).resolve()
    if not out.resolve().is_relative_to(r.original.BASE.resolve()):
        raise ValueError('Archive output must stay inside this project evidence directory')
    binding, report = r.guard()
    owner = r.original.read(r.AUDIT/'owner_native_exit.json')
    if owner['actual_native_exit_code'] != 0 or owner['report_sha256'] != r.original.sha(r.AUDIT/'report.json'):
        raise ValueError('Completed actual interruption audit required')
    files = []
    for name in ('raw', 'retained'):
        files.extend(p for p in (r.AUDIT/name).rglob('*') if p.is_file())
    files.extend(r.AUDIT/name for name in ('report.json', 'owner_native_exit.json'))
    for name in ('plan.json','allocation.json','binding.json','source_snapshot.json',
                 'recovery_binding.json','public_recovery_registration.json','owner_launch.json'):
        files.append(r.OUT/name)
    files.extend(p for p in (r.OUT/'source_snapshot').rglob('*') if p.is_file())
    files.extend([Path(r.__file__),r.PROTOCOL,ROOT/'tests/test_mappo_attempt6_recovery.py',
                  ROOT/'tools/mappo_completed_top_two_attempt6.py',Path(__file__)])
    files = sorted(files)
    if len(files) != len(set(files)) or any(not p.resolve().is_relative_to(ROOT) for p in files):
        raise ValueError('Explicit authorized file scope only')
    for item in report['raw_inventory']:
        p = r.AUDIT/'raw'/item['path']
        if r.original.sha(p) != item['sha256'] or p.stat().st_size != item['bytes']:
            raise ValueError('Original interruption snapshot changed')
    out.mkdir(exist_ok=False)
    inventory = []; parts = []; bundle = None; members = []
    def close_part():
        nonlocal bundle
        if bundle is None:return
        bundle.close()
        with zipfile.ZipFile(part) as z:
            if len(z.namelist()) != len(members):raise ValueError('Wrong member count')
            for q in members:
                data=z.read(q['path'])
                if len(data)!=q['bytes'] or hashlib.sha256(data).hexdigest()!=q['sha256']:
                    raise ValueError('Restored interruption archive bytes differ')
        with part.open('rb+') as f:f.flush();os.fsync(f.fileno())
        parts.append(dict(name=part.name,bytes=part.stat().st_size,sha256=r.original.sha(part)))
        if part.stat().st_size > 48*1024*1024:raise ValueError('Oversized archive part retained')
        bundle=None
    for path in files:
        if bundle is None:
            part=out/f'interruption_registration_part_{len(parts)+1:03d}.zip'
            bundle=zipfile.ZipFile(part,'x',zipfile.ZIP_DEFLATED,compresslevel=6);members=[]
        data=path.read_bytes();name=path.relative_to(ROOT).as_posix()
        item=dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),part=part.name)
        bundle.writestr(name,data);inventory.append(item);members.append(item)
        if bundle.fp.tell() >= 24*1024*1024:close_part()
    close_part()
    result=dict(completed=True,scope='Immutable original interruption snapshot, all642retained committed records, and recovery registration only',
        live_recovery_outputs_excluded=True,recovered_formal_study_complete=False,
        original_native_exit_code=None,unknown_lost_inflight_executions=True,
        all_archive_members_readback_sha256_verified=True,
        retained_records=642,new_required_planned_runs=58750,new_error_allocated=False,
        scratch_copies_excluded_as_reproducible_from_preserved_raw_inputs=True,
        original_bytes=sum(q['bytes'] for q in inventory),files=len(inventory),parts=parts,inventory=inventory,
        restore='Extract all standalone ZIP parts at repository root; preserve original files and unknown exit status.')
    r.durable(out/'inventory.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='inventory'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path)
    run(parser.parse_args().out)
