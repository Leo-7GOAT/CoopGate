"""Archive complete paired recovery AND explicitly incomplete original evidence.

Never labels the interrupted original owner successful or infers its native
attempt count. Every extracted byte is independently hashed before delivery.
"""
import hashlib,json,zipfile
from pathlib import Path
import recover_coupled_holdout_ablation as recovery


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run():
    _,reg=recovery.guard();root=recovery.a.e.ROOT
    source=recovery.OUT;report=recovery.a.e.read(source/'report.json');owner=recovery.a.e.read(source/'owner_native_exit.json')
    if (owner['actual_native_exit_code']!=0 or owner['report_sha256']!=sha(source/'report.json')
        or not report['completed'] or report['complete_recorded_old_gate_outcomes']!=3072
        or report['prior_durable_observations_retained']!=1375 or report['actual_resumed_sumo_executions']!=1697
        or not report['original_root_exit_unknown'] or not report['total_attempt_count_including_lost_inflight_not_identifiable']
        or len(report['children'])!=4 or any(c['actual_native_exit_code']!=0 for c in report['children'])):
        raise ValueError('Exact complete recovery and honest interruption disclosure required')
    out=recovery.a.e.BASE/'paired_recovery_delivery_20261103';out.mkdir(exist_ok=False)
    files=sorted(p for d in (recovery.ORIGINAL,source) for p in d.rglob('*') if p.is_file())
    inventory=[];parts=[];bundle=None;members=[]
    def close():
        nonlocal bundle
        if bundle is None:return
        bundle.close()
        with zipfile.ZipFile(part) as opened:
            if len(opened.namelist())!=len(members):raise ValueError('Member count mismatch')
            for m in members:
                data=opened.read(m['path'])
                if len(data)!=m['bytes'] or hashlib.sha256(data).hexdigest()!=m['sha256']:raise ValueError('Restored bytes differ')
        if part.stat().st_size>48*1024*1024:raise ValueError('Oversized archive retained')
        parts.append(dict(name=part.name,bytes=part.stat().st_size,sha256=sha(part),files=len(members)));bundle=None
    for p in files:
        if not p.resolve().is_relative_to(recovery.ORIGINAL.resolve()) and not p.resolve().is_relative_to(source.resolve()):
            raise ValueError('Escaped explicit source scope')
        if bundle is None:
            part=out/f'evidence_part_{len(parts)+1:03d}.zip';bundle=zipfile.ZipFile(part,'x',zipfile.ZIP_DEFLATED,compresslevel=6);members=[]
        data=p.read_bytes();name=p.relative_to(root).as_posix()
        m=dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),part=part.name)
        bundle.writestr(name,data);members.append(m);inventory.append(m)
        if bundle.fp.tell()>=24*1024*1024:close()
    close()
    manifest=dict(completed_recovery=source.name,interrupted_original=recovery.ORIGINAL.name,
        original_native_exit_code=None,total_attempt_count_including_lost_inflight_unknown=True,
        complete_recorded_outcomes=3072,retained_original_records=1375,actual_resumed_sumo_executions=1697,
        files=len(inventory),parts=parts,original_bytes=sum(m['bytes'] for m in inventory),
        archive_bytes=sum(p['bytes'] for p in parts),all_extracted_bytes_verified=True,
        report_sha256=sha(source/'report.json'),owner_receipt_sha256=sha(source/'owner_native_exit.json'),
        note='paired_record_analysis.json derives all3072 planned outcomes;its legacy actual_additional_sumo_executions field is a record count,not a recovery execution count. Authoritative report.json separates1375 retained and1697 actual resumed executions.',
        restore='Extract every independent ZIP into repository root;original interrupted directory remains incomplete.',inventory=inventory)
    recovery.a.e.write(out/'inventory.json',manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k!='inventory'}),flush=True)


if __name__=='__main__':run()
