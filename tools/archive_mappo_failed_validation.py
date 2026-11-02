"""Deliver failed/noncollected validation evidence without a completion claim."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/mappo_ctde_20261101'
STUDIES={
    'formal_adapter_validation_native_pid_failure_20261102':dict(actual_sumo_executions=64,full_independent_review_complete=False,parent_actual_exit=1,reason='Native venv redirector PID differs from interpreter;review correctly refused'),
    'formal_adapter_validation_bootstrap_args_failure_20261102':dict(actual_sumo_executions=0,full_independent_review_complete=False,parent_actual_exit=1,child_actual_exit=None,reason='Bootstrap duplicated script argument;original child numeric terminal status unavailable'),
    'formal_adapter_validation_precollection_v1_20261102':dict(actual_sumo_executions=0,full_independent_review_complete=False,parent_actual_exit=None,reason='Revised before collection;never started'),
}


def archive():
    out=BASE/'failed_formal_adapter_validation_delivery_20261102';out.mkdir(exist_ok=False)
    inventory=[]
    with zipfile.ZipFile(out/'failed_validation_evidence.zip','x',zipfile.ZIP_DEFLATED,compresslevel=6) as bundle:
        for name in STUDIES:
            root=BASE/'development'/name
            if not (root/'source_snapshot.json').exists():raise ValueError('Exact source snapshots required')
            for path in sorted(root.rglob('*')):
                if path.is_file():
                    data=path.read_bytes();relative=path.relative_to(ROOT).as_posix()
                    inventory.append(dict(path=relative,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
                    bundle.writestr(relative,data)
    with zipfile.ZipFile(out/'failed_validation_evidence.zip') as bundle:
        for item in inventory:
            data=bundle.read(item['path'])
            if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Archive bytes changed')
    data=(out/'failed_validation_evidence.zip').read_bytes()
    report=dict(studies=STUDIES,formal_samples=0,error_allocated=False,not_completed_studies=True,
        all_archive_bytes_verified=True,archive_bytes=len(data),archive_sha256=hashlib.sha256(data).hexdigest(),
        restore='Extract ZIP at repository root;failed prefixes retain original status,not replacement experiments',inventory=inventory)
    (out/'inventory.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='inventory'}),flush=True)


if __name__=='__main__':archive()
