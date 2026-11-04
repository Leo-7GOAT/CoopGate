"""Verify old terminal/hash chains and full allocation inventory;reserve nothing."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/mappo_ctde_20261101'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())


def verify():
    inventory_path=BASE/'error_ledger_full_json_audit_20261102.json';inventory=read(inventory_path)
    if inventory['parse_errors'] or inventory['size_exclusions'] or inventory['filename_filter_used']:
        raise ValueError('Incomplete allocation inventory')
    evidence={inventory_path.relative_to(ROOT).as_posix():sha(inventory_path)}
    for item in inventory['allocation_candidate_files']:
        if sha(ROOT/item['path'])!=item['sha256']:raise ValueError('Known allocation metadata changed since complete inventory')
        for entry in item['entries']:
            if entry['key_path'].split('/')[-1].lower() in ('attempt','attempt_id','allocation_attempt','error_allocation_attempt_reserved'):
                if type(entry['value']) in (int,float) and entry['value']>=3:raise ValueError('Attempt3 or later already present')
    withdrawal=ROOT/'artifacts/legacy_withdrawal_20261027';amendment=read(withdrawal/'amendment.json')
    if amendment['old_alpha']!=.025 or amendment['new_alpha']!=.05/6 or amendment['old_alpha_reassigned']:
        raise ValueError('Withdrawn allocation must never be reassigned')
    if not read(withdrawal/'authorization.json')['human_approval_received']:raise ValueError('Old withdrawal lacks authorization')
    for name,digest in amendment['inputs'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Withdrawal authorization/terminal chain changed: '+name)
        evidence[name]=digest
    stopped=read(withdrawal/'process_stop.json')
    if any(type(p.get('actual_exit_code')) is not int for p in stopped['processes']):raise ValueError('Old predecessor handle lacks terminal status')
    path=ROOT/'artifacts/joint_risk_recovery_20261031/analysis';receipt=read(path/'completion_receipt_20261031.json')
    if (receipt['native_finalizer_exit_code']!=0 or receipt['launcher_exit_code']!=0 or not receipt['retained_native_handle_witness']
            or receipt['actual_unique_runs']!=16384 or receipt['independent_initials']!=4096 or not receipt['full_scheduled_key_coverage_exact']):
        raise ValueError('Previous formal family not fully terminal')
    for key,name in (('verification_sha256','verification.json'),('certificate_sha256','certificate.json'),('replay_sha256','replay.json'),
                     ('native_start_sha256','native_finalizer_start.json'),('native_exit_sha256','native_finalizer_exit.json')):
        if sha(path/name)!=receipt[key]:raise ValueError('Old finalizer chain changed: '+name)
        evidence[(path/name).relative_to(ROOT).as_posix()]=receipt[key]
    native=read(path/'native_finalizer_exit.json');certificate=read(path/'certificate.json')
    if native['exit_code']!=0 or not native['retained_native_handle'] or certificate['attempt']!=2 or certificate['allocated_error']!=.05/6:
        raise ValueError('Old actual terminal/allocated attempt differs')
    for stage in ('calibration','confirmation'):
        p=path.parent/stage/'execution';record=read(p/'exit.json')
        if record['exit_code']!=0 or record['stage']!=stage:raise ValueError('Old stage is not terminal zero')
        for name,digest in record['hashes'].items():
            if sha(p/name)!=digest:raise ValueError('Old stage native completion chain changed')
            evidence[(p/name).relative_to(ROOT).as_posix()]=digest
        evidence[(p/'exit.json').relative_to(ROOT).as_posix()]=sha(p/'exit.json')
    evidence[(path/'completion_receipt_20261031.json').relative_to(ROOT).as_posix()]=sha(path/'completion_receipt_20261031.json')
    out=BASE/'mappo_predecessor_budget_audit_20261102';out.mkdir(exist_ok=False)
    archive=out/'full_json_inventory.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:z.write(inventory_path,inventory_path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:
        if hashlib.sha256(z.read(inventory_path.relative_to(ROOT).as_posix())).hexdigest()!=sha(inventory_path):raise ValueError('Inventory archive bytes differ')
    result=dict(full_artifact_json_files_scanned=inventory['json_files_scanned'],parse_errors=0,excluded_json_files=0,
        allocation_candidate_files=inventory['allocation_candidate_files'],predecessor_terminal_verified=True,
        old_withdrawn_alpha_retained=.025,old_completed_alpha_retained=.05/6,
        numeric_attempt3_or_later_in_inventory=False,new_attempt_reserved=None,new_error_allocated=False,
        historical_cross_family_fwer_claim=False,
        inventory_archive_sha256=sha(archive),inventory_json_sha256=sha(inventory_path),evidence_sha256=evidence,
        inference='No attempt3 allocation found in complete project JSON inventory;old terminal/hash chains checked. This audit itself neither reserves nor spends alpha;legacy analysis levels/copies are not new reservations.')
    with (out/'verification.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(json_files=inventory['json_files_scanned'],predecessor_verified=True,new_error_allocated=False,archive_bytes=archive.stat().st_size)),flush=True)


if __name__=='__main__':verify()
