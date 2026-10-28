"""Verify advertised remote commit and the bytes of this withdrawal delivery only."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
CHECKPOINT=ROOT/'.git/coopgate-withdrawal-archive-transfer-20261027.json'
OUTPUT=ROOT/'artifacts/legacy_withdrawal_publication_20261027.json'
PREFIXES=('artifacts/legacy_withdrawal_20261027',
    'artifacts/proposals/legacy_attempt_withdrawal_20261027',
    'tools/legacy_withdrawal_contract_20261027.py',
    'tools/execute_joint_risk_after_withdrawal_20261027.py',
    'tests/test_legacy_withdrawal_20261027.py',
    'operations/withdraw_legacy_processes_20261027.ps1',
    'operations/continue_after_withdrawal_20261027.ps1',
    'docs/legacy_withdrawal_20261027.md','README.md')


def git(*args):
    return subprocess.check_output(['git','-c','maintenance.auto=false','-c','gc.auto=0',
        '-c','http.proxy=http://127.0.0.1:7890','-c','http.version=HTTP/1.1',*args],cwd=ROOT).decode().strip()


def main():
    if OUTPUT.exists():raise FileExistsError('Preserve first delivery verification')
    state=json.loads(CHECKPOINT.read_text())
    if (not state['main_verified'] or not state['temporary_ref_removed']
            or state['main_history_rewritten'] or not all(p['pushed'] for p in state['parts'])):
        raise ValueError('Full unchanged Git archive transport must finish first')
    target=state['target']
    remote=git('ls-remote','--refs','origin','refs/heads/main').split()[0]
    if remote!=target:raise ValueError('Advertised remote main differs; inspect before claiming delivery')
    entries=[]
    for line in git('ls-tree','-r','-l',target,'--',*PREFIXES).splitlines():
        metadata,name=line.split('\t',1)
        mode,kind,blob,size=metadata.split()
        if kind!='blob' or git('hash-object','--',name)!=blob or (ROOT/name).stat().st_size!=int(size):
            raise ValueError('Delivered Git blob differs from local bytes: '+name)
        entries.append(dict(path=name,git_blob=blob,bytes=int(size)))
    inventory_path=ROOT/'artifacts/legacy_withdrawal_20261027/retained_calibration/inventory.json'
    inventory=json.loads(inventory_path.read_text())
    delivered={item['path'] for item in entries}
    if len(inventory['archives'])!=17 or any(a['path'] not in delivered for a in inventory['archives']):
        raise ValueError('Missing retained archive in advertised commit')
    report=dict(advertised_remote_main=target,scope=list(PREFIXES),files=entries,
        verified_files=len(entries),verified_bytes=sum(e['bytes'] for e in entries),
        local_bytes_match_advertised_git_blobs=True,main_history_rewritten=False,
        temporary_transport_ref_removed=True,archive_parts=17,
        original_files_retained=len(inventory['files']),legacy_complete_initials=inventory['completed_initials'],
        legacy_status='withdrawn incomplete; no certificate',new_joint_required_runs=16384,
        inventory_sha256=hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
        statistical_analysis_performed=False,new_sumo_runs=0)
    with OUTPUT.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({k:report[k] for k in ('advertised_remote_main','verified_files','verified_bytes','archive_parts')}))


if __name__=='__main__':main()
