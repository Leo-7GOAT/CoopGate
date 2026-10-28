"""Byte-check the committed interruption recovery delivery, excluding live data."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'artifacts/joint_recovery_publication_20261028.json'
PREFIXES=('tools/joint_recovery_20261028.py','tests/test_joint_recovery_20261028.py',
    'operations/continue_joint_recovery_20261028.ps1',
    'operations/launch_joint_recovery_task_20261028.ps1',
    'operations/verify_joint_recovery_publication_20261028.py',
    'docs/joint_recovery_20261028.md','README.md',
    'artifacts/joint_risk_recovery_20261028/interruption',
    'artifacts/joint_risk_recovery_20261028/recovery.json',
    'artifacts/joint_risk_recovery_20261028/scheduler_execution.json',
    'artifacts/joint_risk_recovery_20261028/boundary_proof')


def git(*args):
    return subprocess.check_output(['git','-c','maintenance.auto=false','-c','gc.auto=0',
        '-c','http.proxy=http://127.0.0.1:7890','-c','http.version=HTTP/1.1',*args],cwd=ROOT).decode().strip()


def main():
    if OUTPUT.exists():raise FileExistsError('Preserve first exact recovery delivery proof')
    target=git('rev-parse','HEAD')
    if git('ls-remote','--refs','origin','refs/heads/main').split()[0]!=target:
        raise ValueError('Advertised remote main is not current committed recovery')
    entries=[]
    for line in git('ls-tree','-r','-l',target,'--',*PREFIXES).splitlines():
        metadata,name=line.split('\t',1);mode,kind,blob,size=metadata.split()
        if (mode!='100644' or kind!='blob' or git('hash-object','--no-filters','--',name)!=blob
                or (ROOT/name).stat().st_size!=int(size)):
            raise ValueError('Exact raw bytes differ: '+name)
        entries.append(dict(path=name,git_blob=blob,bytes=int(size)))
    plan_path=ROOT/'artifacts/joint_risk_recovery_20261028/recovery.json'
    plan=json.loads(plan_path.read_text())
    proof=json.loads((ROOT/'artifacts/joint_risk_recovery_20261028/boundary_proof/00225.json').read_text())
    if (not proof['all_non_timing_fields_exact'] or proof['retained_partial_rows']!=3
            or plan['cumulative_unique_sumo_runs']!=16384 or plan['retained_complete_runs']!=900):
        raise ValueError('Recovery scope or reproduced boundary differs')
    result=dict(utc=datetime.now(timezone.utc).isoformat(),advertised_remote_main=target,
        files=entries,verified_files=len(entries),verified_bytes=sum(e['bytes'] for e in entries),
        raw_local_bytes_match_advertised_git_blobs=True,live_collection_files_included=False,
        recovery_plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        original_native_exit='unknown',complete_runs_reused=900,boundary_rows_reproduced_exactly=3,
        required_unique_runs=16384,statistical_validation_complete=False,new_statistical_analysis=False)
    with OUTPUT.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({k:result[k] for k in ('advertised_remote_main','verified_files','verified_bytes')}))


if __name__=='__main__':main()
