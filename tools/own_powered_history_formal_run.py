"""Own the native formal parent and persist its ACTUAL wait result.

Public registration is verified before launching. Never infer a successful
native exit from file presence or process absence. Exclusive launch witness
prevents silent duplication across an interrupted desktop tool session.
"""
from datetime import datetime,timezone
import json,os,subprocess,sys
from pathlib import Path
import mappo_powered_history_formal_experiment as e
from mappo_native_launch import command


def run():
    e.bound(e.FORMAL,False)
    registration=e.read(e.FORMAL/'public_registration_receipt.json');commit=registration['registered_commit']
    if not registration['remote_verified']:raise ValueError('Registration not public')
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=e.ROOT,text=True).split()[0]
    if subprocess.run(['git','merge-base','--is-ancestor',commit,remote],cwd=e.ROOT).returncode:
        raise ValueError('Public registration is not in remote main history')
    import hashlib
    for name,digest in registration['registered_files_sha256'].items():
        raw=subprocess.check_output(['git','show',commit+':'+name],cwd=e.ROOT)
        if hashlib.sha256(raw).hexdigest()!=digest or e.sha(e.ROOT/name)!=digest:raise ValueError('Public registration bytes changed')
    launch=e.FORMAL/'owner_launch.json'
    if launch.exists() or (e.FORMAL/'execution').exists():raise ValueError('Formal owner already launched;never duplicate')
    args=command(e.ROOT/'tools/run_mappo_powered_history_formal.py','run')
    child=subprocess.Popen(args,cwd=e.ROOT)
    e.write(launch,dict(native_pid=child.pid,wrapper_native_pid=os.getpid(),command=args,
        started_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=e.sha(e.FORMAL/'plan.json'),
        registered_commit=commit,wrapper_sha256=e.sha(__file__)))
    print(json.dumps(dict(native_formal_parent_pid=child.pid,registered_commit=commit)),flush=True)
    code=child.wait()
    terminal=dict(native_pid=child.pid,actual_native_exit_code=code,finished_utc=datetime.now(timezone.utc).isoformat(),
        plan_sha256=e.sha(e.FORMAL/'plan.json'),registered_commit=commit,observed_by_direct_native_wait=True)
    e.write(e.FORMAL/'actual_root_native_terminal.json',terminal)
    if code==0:
        terminal['completion_receipt_sha256']=e.sha(e.FORMAL/'completion_receipt.json')
        e.write(e.FORMAL/'orchestrator_native_receipt.json',terminal)
    print(json.dumps(terminal),flush=True)
    raise SystemExit(code)


if __name__=='__main__':run()
