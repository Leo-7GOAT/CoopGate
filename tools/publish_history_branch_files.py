"""Fast-forward an explicit small preparation file list using a private index."""
import argparse,json,os,subprocess,hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BRANCH='refs/heads/history-gate-source-20261103'


def run(label,files,expected):
    if not label.replace('_','').isalnum():raise ValueError('Explicit simple journal label required')
    out=ROOT/'.git'/('history_branch_'+label);out.mkdir(exist_ok=False)
    def git(*args,env=None,data=None):return subprocess.check_output(['git',*args],cwd=ROOT,env=env,input=data).decode().strip()
    if git('remote','get-url','origin')!='https://github.com/Leo-7GOAT/CoopGate.git':raise ValueError('Remote changed')
    if git('ls-remote','origin',BRANCH).split()[0]!=expected:raise ValueError('Preparation branch changed')
    head,index=git('rev-parse','HEAD'),git('write-tree')
    env=dict(os.environ,GIT_INDEX_FILE=str(out/'index'),GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='Never')
    git('read-tree',expected,env=env);manifest={}
    for name in json.loads(Path(files).read_bytes()):
        p=(ROOT/name).resolve()
        if not p.is_relative_to(ROOT) or name in ('operations/continue_joint_recovery_20261028.ps1','tests/test_execute_prospective_v22.py','tools/execute_prospective_v22.py'):
            raise ValueError('Outside explicit source/data scope')
        data=p.read_bytes();blob=git('hash-object','-w','--stdin',data=data)
        git('update-index','--add','--cacheinfo','100644',blob,name,env=env);manifest[name]=hashlib.sha256(data).hexdigest()
    tree=git('write-tree',env=env);commit=git('commit-tree',tree,'-p',expected,data=f'Publish frozen MAPPO history {label}\nExplicit preparation files;preserve main and working state.\n'.encode())
    state=dict(root_native_pid=os.getpid(),source_branch=BRANCH,parent=expected,commit=commit,source_sha256=manifest,attempts=[],completed=False)
    def save():
        p=out/'receipt.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(out/'receipt.json')
    save()
    for attempt in range(3):
        with (out/f'push{attempt+1}.log').open('xb') as stream:
            child=subprocess.Popen(['git','-c','http.version=HTTP/1.1','push','--porcelain','origin',commit+':'+BRANCH],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            event=dict(native_pid=child.pid,actual_native_exit_code=None);state['attempts'].append(event);save()
            event['actual_native_exit_code']=child.wait();save()
        remote=git('ls-remote','origin',BRANCH).split()[0];event['remote_head']=remote;save()
        if remote==commit:break
        if remote!=expected:raise ValueError('Foreign remote advance;never force')
    if remote!=commit:raise RuntimeError('Publication failed;all terminal events retained')
    if git('rev-parse','HEAD')!=head or git('write-tree')!=index:raise ValueError('Real HEAD/index changed')
    state.update(completed=True,head_and_index_unchanged=True);save()
    print(json.dumps(dict(completed=True,commit=commit,files=len(manifest),head_and_index_unchanged=True)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--files',required=True);p.add_argument('--expected-base',required=True);a=p.parse_args();run(a.label,a.files,a.expected_base)
