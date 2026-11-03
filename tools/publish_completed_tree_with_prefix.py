"""Publish a committed completed-artifact tree using bounded Git prefix deltas.

Same verified transport as the full attempt4 delivery. Fast-forwards only,
private indexes,exact final original tree,protected dirty files untouched.
Do not change real HEAD/index or commit while an active transfer is running.
"""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=None
TRANSFER=None
ORIGINAL=None
BASE=None
EXPECTED_REMOTE="https://github.com/Leo-7GOAT/CoopGate.git"


def run():
    def git(*args,env=None,data=None):return subprocess.check_output(['git',*args],cwd=ROOT,env=env,input=data).decode().strip()
    if OUT.parent!=ROOT/'.git':raise ValueError('Explicit direct Git journal required')
    if git('remote','get-url','origin')!=EXPECTED_REMOTE:raise ValueError('Remote identity changed')
    original=ORIGINAL;base=BASE;tree=git('rev-parse',original+'^{tree}')
    backup='refs/codex/original-'+OUT.name
    present=subprocess.run(['git','rev-parse','--verify',backup],cwd=ROOT,capture_output=True,text=True)
    if present.returncode==0 and present.stdout.strip()!=original:raise ValueError('Original backup ref differs')
    if present.returncode!=0:git('update-ref',backup,original,'')
    if git('rev-parse','HEAD')!=original or git('write-tree')!=tree:raise ValueError('Original HEAD/index changed')
    remote=git('ls-remote','origin','refs/heads/main').split()[0]
    found=git('ls-remote','origin',TRANSFER);existing_transfer=found.split()[0] if found else ''
    if OUT.exists():
        witness=json.loads((OUT/'failed_owner.json').read_bytes());state=json.loads((OUT/'state.json').read_bytes())
        if witness['actual_native_exit_code']==0 or state['completed'] or any(e['actual_native_exit_code'] is None for e in state['events']):raise ValueError('No failed root witness or pending Git child')
        if state['original_commit']!=original or state['base']!=base:raise ValueError('Different transfer request')
        last=next((e['commit'] for e in reversed(state['events']) if e['ref']==TRANSFER and e.get('remote_head')==e['commit']),None)
        if existing_transfer!=last:raise ValueError('Unexpected auxiliary ref')
        state['resumed_root_native_pid']=os.getpid()
    else:
        if existing_transfer or remote!=base:raise ValueError('Expected untouched transfer branch/main base')
        OUT.mkdir(exist_ok=False)
        state=dict(root_native_pid=os.getpid(),original_commit=original,original_tree=tree,base=base,parts=[],events=[],completed=False,
            force_push_used=False,auxiliary_branch=TRANSFER,transport_blob_path='.delivery-transfer/current_blob.bin')
    prior=state
    if remote not in [base]+[p['commit'] for p in state['parts']]:raise ValueError('Foreign main advance;never force')
    def save():
        p=OUT/'state.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(OUT/'state.json')
    save()
    env=dict(os.environ,GIT_INDEX_FILE=str(OUT/'main_index'),GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='Never')
    transport_env=dict(env,GIT_INDEX_FILE=str(OUT/'transfer_index'))
    entries={}
    for row in subprocess.check_output(['git','ls-tree','-r','-z',original],cwd=ROOT).split(b'\0'):
        if row:
            header,name=row.split(b'\t',1);mode,_,digest=header.decode().split();entries[name.decode()]=(mode,digest)
    changed=[n for n in subprocess.check_output(['git','diff','--name-only','-z',base,original],cwd=ROOT).decode().split('\0') if n]
    zips=sorted(n for n in changed if n.endswith('.zip'));batches=[[n] for n in zips]+[[n for n in changed if n not in zips]]
    git('read-tree',base,env=env);git('read-tree',existing_transfer or remote,env=transport_env)
    previous=base;transport=existing_transfer or remote;transfer_remote=existing_transfer
    protected=('operations/continue_joint_recovery_20261028.ps1','tests/test_execute_prospective_v22.py','tools/execute_prospective_v22.py')
    dirty={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in protected}
    def push(commit,ref,label):
        nonlocal remote,transfer_remote
        previous_attempts=sum(e['label']==label for e in state['events'])
        for attempt in range(previous_attempts+1,previous_attempts+4):
            log=OUT/f'{label}_{attempt}.log'
            with log.open('xb') as f:
                child=subprocess.Popen(['git','-c','gc.auto=0','-c','maintenance.auto=false','-c','pack.threads=1','-c','http.version=HTTP/1.1','-c','http.postBuffer=1048576',
                    '-c','pack.window=50','-c','pack.depth=50','push','--porcelain','origin',commit+':'+ref],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
                event=dict(label=label,native_pid=child.pid,actual_native_exit_code=None,commit=commit,ref=ref,log=log.name)
                state['events'].append(event);save();event['actual_native_exit_code']=child.wait();save()
            found=git('ls-remote','origin',ref);head=found.split()[0] if found else ''
            event['remote_head']=head;save()
            if head==commit:
                if ref==TRANSFER:transfer_remote=head
                else:remote=head
                return
            expected=transfer_remote if ref==TRANSFER else previous
            if head!=expected:raise ValueError('Foreign advance;never force')
        raise RuntimeError('Three bounded-delta transfers failed;preserve journal')
    for index,names in enumerate(batches):
        data=b''.join((entries[n][0]+' '+entries[n][1]+'\t'+n+'\0').encode() for n in names)
        git('update-index','-z','--index-info',env=env,data=data);current=git('write-tree',env=env)
        if index<len(prior['parts']):
            part=prior['parts'][index];commit=part['commit']
            if part['files']!=names or part['tree']!=current or git('rev-parse',commit+'^')!=previous:raise ValueError('Prior transport tree differs')
        else:
            commit=git('commit-tree',current,'-p',previous,data=f'Publish completed MAPPO data part {index+1}/{len(batches)}\nOriginal {original};exact tree {tree}.\n'.encode())
            prior['parts'].append(dict(files=names,tree=current,commit=commit));save()
        if subprocess.run(['git','merge-base','--is-ancestor',commit,remote],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:
            previous=commit;continue
        if remote!=previous:raise ValueError('Main is not exact prior verified parent')
        for name in names:
            if not name.endswith('.zip'):continue
            raw=(ROOT/name).read_bytes();step=2*1024*1024;start=min(4*1024*1024,len(raw))
            present=subprocess.run(['git','rev-parse',transport+':'+state['transport_blob_path']],cwd=ROOT,capture_output=True,text=True)
            if present.returncode==0:
                existing_blob=present.stdout.strip();size=int(git('cat-file','-s',existing_blob))
                if size<=len(raw) and git('hash-object','--stdin',data=raw[:size])==existing_blob:
                    start=size+step
                    if size==len(raw):blob=existing_blob
            for length in range(start,len(raw)+step,step):
                prefix=raw[:length];blob=git('hash-object','-w','--stdin',data=prefix)
                present=subprocess.run(['git','rev-parse',transport+':'+state['transport_blob_path']],cwd=ROOT,capture_output=True,text=True)
                if present.returncode==0 and present.stdout.strip()==blob:
                    continue
                git('update-index','--add','--cacheinfo','100644',blob,state['transport_blob_path'],env=transport_env)
                ttree=git('write-tree',env=transport_env)
                tcommit=git('commit-tree',ttree,'-p',transport,data=f'Transport {name} prefix {len(prefix)}/{len(raw)}\n'.encode())
                # Reject a surprising full-blob pack before attempting network.
                packed=subprocess.check_output(['git','-c','pack.threads=1','-c','pack.window=50','-c','pack.depth=50','pack-objects','--thin','--stdout','--revs'],cwd=ROOT,input=f'{tcommit}\n^{transport}\n'.encode())
                if len(packed)>6*1024*1024:raise ValueError('Delta exceeds6MiB;no oversized push started')
                push(tcommit,TRANSFER,f'part{index+1:03d}_prefix{len(prefix):09d}');transport=tcommit
                print(json.dumps(dict(part=index+1,prefix_bytes=len(prefix),blob_bytes=len(raw),pack_bytes=len(packed))),flush=True)
            if blob!=entries[name][1]:raise ValueError('Full transported archive blob differs')
        push(commit,'refs/heads/main',f'part{index+1:03d}_main');previous=commit
        state['main_part_confirmed']=index+1;save()
    if current!=tree or git('rev-parse',previous+'^{tree}')!=tree or git('write-tree')!=tree:raise ValueError('Final/index tree mismatch')
    if any(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h for p,h in dirty.items()):raise ValueError('Historical dirty file changed')
    git('update-ref','refs/heads/main',previous,original)
    state.update(completed=True,published_head=previous,final_tree_identical=True,protected_dirty_files_unchanged=True);save()
    print(json.dumps(dict(completed=True,published_head=previous,original_tree_identical=True)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--original',required=True);p.add_argument('--base',required=True);p.add_argument('--journal',required=True);p.add_argument('--transfer-ref',required=True);a=p.parse_args()
    ORIGINAL=a.original;BASE=a.base;OUT=Path(a.journal).resolve();TRANSFER=a.transfer_ref
    if not TRANSFER.startswith('refs/heads/data-transfer-'):raise ValueError('Explicit transport branch required')
    run()
