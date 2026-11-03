"""Upload identical large blobs in small Git deltas over an auxiliary branch.

The final main tree is EXACTLY the original completed delivery. Archives are
never recompressed or split in that tree. Private index; fast-forwards only.
The earlier resumer's root exit is unknown; its three final Git children have
actual failed exits and its code raises immediately after those failures.
"""
import hashlib,json,os,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'.git/mappo4-complete-publication-20261103'
OUT=ROOT/'.git/mappo4-prefix-transfer-20261103'
TRANSFER='refs/heads/data-transfer-mappo4-20261103'


def run():
    def git(*args,env=None,data=None):
        return subprocess.check_output(['git',*args],cwd=ROOT,env=env,input=data).decode().strip()
    prior=json.loads((OLD/'state.json').read_bytes())
    if prior['completed'] or any(e['actual_native_exit_code'] is None for p in prior['parts'] for e in p['attempts']):
        raise ValueError('Earlier completed or unknown live Git operation')
    last=prior['parts'][-1]
    if last['remote_confirmed'] or len(last['attempts'])!=3 or any(e['actual_native_exit_code']!=1 for e in last['attempts']):
        raise ValueError('Require all three actually failed final transfers')
    for event in last['attempts']:
        r=subprocess.run(['powershell','-NoProfile','-Command',f'if (Get-Process -Id {int(event["native_pid"])} -ErrorAction SilentlyContinue) {{exit 1}}'],capture_output=True)
        if r.returncode:raise ValueError('Explicit earlier Git PID still exists')
    original=prior['original_commit'];tree=prior['original_tree'];base=prior['remote_base']
    if git('rev-parse','HEAD')!=original or git('write-tree')!=tree:raise ValueError('Original HEAD/index changed')
    if git('remote','get-url','origin')!='https://github.com/Leo-7GOAT/CoopGate.git':raise ValueError('Remote changed')
    remote=git('ls-remote','origin','refs/heads/main').split()[0]
    if remote!=prior['parts'][-2]['commit']:raise ValueError('Unexpected remote main')
    found=git('ls-remote','origin',TRANSFER);existing_transfer=found.split()[0] if found else ''
    if OUT.exists():
        witness=json.loads((OUT/'failed_owner.json').read_bytes())
        state=json.loads((OUT/'state.json').read_bytes())
        if witness['actual_native_exit_code']==0 or state['completed'] or any(e['actual_native_exit_code'] is None for e in state['events']):
            raise ValueError('No observed failed root or unknown Git child')
        last_transfer=next(e for e in reversed(state['events']) if e['ref']==TRANSFER and e.get('remote_head')==e['commit'])
        if existing_transfer!=last_transfer['commit']:raise ValueError('Unexpected auxiliary branch advance')
        state['resumed_root_native_pid']=os.getpid()
    else:
        if existing_transfer:raise ValueError('Exclusive auxiliary branch already exists')
        OUT.mkdir(exist_ok=False)
        state=dict(root_native_pid=os.getpid(),prior_resumer_root_native_exit_code=None,
        prior_last_git_actual_exits=[e['actual_native_exit_code'] for e in last['attempts']],
        original_commit=original,original_tree=tree,base=base,events=[],completed=False,force_push_used=False,
        auxiliary_branch=TRANSFER,transport_blob_path='.delivery-transfer/current_blob.bin')
    def save():
        p=OUT/'state.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(OUT/'state.json')
    save()
    env=os.environ.copy();env.update(GIT_INDEX_FILE=str(OUT/'main_index'),GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='Never')
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
                child=subprocess.Popen(['git','-c','http.version=HTTP/1.1','-c','http.postBuffer=1048576',
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
        else:commit=git('commit-tree',current,'-p',previous,data=f'Publish completed MAPPO delivery part {index+1}/{len(batches)}\nOriginal {original};exact tree {tree}.\n'.encode())
        if subprocess.run(['git','merge-base','--is-ancestor',commit,remote],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:
            previous=commit;continue
        if remote!=previous:raise ValueError('Main is not exact prior verified parent')
        for name in names:
            if not name.endswith('.zip'):continue
            raw=(ROOT/name).read_bytes();step=2*1024*1024;start=4*1024*1024
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
                packed=subprocess.check_output(['git','-c','pack.window=50','-c','pack.depth=50','pack-objects','--thin','--stdout','--revs'],cwd=ROOT,input=f'{tcommit}\n^{transport}\n'.encode())
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


if __name__=='__main__':run()
