"""Publish an identical delivery tree in small fast-forward Git transfers.

The consolidated local commit is retained. A private temporary Git index
constructs transport commits;the real index and working files are untouched.
Each push owns its child handle,and every delivered ref is read back remotely.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
BASE='f8a85d5352d3728fe25afae0fb6544471dee4e54'
ORIGINAL='be2b866580'


def git(*args,env=None,input=None):
    return subprocess.check_output(['git',*args],cwd=ROOT,env=env,input=input).decode('utf-8').strip()


def run():
    original=git('rev-parse',ORIGINAL);tree=git('rev-parse',original+'^{tree}')
    if git('rev-parse','HEAD')!=original or git('symbolic-ref','HEAD')!='refs/heads/main':raise ValueError('Expected exact retained delivery HEAD')
    if git('ls-remote','origin','refs/heads/main').split()[0]!=BASE:raise ValueError('Remote advanced;never force or overwrite it')
    metadata=Path(git('rev-parse','--absolute-git-dir')).resolve()
    if not metadata.is_relative_to(ROOT):raise ValueError('Unexpected Git metadata scope')
    journal=metadata/'mappo-chunk-publication-20261102';journal.mkdir(exist_ok=False)
    state=dict(original_commit=original,original_tree=tree,remote_base=BASE,parts=[],completed=False,force_push_used=False,working_files_modified=False)
    def save():
        temporary=journal/'state.tmp';temporary.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8');temporary.replace(journal/'state.json')
    save()
    git('update-ref','refs/heads/mappo-consolidated-delivery-20261102',original,'0'*40)
    env=os.environ.copy();env['GIT_INDEX_FILE']=str(journal/'index');env['GIT_TERMINAL_PROMPT']='0';env['GCM_INTERACTIVE']='Never'
    git('read-tree',BASE,env=env)
    entries={}
    raw=subprocess.check_output(['git','ls-tree','-r','-z',original],cwd=ROOT)
    for item in raw.split(b'\0'):
        if not item:continue
        header,name=item.split(b'\t',1);mode,kind,digest=header.decode().split();entries[name.decode('utf-8')]=(mode,digest)
    changed=subprocess.check_output(['git','diff','--name-only','-z',BASE,original],cwd=ROOT).decode('utf-8').split('\0')
    changed=[x for x in changed if x]
    if any(x not in entries for x in changed):raise ValueError('Deletion requires explicit separate handling')
    zips=sorted(x for x in changed if x.endswith('.zip'));others=[x for x in changed if x not in zips]
    if len(zips)!=19:raise ValueError('Exact nineteen completed data parts required')
    dirty=git('diff','--name-only','-z').split('\0');dirty=[x for x in dirty if x]
    dirty_hash={x:hashlib.sha256((ROOT/x).read_bytes()).hexdigest() for x in dirty}
    previous=BASE
    for index,files in enumerate([[x] for x in zips]+[others]):
        lines=b''.join((entries[name][0]+' '+entries[name][1]+'\t'+name+'\0').encode('utf-8') for name in files)
        git('update-index','-z','--index-info',env=env,input=lines)
        current_tree=git('write-tree',env=env)
        if index==len(zips) and current_tree!=tree:raise ValueError('Final delivery must have EXACTLY the consolidated tree')
        commit=git('commit-tree',current_tree,'-p',previous,input=f'Publish MAPPO formal delivery transport part {index+1:02d}/20\n\nConsolidated source commit {original};original tree {tree}.\n'.encode())
        part=dict(index=index+1,files=files,commit=commit,tree=current_tree,push_attempts=[],remote_confirmed=False)
        state['parts'].append(part);save()
        for attempt in range(3):
            log=journal/f'part{index+1:02d}_attempt{attempt+1}.log'
            with log.open('xb') as stream:
                child=subprocess.Popen(['git','-c','http.version=HTTP/1.1','-c','http.postBuffer=67108864','push','--porcelain','origin',commit+':refs/heads/main'],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                code=child.wait()
            remote=git('ls-remote','origin','refs/heads/main').split()[0]
            part['push_attempts'].append(dict(native_pid=child.pid,actual_native_exit_code=code,remote_head=remote,log=log.name));save()
            if remote==commit:part['remote_confirmed']=True;save();break
            if remote!=previous:raise ValueError('Unexpected remote advance;retained state,never force pushed')
        if not part['remote_confirmed']:raise RuntimeError('Transport still failed;all original files and partial publication journal retained')
        git('update-ref','refs/heads/mappo-chunk-publication-20261102',commit)
        previous=commit;print(json.dumps(dict(part=index+1,total=20,remote_confirmed=True,native_exit=part['push_attempts'][-1]['actual_native_exit_code'],commit=commit)),flush=True)
    if git('rev-parse',previous+'^{tree}')!=tree:raise ValueError('Published tree changed')
    if any(hashlib.sha256((ROOT/x).read_bytes()).hexdigest()!=digest for x,digest in dirty_hash.items()):raise ValueError('Preexisting working change modified externally')
    git('update-ref','refs/heads/main',previous,original)
    state.update(completed=True,published_head=previous,all_parts_remote_confirmed=True,final_tree_identical=True,preexisting_changes_unchanged=True)
    save();print(json.dumps({k:v for k,v in state.items() if k!='parts'}),flush=True)


if __name__=='__main__':run()
