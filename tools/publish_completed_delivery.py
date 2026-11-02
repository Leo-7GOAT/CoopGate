"""Fast-forward an exact committed delivery tree in bounded Git transfers.

Intended for completed large raw-study archives. No force push,no checkout,
no real-index change. Original consolidated commit and every dirty byte are
retained. A failed journal is never silently restarted. Every Git push has an
owned native handle and separately verified remote ref.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def publish(repo,base,original,journal_name,expected_remote):
    repo=Path(repo).resolve()
    if not re.fullmatch('[a-z0-9_-]+',journal_name):raise ValueError('Simple exclusive journal name required')
    def git(*args,env=None,data=None):
        return subprocess.check_output(['git',*args],cwd=repo,env=env,input=data).decode('utf-8').strip()
    def remote_head():return git('ls-remote','origin','refs/heads/main').split()[0]
    original=git('rev-parse',original);base=git('rev-parse',base);tree=git('rev-parse',original+'^{tree}')
    if (git('symbolic-ref','HEAD')!='refs/heads/main' or git('rev-parse','HEAD')!=original
        or git('write-tree')!=tree):raise ValueError('Exact committed main and matching real index required')
    if git('remote','get-url','origin')!=expected_remote:raise ValueError('Remote identity changed')
    if remote_head()!=base:raise ValueError('Remote advanced;never force or overwrite')
    subprocess.run(['git','merge-base','--is-ancestor',base,original],cwd=repo,check=True)
    metadata=Path(git('rev-parse','--absolute-git-dir')).resolve()
    if not metadata.is_relative_to(repo):raise ValueError('Git metadata must remain inside explicit repository')
    dirty=subprocess.check_output(['git','diff','--name-only','-z'],cwd=repo).decode().split('\0')
    def working_digest(name):
        path=repo/name
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    dirty_hash={p:working_digest(p) for p in dirty if p}
    changed=[p for p in subprocess.check_output(['git','diff','--name-only','-z',base,original],cwd=repo).decode().split('\0') if p]
    zips=sorted(p for p in changed if p.endswith('.zip'))
    if not zips:raise ValueError('Use a normal small push when no new ZIP archive exists')
    others=[p for p in changed if p not in zips];batches=[[p] for p in zips]+([others] if others else [])
    entries={}
    for row in subprocess.check_output(['git','ls-tree','-r','-z',original],cwd=repo).split(b'\0'):
        if not row:continue
        header,name=row.split(b'\t',1);mode,_,digest=header.decode().split();entries[name.decode()]=(mode,digest)
    journal=metadata/journal_name;journal.mkdir(exist_ok=False)
    state=dict(original_commit=original,original_tree=tree,remote_base=base,parts=[],completed=False,
        force_push_used=False,preexisting_working_hashes=dirty_hash,root_native_pid=os.getpid())
    def save():
        temporary=journal/'state.tmp';temporary.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8');temporary.replace(journal/'state.json')
    save();git('update-ref','refs/heads/backup-'+journal_name,original,'0'*40)
    env=os.environ.copy();env.update(GIT_INDEX_FILE=str(journal/'index'),GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='Never')
    git('read-tree',base,env=env);previous=base
    for index,names in enumerate(batches):
        data=b''
        for name in names:
            mode,digest=entries.get(name,('0','0'*40))
            data+=(mode+' '+digest+'\t'+name+'\0').encode()
        git('update-index','-z','--index-info',env=env,data=data);current=git('write-tree',env=env)
        if index==len(batches)-1 and current!=tree:raise ValueError('Final delivery tree must be byte-identical')
        commit=git('commit-tree',current,'-p',previous,data=f'Publish completed delivery transport {index+1}/{len(batches)}\n\nOriginal commit {original}; exact final tree {tree}.\n'.encode())
        part=dict(index=index+1,files=names,commit=commit,tree=current,attempts=[],remote_confirmed=False)
        state['parts'].append(part);save()
        for attempt in range(3):
            log=journal/f'part{index+1:03d}_attempt{attempt+1}.log'
            with log.open('xb') as output:
                child=subprocess.Popen(['git','-c','http.version=HTTP/1.1','-c','http.postBuffer=67108864',
                    'push','--porcelain','origin',commit+':refs/heads/main'],cwd=repo,env=env,stdout=output,stderr=subprocess.STDOUT)
                event=dict(native_pid=child.pid,actual_native_exit_code=None,log=log.name)
                part['attempts'].append(event);save()
                event['actual_native_exit_code']=child.wait();save()
            remote=remote_head();event['remote_head']=remote;save()
            if remote==commit:part['remote_confirmed']=True;save();break
            if remote!=previous:raise ValueError('Unexpected remote advance;all evidence retained,never force pushed')
        if not part['remote_confirmed']:raise RuntimeError('Three transfers failed;preserve journal and inspect exact terminal evidence before recovery')
        previous=commit
        print(json.dumps(dict(part=index+1,total=len(batches),remote_confirmed=True,
            actual_push_exit=part['attempts'][-1]['actual_native_exit_code'])),flush=True)
    if git('rev-parse',previous+'^{tree}')!=tree or remote_head()!=previous:raise ValueError('Final remote/tree differs')
    if any(working_digest(p)!=digest for p,digest in dirty_hash.items()):raise ValueError('Preexisting working bytes changed externally')
    if git('write-tree')!=tree:raise ValueError('Real index changed externally;do not advance local main')
    git('update-ref','refs/heads/main',previous,original)
    state.update(completed=True,published_head=previous,final_tree_identical=True,preexisting_changes_unchanged=True)
    save();print(json.dumps(dict(completed=True,published_head=previous,parts=len(batches),tree=tree)),flush=True)
    return state


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=ROOT)
    p.add_argument('--base',required=True);p.add_argument('--original',required=True);p.add_argument('--journal',required=True)
    p.add_argument('--expected-remote',default='https://github.com/Leo-7GOAT/CoopGate.git');a=p.parse_args()
    publish(a.repo,a.base,a.original,a.journal,a.expected_remote)
