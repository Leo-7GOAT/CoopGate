"""Resume a terminal failed delivery journal with identical transport trees.

Uses HTTP/1.1 chunked transfer rather than the previous large post buffer.
Every old commit/tree/parent is rechecked;no force,no scientific file edit.
"""
import json,os,subprocess,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
JOURNAL=ROOT/'.git/mappo4-complete-publication-20261103'


def run():
    def git(*args,env=None,data=None):return subprocess.check_output(['git',*args],cwd=ROOT,env=env,input=data).decode().strip()
    state=json.loads((JOURNAL/'state.json').read_bytes());witness=json.loads((JOURNAL/'failed_owner.json').read_bytes())
    if witness['actual_native_exit_code']==0 or state['completed']:raise ValueError('Only observed terminal failed publication may resume')
    if any(x['actual_native_exit_code'] is None for p in state['parts'] for x in p['attempts']):raise ValueError('Unknown live Git child;do not duplicate')
    original=state['original_commit'];tree=state['original_tree'];base=state['remote_base']
    if git('rev-parse','HEAD')!=original or git('write-tree')!=tree:raise ValueError('Original committed tree/index changed')
    if git('remote','get-url','origin')!='https://github.com/Leo-7GOAT/CoopGate.git':raise ValueError('Remote identity changed')
    remote=git('ls-remote','origin','refs/heads/main').split()[0]
    if remote not in [base]+[p['commit'] for p in state['parts']]:raise ValueError('Unexpected remote advance')
    def save():
        p=JOURNAL/'resume_state.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(JOURNAL/'state.json')
    entries={}
    for row in subprocess.check_output(['git','ls-tree','-r','-z',original],cwd=ROOT).split(b'\0'):
        if row:
            header,name=row.split(b'\t',1);mode,_,digest=header.decode().split();entries[name.decode()]=(mode,digest)
    changed=[p for p in subprocess.check_output(['git','diff','--name-only','-z',base,original],cwd=ROOT).decode().split('\0') if p]
    zips=sorted(p for p in changed if p.endswith('.zip'));batches=[[p] for p in zips]+[[p for p in changed if p not in zips]]
    env=os.environ.copy();env.update(GIT_INDEX_FILE=str(JOURNAL/'resume_index'),GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='Never')
    if Path(env['GIT_INDEX_FILE']).exists():raise ValueError('Exclusive recovery index already exists;inspect it first')
    git('read-tree',base,env=env);previous=base
    for index,names in enumerate(batches):
        data=b''.join((entries[n][0]+' '+entries[n][1]+'\t'+n+'\0').encode() for n in names)
        git('update-index','-z','--index-info',env=env,data=data);current=git('write-tree',env=env)
        if index<len(state['parts']):
            part=state['parts'][index];commit=part['commit']
            if part['files']!=names or part['tree']!=current or git('rev-parse',commit+'^')!=previous:raise ValueError('Existing transfer differs from original plan')
        else:
            commit=git('commit-tree',current,'-p',previous,data=f'Publish completed MAPPO delivery part {index+1}/{len(batches)}\nOriginal {original};exact tree {tree}.\n'.encode())
            part=dict(index=index+1,files=names,commit=commit,tree=current,attempts=[],remote_confirmed=False);state['parts'].append(part);save()
        if subprocess.run(['git','merge-base','--is-ancestor',commit,remote],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:
            part['remote_confirmed']=True;save();previous=commit;continue
        if remote!=previous:raise ValueError('Remote is not the verified previous transport parent')
        for attempt in range(3):
            log=JOURNAL/f'resume_part{index+1:03d}_attempt{attempt+1}.log'
            with log.open('xb') as stream:
                child=subprocess.Popen(['git','-c','http.version=HTTP/1.1','-c','http.postBuffer=1048576',
                    'push','--progress','--porcelain','origin',commit+':refs/heads/main'],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                event=dict(native_pid=child.pid,actual_native_exit_code=None,log=log.name,transfer='HTTP1.1 chunked');part['attempts'].append(event);save()
                event['actual_native_exit_code']=child.wait();save()
            remote=git('ls-remote','origin','refs/heads/main').split()[0];event['remote_head']=remote;save()
            if remote==commit:part['remote_confirmed']=True;save();break
            if remote!=previous:raise ValueError('Foreign remote advance;never force')
        if not part['remote_confirmed']:raise RuntimeError('Chunked transfer failed;all terminal evidence retained')
        previous=commit;print(json.dumps(dict(part=index+1,total=len(batches),remote_confirmed=True)),flush=True)
    if current!=tree or git('rev-parse',previous+'^{tree}')!=tree:raise ValueError('Final tree not identical')
    for name,digest in state['preexisting_working_hashes'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Historical dirty file changed externally')
    if git('write-tree')!=tree:raise ValueError('Real index changed')
    git('update-ref','refs/heads/main',previous,original)
    state.update(completed=True,published_head=previous,final_tree_identical=True,preexisting_changes_unchanged=True)
    save();print(json.dumps(dict(completed=True,published_head=previous,parts=len(batches))),flush=True)


if __name__=='__main__':run()
