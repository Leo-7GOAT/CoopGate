"""Publish a small source-only branch while old raw-data transport continues.

Private index and a new remote branch preserve main,HEAD,the real index,and
every dirty working file. Every prospective source blob is verified in Git.
"""
import json,os,subprocess
from pathlib import Path
import mappo_history_formal_experiment as e


def run():
    root=e.ROOT;out=root/'.git/history-source-publication-20261103';out.mkdir(exist_ok=False)
    def git(*args,env=None,data=None):return subprocess.check_output(['git',*args],cwd=root,env=env,input=data).decode().strip()
    if git('remote','get-url','origin')!='https://github.com/Leo-7GOAT/CoopGate.git':raise ValueError('Remote changed')
    branch='refs/heads/history-gate-source-20261103'
    if git('ls-remote','origin',branch):raise ValueError('Exclusive source branch already exists')
    head=git('rev-parse','HEAD');index=git('write-tree');base=git('ls-remote','origin','refs/heads/main').split()[0]
    env=dict(os.environ,GIT_INDEX_FILE=str(out/'index'),GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='Never')
    git('read-tree',base,env=env)
    paths=set(e.source_files())
    extra=['artifacts/mappo_ctde_20261101/observed_history_fixed_candidate_20261103.json',
      'artifacts/mappo_ctde_20261101/history_pre_registration_source_review_20261103.json',
      'artifacts/mappo_ctde_20261101/history_formal_adapter_generation_20261103.json',
      'artifacts/mappo_ctde_20261101/mappo_attempt5_predecessor_audit_20261103/verification.json',
      'artifacts/mappo_ctde_20261101/mappo_attempt5_predecessor_audit_20261103/owner_native_exit.json',
      'tools/validate_observed_history_gate.py','tools/validate_strict_phase_ablation.py','tests/test_observed_lag_bias_fit.py',
      'docs/observed_history_development_20261103_protocol.md','docs/observed_history_20261103_results.md',
      'docs/strict_phase_ablation_20261103_protocol.md','docs/lag_inverse_identifiability_20261103.md']
    for directory in ('observed_history_validation_20261103','strict_phase_ablation_20261103'):
        extra += ['artifacts/mappo_ctde_20261101/development/'+directory+'/'+n for n in ('plan.json','report.json','owner_native_exit.json')]
    paths.update(root/p for p in extra)
    manifest={}
    for p in sorted(paths):
        name=p.relative_to(root).as_posix();blob=git('hash-object','-w','--stdin',data=p.read_bytes())
        git('update-index','--add','--cacheinfo','100644',blob,name,env=env);manifest[name]=e.sha(p)
    tree=git('write-tree',env=env)
    commit=git('commit-tree',tree,'-p',base,data=b'Freeze observed-history MAPPO source before independent validation\nSource-only preparation;no attempt5 allocation or formal samples.\n')
    for name,digest in manifest.items():
        import hashlib
        if hashlib.sha256(subprocess.check_output(['git','show',commit+':'+name],cwd=root)).hexdigest()!=digest:
            raise ValueError('Committed source bytes differ')
    state=dict(root_native_pid=os.getpid(),commit=commit,tree=tree,parent_public_main=base,branch=branch,
        source_sha256=manifest,changes_real_head=False,changes_real_index=False,new_error_allocated=False,attempts=[])
    e.write(out/'prepared.json',state)
    for attempt in range(3):
        with (out/f'push{attempt+1}.log').open('xb') as f:
            child=subprocess.Popen(['git','-c','http.version=HTTP/1.1','push','--porcelain','origin',commit+':'+branch],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
            code=child.wait()
        row=dict(native_pid=child.pid,actual_native_exit_code=code)
        state['attempts'].append(row)
        remote=git('ls-remote','origin',branch);row['remote_head']=remote.split()[0] if remote else None
        if row['remote_head']==commit:break
        if row['remote_head'] is not None:raise ValueError('Foreign source-branch advance')
    state.update(remote_verified=row['remote_head']==commit,real_head_unchanged=git('rev-parse','HEAD')==head,real_index_unchanged=git('write-tree')==index)
    e.write(out/'receipt.json',state)
    if not all(state[k] for k in ('remote_verified','real_head_unchanged','real_index_unchanged')):raise ValueError('Publication/unchanged-state check failed')
    print(json.dumps(dict(source_commit=commit,remote_branch=branch,source_files=len(paths),head_index_unchanged=True)),flush=True)


if __name__=='__main__':run()
