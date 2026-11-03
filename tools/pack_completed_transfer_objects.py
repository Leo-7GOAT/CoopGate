"""Pack only the explicitly completed transfer graph,then prune packed duplicates.

No working file,SQLite,artifact,unique Git object,or unrelated process is
removed. Do not run global garbage collection or recursive filesystem cleanup.
"""
import argparse,hashlib,json,os,re,shutil,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def run(journal,out):
    journal=Path(journal).resolve();out=Path(out).resolve()
    if journal.parent!=ROOT/'.git' or out.parent!=ROOT/'artifacts/mappo_ctde_20261101':raise ValueError('Explicit owned scope required')
    state=json.loads((journal/'state.json').read_bytes());owner=json.loads((journal/'owner_native_exit.json').read_bytes())
    if not state['completed'] or owner['actual_native_exit_code']!=0 or owner['state_sha256']!=hashlib.sha256((journal/'state.json').read_bytes()).hexdigest():
        raise ValueError('Observed complete data transfer required')
    tip=next(e['commit'] for e in reversed(state['events']) if e['ref']==state['auxiliary_branch'] and e.get('remote_head')==e['commit'])
    base=state['base'];out.mkdir(exist_ok=False)
    def git(*args,data=None):return subprocess.check_output(['git',*args],cwd=ROOT,input=data).decode().strip()
    objects=[line.split()[0] for line in git('rev-list','--objects',tip,'^'+base).splitlines()]
    if not objects or any(not re.fullmatch('[0-9a-f]{40}',v) for v in objects):raise ValueError('Invalid owned object graph')
    loose=lambda:sum((ROOT/'.git/objects'/v[:2]/v[2:]).is_file() for v in objects)
    before=shutil.disk_usage(ROOT).free;loose_before=loose()
    args=['git','-c','pack.threads=1','pack-objects','--revs','--window=50','--depth=50',str(ROOT/'.git/objects/pack/pack')]
    with (out/'pack.log').open('xb') as log:
        child=subprocess.Popen(args,cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log)
        raw,_=child.communicate(f'{tip}\n^{base}\n'.encode());code=child.returncode
    if code:raise RuntimeError('Owned pack failed;all original objects retained')
    digest=raw.decode().strip()
    if not re.fullmatch('[0-9a-f]{40}',digest):raise ValueError('Unexpected pack result')
    index=ROOT/f'.git/objects/pack/pack-{digest}.idx';pack=index.with_suffix('.pack')
    verified=git('verify-pack','-v',str(index));packed={line.split()[0] for line in verified.splitlines() if re.match('^[0-9a-f]{40} ',line)}
    if not set(objects)<=packed:raise ValueError('Every owned object must be verified in the new complete pack before pruning')
    prune=subprocess.run(['git','prune-packed'],cwd=ROOT,capture_output=True,text=True)
    if prune.returncode:raise RuntimeError('Packed-duplicate pruning failed;unique originals remain')
    checked=git('cat-file','--batch-check',data=('\n'.join(objects)+'\n').encode()).splitlines()
    if len(checked)!=len(objects) or any('missing' in line for line in checked):raise ValueError('Owned Git graph unavailable after packing')
    result=dict(completed=True,actual_pack_native_pid=child.pid,actual_pack_native_exit_code=code,
        actual_prune_packed_native_exit_code=prune.returncode,transfer_tip=tip,excluded_original_base=base,
        verified_owned_objects=len(objects),loose_owned_objects_before=loose_before,loose_owned_objects_after=loose(),
        new_pack_bytes=pack.stat().st_size,space_recovered_bytes=shutil.disk_usage(ROOT).free-before,
        every_owned_object_readable=True,only_already_packed_duplicate_objects_pruned=True,
        working_files_modified=False,global_garbage_collection=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--journal',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.journal,a.out)
