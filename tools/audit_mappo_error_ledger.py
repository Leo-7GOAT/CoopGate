"""Full JSON-artifact allocation inventory; never reserves or spends alpha."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
KEYS={'attempt','attempt_id','attempt_alpha','allocated_error','alpha','total_attempt_error',
      'old_alpha','new_alpha','allocation_attempt','error_allocation_attempt_reserved'}


def audit():
    scope=(ROOT/'artifacts').resolve()
    paths=subprocess.check_output(['rg','--files',str(scope),'-g','*.json'],cwd=ROOT,text=True).splitlines()
    inventory=[];hits=[];errors=[]
    for index,raw in enumerate(paths):
        p=Path(raw)
        if not p.resolve().is_relative_to(scope):raise ValueError('Artifact resolves outside the authorized project')
        data=p.read_bytes();identity=dict(path=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
        inventory.append(identity)
        try:obj=json.loads(data)
        except (ValueError,UnicodeDecodeError) as exc:
            errors.append(dict(**identity,error=str(exc)));continue
        found=[];stack=[('',obj)]
        while stack:
            location,value=stack.pop()
            if isinstance(value,dict):
                for key,child in value.items():
                    label=location+'/'+key
                    if key.lower() in KEYS and type(child) in (int,float,str):found.append(dict(key_path=label,value=child))
                    if isinstance(child,(dict,list)):stack.append((label,child))
            elif isinstance(value,list):
                stack.extend((location+'/'+str(i),v) for i,v in enumerate(value) if isinstance(v,(dict,list)))
        if found:hits.append(dict(**identity,entries=found))
        if (index+1)%500==0:print(json.dumps(dict(json_files_scanned=index+1,allocation_candidate_files=len(hits))),flush=True)
    result=dict(json_files_scanned=len(inventory),filename_filter_used=False,size_exclusions=[],parse_errors=errors,
        allocation_candidate_files=hits,inventory=inventory,new_error_allocated=False,attempt_reserved=None,
        interpretation='Complete artifact JSON inventory only. Distinguish real allocations from tests,analysis levels and copies;verify predecessor receipts before reservation.')
    out=scope/'mappo_ctde_20261101/error_ledger_full_json_audit_20261102.json'
    with out.open('x',encoding='utf-8',newline='\n') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(scanned=len(inventory),candidate_files=len(hits),parse_errors=len(errors),attempts_ge3=[
        dict(path=h['path'],entry=e) for h in hits for e in h['entries']
        if e['key_path'].split('/')[-1].lower() in ('attempt','attempt_id','allocation_attempt','error_allocation_attempt_reserved')
        and isinstance(e['value'],(int,float)) and e['value']>=3])),flush=True)


if __name__=='__main__':audit()
