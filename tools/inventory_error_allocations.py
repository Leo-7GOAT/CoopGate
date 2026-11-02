"""Read-only inventory of existing artifact metadata; allocates no error."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
KEYS={'attempt','attempt_id','alpha','attempt_alpha','allocated_error','old_alpha','new_alpha','total_error','total_attempt_error'}


def inventory():
    scope=ROOT/'artifacts'
    paths=subprocess.check_output(['rg','--files',str(scope),'-g','*.json'],cwd=ROOT,text=True).splitlines()
    records=[];skipped=[];scanned=0
    def walk(value,location,found):
        if isinstance(value,dict):
            for key,child in value.items():
                path=location+'/'+key
                if key.lower() in KEYS and type(child) in (int,float,str):found.append(dict(key_path=path,value=child))
                if isinstance(child,(dict,list)):walk(child,path,found)
        elif isinstance(value,list):
            for i,child in enumerate(value):
                if isinstance(child,(dict,list)):walk(child,location+'/'+str(i),found)
    for raw in paths:
        path=Path(raw)
        if not any(word in path.name.lower() for word in ('plan','contract','budget','allocation','amendment','authorization','completion','withdrawal')):continue
        if not path.resolve().is_relative_to(scope.resolve()):raise ValueError('Outside artifact scope')
        if path.stat().st_size>10*1024*1024:
            skipped.append(str(path.relative_to(ROOT)));continue
        data=path.read_bytes()
        try:value=json.loads(data)
        except (ValueError,UnicodeDecodeError):skipped.append(str(path.relative_to(ROOT)));continue
        scanned+=1;found=[];walk(value,'',found)
        if found:records.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(data).hexdigest(),entries=found))
    result=dict(metadata_files_scanned=scanned,records=records,skipped=skipped,
        new_error_allocated=False,unused_attempt_proved=False,
        limitation='Metadata inventory only; contexts and predecessor terminal receipts still need review. No alpha reset, reuse or new experiment is authorized by this output.')
    out=ROOT/'artifacts/mappo_ctde_20261101/error_allocation_inventory_20261102.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(scanned=scanned,metadata_with_allocations=len(records),skipped=len(skipped),new_error_allocated=False)),flush=True)


if __name__=='__main__':inventory()
