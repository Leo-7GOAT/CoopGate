"""Complete current artifact JSON inventory, reusing only verified parse proofs.

Every current JSON byte is hashed. Only files identical to the previous complete
zero-error inventory reuse its parsed key results. All new/changed files are
parsed without filename/size exclusions. This audits; it allocates no error.
"""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
import subprocess
import zipfile
from audit_mappo_error_ledger import KEYS
import mappo_formal_experiment as e
from plot_mappo_formal_results import complete_evidence

OUT=e.BASE/'mappo_attempt4_predecessor_audit_20261102'


def entries(obj):
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
    return found


def run():
    complete_evidence();e.bound(e.FORMAL,False)
    previous=e.read(e.BASE/'mappo_predecessor_budget_audit_20261102/verification.json')
    for path,digest in previous['evidence_sha256'].items():
        if e.sha(e.ROOT/path)!=digest:raise ValueError('Earlier terminal/allocation evidence changed: '+path)
    oldpath=e.BASE/'error_ledger_full_json_audit_20261102.json'
    if e.sha(oldpath)!=previous['inventory_json_sha256']:raise ValueError('Prior parse proof changed')
    old=e.read(oldpath)
    if old['parse_errors'] or old['size_exclusions'] or old['filename_filter_used']:raise ValueError('Invalid previous complete parse proof')
    known={x['path']:x for x in old['inventory']};knownhits={x['path']:x['entries'] for x in old['allocation_candidate_files']}
    scope=(e.ROOT/'artifacts').resolve()
    paths=subprocess.check_output(['rg','--files',str(scope),'-g','*.json'],cwd=e.ROOT,text=True).splitlines()
    inventory=[];hits=[];errors=[];reused=parsed=0;seen=set()
    @lru_cache(maxsize=32768)
    def resolved_parent(parent):
        resolved=parent.resolve()
        if not resolved.is_relative_to(scope):raise ValueError('Escaped project artifact scope')
        return resolved
    def inspect(raw):
        path=Path(raw);parent=resolved_parent(path.parent)
        actual=path.resolve() if path.is_symlink() else parent/path.name
        if not actual.is_relative_to(scope):raise ValueError('Escaped project artifact file')
        name=path.relative_to(e.ROOT).as_posix();data=path.read_bytes();digest=hashlib.sha256(data).hexdigest()
        identity=dict(path=name,sha256=digest,bytes=len(data))
        if name in known and known[name]['sha256']==digest:
            return identity,knownhits.get(name,[]),None,True
        else:
            try:return identity,entries(json.loads(data)),None,False
            except (ValueError,UnicodeDecodeError) as exc:return identity,[],dict(**identity,error=str(exc)),False
    # Bounded batches avoid hundreds of thousands of queued Future objects.
    # Hash every file,overlapping SSD metadata/read latency;retain rg order.
    with ThreadPoolExecutor(max_workers=16) as pool:
        for start in range(0,len(paths),512):
            for identity,found,error,same in pool.map(inspect,paths[start:start+512]):
                inventory.append(identity);seen.add(identity['path']);reused+=int(same);parsed+=int(not same)
                if error:errors.append(error)
                if found:hits.append(dict(**identity,entries=found))
            if start//512%48==0:print(json.dumps(dict(verified_json=len(inventory),reused_parse_proofs=reused,new_or_changed_parsed=parsed)),flush=True)
    removed=sorted(set(known)-seen)
    removed_allocations=sorted(set(knownhits)&set(removed))
    later=[]
    for item in hits:
        for entry in item['entries']:
            key=entry['key_path'].split('/')[-1].lower();value=entry['value']
            if key not in ('attempt','attempt_id','allocation_attempt','error_allocation_attempt_reserved'):continue
            numeric=float(value) if isinstance(value,(int,float)) or (isinstance(value,str) and value.isdigit()) else None
            if numeric is not None and numeric>=4:later.append(dict(path=item['path'],entry=entry))
    allocation=e.read(e.FORMAL/'allocation.json')
    if allocation['attempt']!=3 or allocation['allocated_error']!=.05/12 or not allocation['predecessor_verified']:
        raise ValueError('Completed attempt3 allocation changed')
    OUT.mkdir(exist_ok=False)
    result=dict(json_files_scanned=len(inventory),all_current_json_bytes_hashed=True,
        reused_identical_previous_parse_proofs=reused,new_or_changed_files_parsed=parsed,
        filename_filter_used=False,size_exclusions=[],parse_errors=errors,removed_prior_files=removed,
        removed_prior_allocation_candidates=removed_allocations,allocation_candidate_files=hits,
        attempt4_or_later_entries=later,inventory=inventory,previous_inventory_sha256=e.sha(oldpath),
        attempt3_owner_sha256=e.sha(e.FORMAL/'orchestrator_native_receipt.json'),
        attempt3_allocation_sha256=e.sha(e.FORMAL/'allocation.json'),
        old_spent_errors_retained=[.025,.05/6,.05/12],next_scheduled_error=.05/20,
        new_attempt_reserved=None,new_error_allocated=False,historical_cross_family_fwer_claim=False,
        no_numeric_attempt4_or_later_found=not later,complete=not errors and not removed_allocations and not later)
    raw=json.dumps(result,indent=2).encode()+b'\n'
    with zipfile.ZipFile(OUT/'full_inventory.zip','x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:z.writestr('current_inventory.json',raw)
    with zipfile.ZipFile(OUT/'full_inventory.zip') as z:
        if z.read('current_inventory.json')!=raw:raise ValueError('Archived audit bytes differ')
    compact={k:v for k,v in result.items() if k not in ('inventory','allocation_candidate_files')}
    compact.update(full_inventory_sha256=hashlib.sha256(raw).hexdigest(),archive_sha256=e.sha(OUT/'full_inventory.zip'),
                   audit_source_sha256=e.sha(__file__))
    e.write(OUT/'verification.json',compact)
    print(json.dumps(compact),flush=True)
    if not result['complete']:raise ValueError('Budget audit incomplete/conflicting;allocate nothing')


if __name__=='__main__':run()
