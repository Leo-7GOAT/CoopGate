"""Explicit recovery of the genuinely interrupted, unchanged full V42 study."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import FunctionType, SimpleNamespace
import zipfile

from coopgate.episode_store_v21 import inventory, observations, readonly, unpack
import execute_joint_risk_v42 as original
import parallel_joint_risk_replay_20261027 as replay
import legacy_withdrawal_contract_20261027 as withdrawal
from recover_ood_20261026 import non_timing

ROOT=original.ROOT
PRIOR=original.OUT
OUT=ROOT/'artifacts/joint_risk_recovery_20261028'
SNAP=OUT/'interruption'
PLAN=OUT/'recovery.json'
SCRIPT='tools/joint_recovery_20261028.py'
SOURCES=(SCRIPT,'tests/test_joint_recovery_20261028.py',
    'operations/continue_joint_recovery_20261028.ps1',
    'operations/launch_joint_recovery_task_20261028.ps1',
    'docs/joint_recovery_20261028.md')
OPERATIONAL=('.git/mainline-after-authorized-withdrawal-20261027.jsonl',
    '.git/mainline-after-authorized-withdrawal-20261027.stdout.log',
    '.git/durable_after_withdrawal_launch_20261027.json')


def clone(function, namespace):
    result=FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)
    result.__kwdefaults__=function.__kwdefaults__
    return result


def assert_original_stopped():
    code='[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false); Get-CimInstance Win32_Process | Select-Object ProcessId,CreationDate,CommandLine | ConvertTo-Json -Compress'
    result=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',code],capture_output=True,check=True)
    processes=json.loads(result.stdout.decode('utf-8-sig'))
    original_identity=original.io.read(PRIOR/'calibration/execution/launch.json')
    live=[]
    for p in processes:
        command=(p.get('CommandLine') or '').replace('\\','/')
        same_native=(p['ProcessId']==original_identity['native_pid'] and p['CreationDate']==original_identity['native_creation_date'])
        if (same_native or 'tools/execute_joint_risk_after_withdrawal_20261027.py' in command
                or 'operations/continue_after_withdrawal_20261027.ps1' in command):
            live.append(p['ProcessId'])
    if live:raise ValueError('Original target identity remains live: '+str(live))
    execution=PRIOR/'calibration/execution'
    if any((execution/n).exists() for n in ('exit.json','collection_complete.json','blocks_manifest.json')):
        raise ValueError('Original terminal evidence exists; inspect instead of recovering')
    if (PRIOR/'confirmation/execution').exists() or (PRIOR/'analysis').exists():
        raise ValueError('Unexpected later original stage; preserve and inspect')


def file_map():
    paths=sorted(p for p in (PRIOR/'calibration').rglob('*') if p.is_file())
    paths += [original.PLAN,original.BINDING,PRIOR/'parallel_replay_binding.json']
    paths += [ROOT/n for n in OPERATIONAL]
    return {p.relative_to(ROOT).as_posix():original.io.digest(p) for p in paths}


def prefix_state(state, metadata, expected):
    n=len(state['cases'])
    if state['metadata']!=metadata or n>len(expected) or state['cases']!=expected[:n]:
        raise ValueError('Interrupted records are not the exact scheduled prefix')
    return n


def snapshot():
    assert_original_stopped()
    plan,binding=original.bound_inputs()
    SNAP.mkdir(parents=True,exist_ok=False)
    before=file_map()
    entries=[]
    directories=sorted((PRIOR/'calibration/blocks').iterdir())
    if [p.name for p in directories]!=[f'{i:05d}' for i in range(len(directories))]:
        raise ValueError('Interrupted directories are not a contiguous prefix')
    for index,directory in enumerate(directories):
        case=plan['stages']['calibration'][index]
        state=inventory(directory/'observations.sqlite')
        metadata=dict(stage='calibration',cluster_id=case['cluster_id'],
            plan_sha256=original.io.digest(original.PLAN),binding_sha256=original.io.digest(original.BINDING))
        n=prefix_state(state,metadata,original.collect.schedule(case))
        seal,complete=directory/'observations.sqlite.seal.json',directory/'complete.json'
        if seal.exists():
            value=original.io.read(seal)
            if (n!=4 or value['database_sha256']!=original.io.digest(directory/'observations.sqlite')
                    or {k:value[k] for k in state}!=state):raise ValueError('Original seal changed')
        status='complete' if seal.exists() and complete.exists() else 'unsealed_boundary'
        if status=='complete' and n!=4:raise ValueError('Incomplete sealed original block')
        if status!='complete' and index!=len(directories)-1:raise ValueError('Only final block may be incomplete')
        if complete.exists() and not seal.exists():raise ValueError('Unexpected incomplete seal chain')
        entries.append(dict(index=index,status=status,records=n,inventory=state,
            files={p.name:original.io.digest(p) for p in directory.iterdir() if p.is_file()}))
    if not entries:raise ValueError('No original prefix to preserve')
    batches=[];batch=[];size=0
    for name in before:
        length=(ROOT/name).stat().st_size
        if batch and size+length>60*1024*1024:batches.append(batch);batch=[];size=0
        batch.append(name);size+=length
    if batch:batches.append(batch)
    archives=[]
    import hashlib
    for index,names in enumerate(batches):
        path=SNAP/f'part-{index:03d}.zip'
        with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for name in names:z.write(ROOT/name,name)
        with zipfile.ZipFile(path) as z:
            if z.namelist()!=names or any(hashlib.sha256(z.read(n)).hexdigest()!=before[n] for n in names):
                raise ValueError('Snapshot bytes differ')
        archives.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=original.io.digest(path)))
    if file_map()!=before:raise ValueError('Interrupted originals changed during snapshot')
    original.io.exclusive_json(SNAP/'inventory.json',dict(original_exit_status='unknown',
        original_handles_absent=True,terminal_receipts_absent=True,partial_effects_analyzed=False,
        files=before,entries=entries,archives=archives,all_archived_bytes_verified=True,
        observed_utc=original.io.utc(),plan_sha256=original.io.digest(original.PLAN),
        binding_sha256=original.io.digest(original.BINDING)))
    print(json.dumps(dict(complete_blocks=sum(e['status']=='complete' for e in entries),
        boundary_records=sum(e['records'] for e in entries if e['status']!='complete'),archives=len(archives))),flush=True)


def freeze():
    assert_original_stopped();original.bound_inputs()
    if PLAN.exists():raise FileExistsError('Preserve first explicit recovery registration')
    proof=original.io.read(SNAP/'inventory.json')
    if proof['files']!=file_map() or proof['original_exit_status']!='unknown':raise ValueError('Snapshot identity changed')
    for item in proof['archives']:
        path=ROOT/item['path'];original.io.committed(path)
        if original.io.digest(path)!=item['sha256']:raise ValueError('Snapshot archive changed')
    original.io.committed(SNAP/'inventory.json')
    for name in SOURCES:original.io.committed(ROOT/name)
    complete=[e['index'] for e in proof['entries'] if e['status']=='complete']
    boundary=[e for e in proof['entries'] if e['status']!='complete']
    original.io.exclusive_json(PLAN,dict(schema='coopgate.joint-recovery.20261028',utc=original.io.utc(),
        original_plan_sha256=original.io.digest(original.PLAN),original_binding_sha256=original.io.digest(original.BINDING),
        snapshot_inventory_sha256=original.io.digest(SNAP/'inventory.json'),
        source_sha256={n:original.io.digest(ROOT/n) for n in SOURCES},
        retained_complete_blocks=complete,retained_complete_runs=4*len(complete),
        boundary_replay=[dict(index=e['index'],retained_partial_records=e['records']) for e in boundary],
        boundary_rule='Rerun only unsealed block; every saved row must match exactly except declared timing; duplicates are engineering evidence, never extra N',
        original_partial_engineering_runs=sum(e['records'] for e in boundary),
        additional_sumo_runs=16384-4*len(complete),cumulative_unique_sumo_runs=16384,
        independent_initials=4096,statistical_attempt=2,alpha=.05/6,
        thresholds_models_seeds_and_inference_unchanged=True,original_exits_not_rewritten=True,
        partial_effects_analyzed=False,automatic_retry=False))


def inputs(full=False, committed=True):
    if committed:original.io.committed(PLAN)
    plan=original.io.read(PLAN)
    if (plan['original_plan_sha256']!=original.io.digest(original.PLAN)
            or plan['original_binding_sha256']!=original.io.digest(original.BINDING)
            or plan['snapshot_inventory_sha256']!=original.io.digest(SNAP/'inventory.json')
            or plan['cumulative_unique_sumo_runs']!=16384 or plan['independent_initials']!=4096
            or plan['statistical_attempt']!=2 or plan['alpha']!=.05/6
            or not plan['thresholds_models_seeds_and_inference_unchanged']
            or not plan['original_exits_not_rewritten'] or plan['partial_effects_analyzed'] or plan['automatic_retry']):
        raise ValueError('Recovery scope changed')
    for name,sha in plan['source_sha256'].items():
        if committed:original.io.committed(ROOT/name)
        if original.io.digest(ROOT/name)!=sha:raise ValueError('Recovery source changed: '+name)
    proof=original.io.read(SNAP/'inventory.json')
    validate_accounting(plan,proof)
    if committed:withdrawal.inputs(full=full)
    if full:
        assert_original_stopped()
        proof=original.io.read(SNAP/'inventory.json')
        if file_map()!=proof['files']:raise ValueError('Interrupted source bytes changed')
        for item in proof['archives']:
            if original.io.digest(ROOT/item['path'])!=item['sha256']:raise ValueError('Snapshot archive changed')
    return plan


def validate_accounting(plan, proof):
    complete=[e['index'] for e in proof['entries'] if e['status']=='complete']
    partial=sum(e['records'] for e in proof['entries'] if e['status']!='complete')
    if (plan['retained_complete_blocks']!=complete or plan['retained_complete_runs']!=4*len(complete)
            or plan['original_partial_engineering_runs']!=partial
            or plan['additional_sumo_runs']!=16384-4*len(complete)):
        raise ValueError('Cumulative unique sample or extra replay accounting changed')


def verify_boundary(old_path,new_path,entry,case):
    expected=original.collect.schedule(case)
    old=inventory(old_path)
    if old!=entry['inventory']:raise ValueError('Saved partial prefix changed')
    new_records=list(observations(new_path,old['metadata'],expected))
    with readonly(old_path) as connection:
        rows=list(connection.execute('SELECT case_key,row_blob,trace_blob,row_sha256,trace_sha256 FROM episodes ORDER BY position'))
    for i,(key,row,trace,rh,th) in enumerate(rows):
        old_record=dict(**unpack(row,rh),trace=unpack(trace,th))
        new_key,body,new_trace=new_records[i]
        if json.loads(key)!=list(new_key) or non_timing(old_record)!=non_timing(dict(**body,trace=new_trace)):
            raise ValueError('Unsealed boundary changed non-timing observations; preserve and stop')
    return len(rows)


def collect_cluster(case,directory,models,**kwargs):
    plan=inputs(committed=False)
    proof=original.io.read(SNAP/'inventory.json')
    entry=proof['entries'][case['index']] if case['stage']=='calibration' and case['index']<len(proof['entries']) else None
    source=PRIOR/'calibration/blocks'/f"{case['index']:05d}"
    if entry:
        actual={p.name:original.io.digest(p) for p in source.iterdir() if p.is_file()}
        if actual!=entry['files']:raise ValueError('Original block bytes changed')
    if entry and entry['status']=='complete':
        shutil.copytree(source,directory)
        if {p.name:original.io.digest(p) for p in Path(directory).iterdir() if p.is_file()}!=entry['files']:
            raise ValueError('Copied completed block differs')
        return None
    result=original.collect.collect_cluster(case,directory,models,**kwargs)
    if entry:
        n=verify_boundary(source/'observations.sqlite',Path(directory)/'observations.sqlite',entry,case)
        target=OUT/'boundary_proof';target.mkdir(exist_ok=True)
        original.io.exclusive_json(target/f"{case['index']:05d}.json",dict(index=case['index'],
            retained_partial_rows=n,all_non_timing_fields_exact=True,extra_independent_initials=0,
            original_database_sha256=original.io.digest(source/'observations.sqlite')))
    return result


def routed_functions():
    namespace={**vars(original),'OUT':OUT,'SCRIPT':SCRIPT}
    for name,function in vars(original).items():
        if isinstance(function,FunctionType) and function.__module__==original.__name__:
            namespace[name]=clone(function,namespace)
    canonical_bound=namespace['bound_inputs']
    def bound_inputs():
        plan,binding=canonical_bound();extension=inputs()
        return plan,{**binding,'sources':{**binding['sources'],**extension['source_sha256'],
            PLAN.relative_to(ROOT).as_posix():original.io.digest(PLAN),
            (SNAP/'inventory.json').relative_to(ROOT).as_posix():original.io.digest(SNAP/'inventory.json')}}
    namespace['bound_inputs']=bound_inputs
    proxy=SimpleNamespace(**vars(original.collect));proxy.collect_cluster=collect_cluster
    namespace['collect']=proxy
    return namespace


def finalize(namespace):
    inputs(full=True)
    # Only the parent finalizer's output directory changes. Pool workers stay
    # canonical top-level functions, preserving Windows spawn picklability.
    proxy=SimpleNamespace(**namespace)
    parent=clone(replay.finalize,{**vars(replay),'original':proxy})
    parent(namespace)
    plan=inputs(full=True)
    original.io.exclusive_json(OUT/'analysis/recovery_accounting.json',dict(
        recovery_plan_sha256=original.io.digest(PLAN),original_native_exit='unknown',
        original_complete_runs_reused=plan['retained_complete_runs'],
        original_partial_engineering_runs=plan['original_partial_engineering_runs'],
        additional_sumo_runs=plan['additional_sumo_runs'],cumulative_unique_sumo_runs=16384,
        independent_initials=4096,all_original_criteria_retained=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ('snapshot','freeze','verify','finalize'):commands.add_parser(name)
    for name in ('watch','worker'):commands.add_parser(name).add_argument('stage',choices=original.design.STAGES)
    args=parser.parse_args()
    if args.command in ('snapshot','freeze'):return globals()[args.command]()
    inputs(full=args.command in ('verify','finalize'))
    if args.command=='verify':return
    namespace=routed_functions()
    if args.command=='finalize':return finalize(namespace)
    namespace[args.command](args.stage)


if __name__=='__main__':main()
