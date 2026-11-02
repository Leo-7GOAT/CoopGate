"""Repeat one COMPLETE formal shard audit without changing original receipts.

Runs deterministic numerical/controller replay,not new SUMO episodes or
independent samples. Output is exclusive and restricted to this repo's runs.
"""
import argparse
import json
from pathlib import Path
import time
import torch
import mappo_formal_experiment as e
import mappo_formal_runtime as runtime


def replay(stage,shard,out):
    out=Path(out).resolve()
    if not out.is_relative_to(e.ROOT/'runs') or out.exists():raise ValueError('New report inside repository runs/ required')
    root=e.FORMAL;plan=e.bound(root,False)
    complete=e.read(root/'completion_receipt.json');native=e.read(root/'orchestrator_native_receipt.json')
    if native['actual_native_exit_code']!=0 or complete['actual_native_finalizer']['actual_native_exit_code']!=0:
        raise ValueError('Original complete native owner/finalizer required')
    entries=[x for x in complete['stages'] if x['stage']==stage and x['shard']==shard]
    if len(entries)!=2 or {x['action'] for x in entries}!={'collect','review'} or any(x['actual_native_exit_code']!=0 for x in entries):
        raise ValueError('Original native collect/review pair required')
    directory=e.stage_path(root,stage,shard)
    for x in entries:
        key='verification' if x['action']=='review' else 'collection_complete'
        if e.sha(directory/(key+'.json'))!=x[key+'_sha256']:raise ValueError('Original reviewed data changed')
    torch.set_num_threads(1);models=e.load_models();count=commands=substeps=failures=0;started=time.perf_counter()
    for case,record in e.reviewed_records(root,stage,shard,plan):
        model=None if record['method']=='baseline' else models[record['method'].split('_')[-1]]
        proof=runtime.audit_record(case,record,model);count+=1;commands+=proof['reference_replayed_commands']
        substeps+=proof['physical_substeps'];failures+=proof['recorded_failure']
        if count%256==0:print(json.dumps(dict(stage=stage,shard=shard,reviewed=count)),flush=True)
    if count!=(5376 if stage=='calibration' else 4096):raise ValueError('Full original shard required')
    e.bound(root,False);out.parent.mkdir(parents=True,exist_ok=True)
    result=dict(stage=stage,shard=shard,executions=count,commands=commands,physical_substeps=substeps,
        accurately_recorded_failures_retained=failures,elapsed_s=time.perf_counter()-started,
        original_verification_sha256=e.sha(directory/'verification.json'),plan_sha256=e.sha(root/'plan.json'),
        original_files_modified=False,new_sumo_runs=0,new_independent_initials=0,all_recorded_outcomes_replay=True)
    e.write(out,result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('calibration','confirmation'),required=True)
    p.add_argument('--shard',type=int,choices=range(4),required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();replay(a.stage,a.shard,a.out)
