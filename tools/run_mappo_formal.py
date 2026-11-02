"""Register explicit unspent attempt3,then own every exact native worker handle."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
from mappo_native_launch import command
import mappo_formal_experiment as experiment

ROOT=experiment.ROOT;BASE=experiment.BASE;OUT=experiment.FORMAL


def register():
    plan=experiment.verify_plan(OUT,False)
    if plan['validation_only'] or plan['main_sumo_runs']!=16384 or plan['total_sumo_runs']!=37888:
        raise ValueError('Unchanged full-size main/comparison study required')
    experiment.design.validate_candidate()
    proof_path=BASE/'mappo_predecessor_budget_audit_20261102/verification.json';proof=experiment.read(proof_path)
    if not proof['predecessor_terminal_verified'] or proof['numeric_attempt3_or_later_in_inventory'] or proof['parse_errors']:
        raise ValueError('No complete unused-attempt/predecessor audit')
    for name,digest in proof['evidence_sha256'].items():
        if experiment.sha(ROOT/name)!=digest:raise ValueError('Predecessor inventory/terminal chain changed')
    validation=experiment.VALIDATION;v=experiment.read(validation/'verification.json');native=experiment.read(validation/'orchestrator_native_receipt.json')
    if (v['actual_sumo_executions']!=64 or v['full_independent_reviews']!=64 or native['actual_native_exit_code']!=0
            or any(e['actual_native_exit_code']!=0 for e in v['stages'])
            or v['source_verification_sha256']!=experiment.sha(validation/'validation/shard0/verification.json')):
        raise ValueError('Full actual-runtime validation/native exit required')
    # Explicit new slot;existing allocations never overwritten or reassigned.
    allocation=dict(attempt=3,allocated_error=.05/12,total_attempt_error=.05,predecessor_verified=True,
        predecessor_audit_sha256=experiment.sha(proof_path),old_withdrawn_alpha_retained=.025,old_completed_alpha_retained=.05/6,
        old_error_reassigned=False,old_certificate_transferred=False,historical_cross_family_fwer_claim=False,
        fixed_models=[4703,4704,4705],common_floor=.0025,plan_sha256=experiment.sha(OUT/'plan.json'),
        validation_sha256=experiment.sha(validation/'verification.json'),reserved_utc=datetime.now(timezone.utc).isoformat())
    experiment.write(OUT/'allocation.json',allocation)
    experiment.write(OUT/'binding.json',dict(plan_sha256=experiment.sha(OUT/'plan.json'),allocation_sha256=experiment.sha(OUT/'allocation.json'),
        predecessor_audit_sha256=experiment.sha(proof_path),validation_sha256=experiment.sha(validation/'verification.json'),
        source_commit=plan['source_commit'],formal_runs_at_binding=0,controller_or_threshold_selection=False))
    print(json.dumps(dict(attempt_reserved=3,alpha=.05/12,formal_runs=0)),flush=True)


def run():
    experiment.bound(OUT,False);execution=OUT/'execution';execution.mkdir(exist_ok=False)
    stages=[]
    def record(event):
        with (execution/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(event)+'\n')
    def child_group(action,stage):
        running=[]
        for shard in range(4):
            args=command(ROOT/'tools/mappo_formal_experiment.py',action,'--stage',stage,'--shard',shard)
            log=(execution/f'{stage}_{action}_{shard}.log').open('x',encoding='utf-8')
            child=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            event=dict(stage=stage,action=action,shard=shard,native_pid=child.pid,command=args,started_utc=datetime.now(timezone.utc).isoformat())
            record(event);running.append((child,log,event))
        failed=False
        for child,log,event in running:
            code=child.wait();log.close();event.update(actual_native_exit_code=code,finished_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=experiment.sha(OUT/'plan.json'))
            shard=event['shard'];directory=experiment.stage_path(OUT,stage,shard)
            if code==0 and action=='collect':
                complete=experiment.read(directory/'collection_complete.json')
                if complete['native_pid']!=child.pid:raise ValueError('Actual native child PID differs from execution identity')
                event['collection_complete_sha256']=experiment.sha(directory/'collection_complete.json')
                experiment.write(directory/'collect_native_exit.json',event)
            if code==0 and action=='review':event['verification_sha256']=experiment.sha(directory/'verification.json')
            record(event);stages.append(event);failed|=code!=0;print(json.dumps(event),flush=True)
        if failed:raise SystemExit(1)
    def one(action):
        args=command(ROOT/'tools/mappo_formal_finalize.py',action)
        with (execution/f'{action}.log').open('x',encoding='utf-8') as log:
            child=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            event=dict(action=action,native_pid=child.pid,command=args,started_utc=datetime.now(timezone.utc).isoformat());record(event)
            code=child.wait();event.update(actual_native_exit_code=code,finished_utc=datetime.now(timezone.utc).isoformat());record(event)
            print(json.dumps(event),flush=True)
            if code:raise SystemExit(code)
            return event
    for stage in ('calibration','confirmation'):
        child_group('collect',stage);child_group('review',stage)
        experiment.write(OUT/stage/'native_stage_receipt.json',dict(stage=stage,plan_sha256=experiment.sha(OUT/'plan.json'),
            stages=[e for e in stages if e['stage']==stage]))
        if stage=='calibration':one('matching')
    finalizer=one('finalize')
    experiment.write(OUT/'completion_receipt.json',dict(actual_native_finalizer=finalizer,all_collector_and_reviewer_native_exits_zero=True,
        stages=stages,main_actual_sumo_runs=16384,total_actual_sumo_runs=37888,independent_initials=4096,
        certificate_sha256=experiment.sha(OUT/'analysis/certificate.json'),comparisons_sha256=experiment.sha(OUT/'analysis/comparisons.json'),
        joint_deployment_certified=experiment.read(OUT/'analysis/certificate.json')['joint_deployment_certified']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('register','run'));a=p.parse_args()
    (register if a.action=='register' else run)()
