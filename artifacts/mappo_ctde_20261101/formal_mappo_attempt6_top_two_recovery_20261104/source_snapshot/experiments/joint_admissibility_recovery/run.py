"""Replay all retained sealed blocks; execute only absent original case blocks."""
import argparse
from contextlib import ExitStack
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import numpy as np
import torch
from coopgate.episode_store_v21 import EpisodeStore,observations,inventory
from mappo_native_launch import command
from experiments.joint_admissibility_recovery.audit import audit_record
import experiments.joint_admissibility.run_development as old

e=old.e;runtime=old.runtime
OUT=e.BASE/'development/joint_admissibility_audit_recovery_20261103'
PROTOCOL=ROOT/'docs/joint_admissibility_audit_recovery_20261103_protocol.md'


def expected(index,case):
    offset=index%len(old.VARIANTS)
    order=old.VARIANTS[offset:]+old.VARIANTS[:offset]
    return [[case['cluster_id'],v,m] for v,m in order]


def freeze():
    original=old.guard();terminal=e.read(old.OUT/'owner_native_exit.json')
    if terminal['actual_native_exit_code']!=1:
        raise ValueError('Require original actual failed native owner; never replace it')
    for shard in range(4):
        receipt=e.read(old.OUT/f'shard{shard}_native_exit.json')
        if receipt['actual_native_exit_code'] not in (0,1):raise ValueError('Unexpected original terminal')
        if receipt['actual_native_exit_code']:
            if 'Gain threshold admission differs' not in (old.OUT/f'shard{shard}.log').read_text():
                raise ValueError('Recovery scope only covers the known trace-schema audit mismatch')
    schedules=[];seen=set();retained=0
    for shard in range(4):
        for index,case in enumerate(original['cases'][shard::4]):
            db=old.OUT/f'shard{shard}'/f'block{index:03d}'/'observations.sqlite'
            item=dict(shard=shard,index=index,case=case,keys=expected(index,case),retained=db.exists())
            if db.exists():
                seal=Path(str(db)+'.seal.json')
                if not seal.exists():raise ValueError('Unsealed original block must remain unresolved')
                state=inventory(db);receipt=e.read(seal)
                metadata=dict(plan_sha256=e.sha(old.OUT/'plan.json'),shard=shard,index=index,case=case)
                if state['metadata']!=metadata or state['cases']!=item['keys']:
                    raise ValueError('Original sealed block differs from complete schedule')
                if receipt['database_sha256']!=e.sha(db):raise ValueError('Original block bytes changed')
                item.update(database=db.relative_to(ROOT).as_posix(),database_sha256=e.sha(db),
                    seal_sha256=e.sha(seal),metadata=metadata)
                retained+=len(item['keys']);seen.add(db.resolve())
            schedules.append(item)
    if set(p.resolve() for p in old.OUT.glob('shard*/block*/observations.sqlite'))!=seen:
        raise ValueError('Every original block must be accounted for')
    if len(schedules)!=100 or retained!=644:raise ValueError('Actual original sealed inventory differs')
    sources=dict(original['sources'])
    for relative in ('experiments/joint_admissibility_recovery/__init__.py',
        'experiments/joint_admissibility_recovery/audit.py',
        'experiments/joint_admissibility_recovery/run.py',
        'tests/test_joint_admissibility_recovery_audit.py',
        'docs/joint_admissibility_audit_recovery_20261103_protocol.md'):
        sources[relative]=e.sha(ROOT/relative)
    OUT.mkdir(exist_ok=False)
    e.write(OUT/'plan.json',dict(original_plan=original,
        original_plan_sha256=e.sha(old.OUT/'plan.json'),
        original_failed_owner_sha256=e.sha(old.OUT/'owner_native_exit.json'),
        original_native_exit_code=1,sources=sources,schedules=schedules,
        retained_sealed_actual_executions=retained,missing_actual_executions=700-retained,
        total_effective_actual_executions=700,all_original_initials=100,
        controller_changed=False,original_data_modified=False,new_error_allocated=False,
        registered_utc=datetime.now(timezone.utc).isoformat()))


def guard(verify_inventory=True):
    old.guard();plan=e.read(OUT/'plan.json')
    if e.sha(old.OUT/'plan.json')!=plan['original_plan_sha256']:
        raise ValueError('Original failed development design changed')
    if e.sha(old.OUT/'owner_native_exit.json')!=plan['original_failed_owner_sha256']:
        raise ValueError('Original actual failure receipt changed')
    for name,digest in plan['sources'].items():
        if e.sha(ROOT/name)!=digest:raise ValueError('Frozen recovery source changed: '+name)
    for item in plan['schedules'] if verify_inventory else ():
        if item['retained']:
            db=ROOT/item['database']
            if e.sha(db)!=item['database_sha256'] or e.sha(Path(str(db)+'.seal.json'))!=item['seal_sha256']:
                raise ValueError('Never modify retained original observations')
    return plan


def worker(shard):
    torch.set_num_threads(1);plan=guard();models=e.load_models();started=time.perf_counter()
    root=OUT/f'shard{shard}';root.mkdir(exist_ok=False);blocks=[]
    items=[c for c in plan['schedules'] if c['shard']==shard]
    with ExitStack() as stack:
        envs={p:stack.enter_context(runtime.MatchedShiftedMergeEnv(plant=p))
              for p in {c['case']['plant'] for c in items if not c['retained']}}
        for item in items:
            guard(False);case=item['case'];index=item['index']
            directory=root/f'block{index:03d}';directory.mkdir(exist_ok=False)
            if item['retained']:
                db=ROOT/item['database'];metadata=item['metadata']
            else:
                db=directory/'observations.sqlite'
                metadata=dict(recovery_plan_sha256=e.sha(OUT/'plan.json'),shard=shard,index=index,case=case)
                store=EpisodeStore(db,metadata)
                try:
                    for _,variant,method in item['keys']:
                        model=None if method=='baseline' else models[method]
                        record=runtime.execute_episode(case,method,model,envs[case['plant']],
                                                       controller_factory=old.factory(variant))
                        payload={k:v for k,v in record.items() if k!='trace'};payload['variant']=variant
                        store.append([case['cluster_id'],variant,method],payload,record['trace'])
                    store.seal(item['keys'])
                finally:store.close()
            for key,record,trace in observations(db,metadata,item['keys']):
                record['trace']=trace
                model=None if record['method']=='baseline' else models[record['method']]
                audit_record(case,record,model,reference_factory=old.factory(record['variant']))
                old.episode_outcome(record['row'])
            proof=dict(case=case,original_index=index,retained=item['retained'],keys=item['keys'],
                metadata=metadata,database=db.relative_to(ROOT).as_posix(),
                database_sha256=e.sha(db),seal_sha256=e.sha(Path(str(db)+'.seal.json')),
                complete_controller_and_physics_replay=True)
            e.write(directory/'reviewed.json',proof);blocks.append(proof)
            print(json.dumps(dict(shard=shard,replayed_initials=len(blocks),expected=25,
                effective_executions=7*len(blocks),new_sumo=sum(7 for b in blocks if not b['retained']),
                elapsed_s=time.perf_counter()-started)),flush=True)
    e.write(root/'report.json',dict(native_pid=os.getpid(),blocks=blocks,
        executions=7*len(blocks),new_actual_sumo=sum(7 for b in blocks if not b['retained'])))


def finalize(receipts):
    plan=guard();totals={};timings={};adverse=[];paired=[];zero_equal=effective=added=0
    for shard in range(4):
        for block in e.read(OUT/f'shard{shard}/report.json')['blocks']:
            db=ROOT/block['database']
            if e.sha(db)!=block['database_sha256']:raise ValueError('Reviewed bytes changed')
            case=block['case'];records={}
            for key,record,trace in observations(db,block['metadata'],block['keys']):
                record['trace']=trace;records[record['variant'],record['method']]=record
                effective+=1;added+=int(not block['retained'])
            base=records['baseline','baseline'];b=old.episode_outcome(base['row']);outcomes={}
            for (variant,method),record in records.items():
                row=record['row'];actual=old.episode_outcome(row)
                harm=bool(actual['failed'] or b['failed'] or np.any(actual['costs']>b['costs']+1e-9)
                          or np.any(actual['arrivals']>b['arrivals']+1e-9))
                gain=100*(b['team_cost']-actual['team_cost'])/b['team_cost']
                outcomes[variant,method]=dict(adopted=actual['adopted'],harm=harm,
                    failed=actual['failed'],gain=gain)
                name=case['stage']+':'+variant+':'+method
                r=totals.setdefault(name,dict(initials=0,adopted=0,harmful_adopted=0,
                                               execution_failures=0,gain_sum=0.))
                r['initials']+=1;r['adopted']+=int(actual['adopted'])
                r['harmful_adopted']+=int(actual['adopted'] and harm)
                r['execution_failures']+=int(actual['failed']);r['gain_sum']+=gain
                timings.setdefault(name,[]).extend(c['wall_ms'] for c in record['calls'])
                if variant!='baseline' and not actual['adopted'] and not actual['failed'] and not b['failed']:
                    a=[(t['t'],t['command'],t['x']) for t in record['trace']]
                    z=[(t['t'],t['command'],t['x']) for t in base['trace']]
                    if a!=z:raise ValueError('Zero-adoption physical fallback differs')
                    zero_equal+=1
                if variant!='baseline' and (actual['failed'] or harm):
                    adverse.append(dict(case=case,variant=variant,method=method,
                        adopted=actual['adopted'],failed=actual['failed'],
                        cost_differences=(actual['costs']-b['costs']).tolist(),
                        arrival_differences=(actual['arrivals']-b['arrivals']).tolist(),
                        source_database=block['database']))
            for method in ('4703','4704','4705'):
                paired.append(dict(case=case,method=method,history=outcomes['history',method],
                                   joint=outcomes['joint',method]))
    if effective!=700 or added!=56 or len(paired)!=300:
        raise ValueError('Every original paired execution must remain')
    for name,r in totals.items():
        r['mean_paired_percentage_gain']=r.pop('gain_sum')/r['initials'];t=np.asarray(timings[name])
        r.update(callback_calls=len(t),callback_p95_ms=float(np.quantile(t,.95)),
                 callback_max_ms=float(t.max()),callback_over_500ms=int((t>500).sum()))
    e.write(OUT/'all_adverse.json',adverse);e.write(OUT/'paired_outcomes.json',paired)
    e.write(OUT/'report.json',dict(completed=True,actual_sumo_executions=700,
        original_retained_actual_sumo_executions=644,new_recovery_actual_sumo_executions=56,
        original_root_actual_native_exit_code=1,original_failure_receipt_sha256=plan['original_failed_owner_sha256'],
        children=receipts,results=totals,zero_adoption_exact_physical_fallbacks=zero_equal,
        all_raw_blocks_sealed_and_exact_replayed=True,new_independent_development_initials=80,
        known_regression_initials=20,controller_changed=False,original_data_modified=False,
        statistical_certificate=False,alpha_allocated=False,deployment_certified=False,
        paired_outcomes_sha256=e.sha(OUT/'paired_outcomes.json')))
    print(json.dumps(dict(completed=True,effective_executions=700,new_actual_sumo=56,results=totals)),flush=True)


def run():
    freeze();children=[]
    for shard in range(4):
        stream=(OUT/f'shard{shard}.log').open('xb')
        child=subprocess.Popen(command(__file__,'--shard',shard),cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        e.write(OUT/f'shard{shard}_started.json',dict(native_pid=child.pid,owner_pid=os.getpid()))
        children.append((shard,child,stream))
    receipts=[]
    for shard,child,stream in children:
        code=child.wait();stream.close()
        receipt=dict(shard=shard,native_pid=child.pid,actual_native_exit_code=code,
                     observed_by_direct_native_wait=True)
        e.write(OUT/f'shard{shard}_native_exit.json',receipt);receipts.append(receipt)
    if any(r['actual_native_exit_code'] for r in receipts):raise RuntimeError('Retain failed recovery; never replace it')
    finalize(receipts)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--shard',type=int,choices=range(4));args=parser.parse_args()
    run() if args.shard is None else worker(args.shard)
