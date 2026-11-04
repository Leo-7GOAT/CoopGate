"""Reuse the complete1000-run cohort; compare both bounded rank-two gates."""
import argparse
from contextlib import ExitStack
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import sys
import time
import json

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import numpy as np
import torch
from coopgate.episode_store_v21 import EpisodeStore,observations
from mappo_native_launch import command
from experiments.top_two_candidates.smoke import factory,VARIANTS
from experiments.top_two_candidates.audit import audit_record
import experiments.switching_reserve.run as reference

old=reference.old;e=old.e;runtime=old.runtime
OUT=e.BASE/'development/local_top_two_development_20261103'
SMOKE=e.BASE/'development/local_top_two_smoke_20261103'
SEEDS=('4703','4704','4705')


def owned(directory):
    receipt=e.read(directory/'owner_native_exit.json')
    if receipt['actual_native_exit_code']!=0 or receipt['report_sha256']!=e.sha(directory/'report.json'):
        raise ValueError('Actual successful terminal bound to report required')


def freeze():
    parent=reference.guard();owned(reference.OUT);owned(SMOKE)
    smoke=e.read(SMOKE/'plan.json');schedules=[]
    for shard in range(4):
        for index,block in enumerate(e.read(reference.OUT/f'shard{shard}/report.json')['blocks']):
            schedules.append(dict(shard=shard,index=index,case=block['case'],
                                  reference=block,reuse_smoke=block['case']==smoke['case']))
    if len(schedules)!=100 or sum(c['reuse_smoke'] for c in schedules)!=1:
        raise ValueError('All100 prior cases and exact6-run integrity preflight required')
    sources=dict(parent['sources'])
    for name in ('experiments/top_two_candidates/__init__.py','experiments/top_two_candidates/gate.py',
        'experiments/top_two_candidates/audit.py','experiments/top_two_candidates/run.py',
        'experiments/top_two_candidates/smoke.py','tests/test_local_top_two_candidates.py',
        'docs/top_two_candidates_development_20261103_protocol.md'):
        sources[name]=e.sha(ROOT/name)
    OUT.mkdir(exist_ok=False)
    e.write(OUT/'plan.json',dict(schedules=schedules,sources=sources,
        reference_report_sha256=e.sha(reference.OUT/'report.json'),
        reference_owner_sha256=e.sha(reference.OUT/'owner_native_exit.json'),
        smoke_plan_sha256=e.sha(SMOKE/'plan.json'),smoke_database_sha256=e.sha(SMOKE/'observations.sqlite'),
        additional_actual_sumo=594,reused_smoke_actual_sumo=6,reused_reference_actual_sumo=1000,
        effective_paired_executions=1600,reused_initials=100,new_independent_initials=0,
        alpha_allocated=False,registered_utc=datetime.now(timezone.utc).isoformat()))


def guard():
    p=e.read(OUT/'plan.json');owned(reference.OUT);owned(SMOKE)
    if e.sha(reference.OUT/'report.json')!=p['reference_report_sha256']:
        raise ValueError('Completed reference changed')
    if e.sha(SMOKE/'observations.sqlite')!=p['smoke_database_sha256']:
        raise ValueError('Integrity-preflight observations changed')
    for name,digest in p['sources'].items():
        if e.sha(ROOT/name)!=digest:raise ValueError('Frozen reserve study source changed: '+name)
    return p


def worker(shard):
    torch.set_num_threads(1);p=guard();models=e.load_models();started=time.perf_counter()
    root=OUT/f'shard{shard}';root.mkdir(exist_ok=False);blocks=[]
    with ExitStack() as stack:
        envs={profile:stack.enter_context(runtime.MatchedShiftedMergeEnv(plant=profile)) for profile in e.design.PLANTS}
        for item in sorted((x for x in p['schedules'] if x['shard']==shard),key=lambda x:x['index']):
            guard();case=item['case'];index=item['index'];directory=root/f'block{index:03d}';directory.mkdir(exist_ok=False)
            if item['reuse_smoke']:
                db=SMOKE/'observations.sqlite';seal=e.read(Path(str(db)+'.seal.json'))
                metadata=seal['metadata'];keys=seal['cases']
            else:
                db=directory/'observations.sqlite'
                methods=[(v,s) for v in VARIANTS for s in SEEDS]
                offset=index%len(methods);methods=methods[offset:]+methods[:offset]
                keys=[[case['cluster_id'],v,s] for v,s in methods]
                metadata=dict(plan_sha256=e.sha(OUT/'plan.json'),case=case,shard=shard,index=index)
                store=EpisodeStore(db,metadata)
                try:
                    for _,variant,method in keys:
                        record=runtime.execute_episode(case,method,models[method],envs[case['plant']],controller_factory=factory(variant))
                        payload={k:v for k,v in record.items() if k!='trace'};payload['variant']=variant
                        store.append([case['cluster_id'],variant,method],payload,record['trace'])
                    store.seal(keys)
                finally:store.close()
            for key,record,trace in observations(db,metadata,keys):
                record['trace']=trace
                audit_record(case,record,models[record['method']],reference_factory=factory(record['variant']))
                old.episode_outcome(record['row'])
            proof=dict(case=case,reference=item['reference'],reuse_smoke=item['reuse_smoke'],
                database=db.relative_to(ROOT).as_posix(),database_sha256=e.sha(db),metadata=metadata,keys=keys,
                complete_controller_and_physics_replay=True)
            e.write(directory/'reviewed.json',proof);blocks.append(proof)
            print(json.dumps(dict(shard=shard,reviewed_initials=len(blocks),expected=25,
                new_actual_sumo=sum(6 for b in blocks if not b['reuse_smoke']),elapsed_s=time.perf_counter()-started)),flush=True)
    e.write(root/'report.json',dict(blocks=blocks,native_pid=os.getpid()))


def read_block(proof):
    db=ROOT/proof['database']
    if e.sha(db)!=proof['database_sha256']:raise ValueError('Reviewed database changed')
    result={}
    for key,record,trace in observations(db,proof['metadata'],proof['keys']):
        record['trace']=trace;result[record.get('variant','reserve'),record['method']]=record
    return result


def finalize(receipts):
    guard();totals={};times={};paired=[];adverse=[];zero_equal=0;count=0
    for shard in range(4):
        for block in e.read(OUT/f'shard{shard}/report.json')['blocks']:
            records=reference.read_block(block['reference']['reference'])
            records.update(reference.read_block(block['reference']))
            records.update(read_block(block));case=block['case']
            base=records['baseline','baseline'];b=old.episode_outcome(base['row']);outcomes={}
            if len(records)!=16:raise ValueError('Every old/new method must remain paired')
            for (variant,method),record in records.items():
                count+=1;a=old.episode_outcome(record['row'])
                harm=bool(a['failed'] or b['failed'] or np.any(a['costs']>b['costs']+1e-9)
                          or np.any(a['arrivals']>b['arrivals']+1e-9))
                gain=100*(b['team_cost']-a['team_cost'])/b['team_cost']
                outcomes[variant,method]=dict(adopted=a['adopted'],harm=harm,failed=a['failed'],gain=gain)
                name=case['stage']+':'+variant+':'+method
                r=totals.setdefault(name,dict(initials=0,adopted=0,harmful_adopted=0,execution_failures=0,gain_sum=0.))
                r['initials']+=1;r['adopted']+=int(a['adopted']);r['harmful_adopted']+=int(a['adopted'] and harm)
                r['execution_failures']+=int(a['failed']);r['gain_sum']+=gain
                times.setdefault(name,[]).extend(c['wall_ms'] for c in record['calls'])
                if variant in VARIANTS and not a['adopted'] and not a['failed'] and not b['failed']:
                    motion=lambda r:[(t['t'],t['command'],t['x']) for t in r['trace']]
                    if motion(record)!=motion(base):raise ValueError('Reserve fallback differs from actual baseline')
                    zero_equal+=1
                if variant in VARIANTS and harm:
                    adverse.append(dict(case=case,method=method,adopted=a['adopted'],failed=a['failed'],
                        cost_differences=(a['costs']-b['costs']).tolist(),arrival_differences=(a['arrivals']-b['arrivals']).tolist()))
            for method in SEEDS:
                paired.append(dict(case=case,method=method,history=outcomes['history',method],
                                   joint=outcomes['joint',method],reserve=outcomes['reserve',method],
                                   top_history=outcomes['top_history',method],top_reserve=outcomes['top_reserve',method]))
    if count!=1600 or len(paired)!=300:raise ValueError('Incomplete fixed reuse experiment')
    for name,r in totals.items():
        r['mean_paired_percentage_gain']=r.pop('gain_sum')/r['initials'];t=np.asarray(times[name])
        r.update(callback_calls=len(t),callback_p95_ms=float(np.quantile(t,.95)),callback_max_ms=float(t.max()),callback_over_500ms=int((t>500).sum()))
    promotion={}
    for variant in VARIANTS:
        prior=[r for r in paired if r['case']['stage']=='fresh']
        known=[r for r in paired if r['case']['stage']=='regression']
        gains={seed:totals['fresh:'+variant+':'+seed]['mean_paired_percentage_gain']-
            totals['fresh:history:'+seed]['mean_paired_percentage_gain'] for seed in SEEDS}
        adoption={seed:totals['fresh:'+variant+':'+seed]['adopted']-
            totals['fresh:history:'+seed]['adopted'] for seed in SEEDS}
        fail=sum(r[variant]['failed'] for r in paired)
        harm_prior=sum(r[variant]['harm'] for r in prior)
        harm_known=sum(r[variant]['harm'] for r in known)
        introduced=sum(not r['history']['harm'] and r[variant]['harm'] for r in known)
        promotion[variant]=dict(passed=bool(fail==0 and harm_prior==0 and harm_known<=14 and introduced==0
            and min(gains.values())>=.05 and min(adoption.values())>=0),
            prior80_gain_change_pp=gains,prior80_adoption_change=adoption,
            execution_failures=fail,prior80_harm=harm_prior,known20_harm=harm_known,
            known20_harm_introduced_vs_history=int(introduced),statistical_certificate=False)
    e.write(OUT/'paired_outcomes.json',paired);e.write(OUT/'all_adverse.json',adverse)
    e.write(OUT/'report.json',dict(completed=True,additional_actual_sumo_executions=594,
        reused_smoke_actual_executions=6,reused_reference_actual_executions=1000,
        effective_paired_executions=1600,new_independent_initials=0,reused_development_initials=100,
        original_source_stage_fresh_is_now_reused=True,results=totals,children=receipts,
        development_promotion=promotion,
        zero_adoption_exact_physical_fallbacks=zero_equal,statistical_certificate=False,
        alpha_allocated=False,no_cross_batch_causal_wall_time_speedup_claim=True))
    print(json.dumps(dict(completed=True,additional_sumo=594,results=totals)),flush=True)


def run():
    freeze();children=[];receipts=[]
    for shard in range(4):
        stream=(OUT/f'shard{shard}.log').open('xb')
        child=subprocess.Popen(command(__file__,'--shard',shard),cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        e.write(OUT/f'shard{shard}_started.json',dict(native_pid=child.pid,owner_pid=os.getpid()))
        children.append((shard,child,stream))
    for shard,child,stream in children:
        code=child.wait();stream.close();receipt=dict(shard=shard,native_pid=child.pid,actual_native_exit_code=code)
        e.write(OUT/f'shard{shard}_native_exit.json',receipt);receipts.append(receipt)
    if any(r['actual_native_exit_code'] for r in receipts):raise RuntimeError('Retain incomplete reserve experiment')
    finalize(receipts)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--shard',type=int,choices=range(4));a=parser.parse_args()
    run() if a.shard is None else worker(a.shard)
