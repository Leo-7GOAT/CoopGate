"""Fixed all-case old/new gate mechanism ablation;no new risk certificate.

The original 1024 confirmation initials and three old gated actors were fixed
before confirmation collection. Extra SUMO runs start only after the primary
actual owner exits zero. All original formal data/thresholds remain untouched.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time
import torch
import numpy as np
from coopgate.episode_store_v21 import EpisodeStore,observations
from coopgate.compact_coupled_gate import CompactCoupledGate
from coopgate.strict_phase_only_gate import StrictPhaseOnlyGate
from mappo_native_launch import command
from mappo_powered_history_joint_data import episode_outcome
import mappo_powered_history_formal_experiment as e
import mappo_powered_history_formal_runtime as runtime
from mappo_completed_powered_attempt5 import complete_evidence

VARIANT=None
OUT=None
PROTOCOL=e.ROOT/'docs/mappo_powered_history_holdout_ablation_20261103_protocol.md'


def guard(require_complete=False):
    plan=e.read(OUT/'plan.json')
    if plan['variant']!=VARIANT:raise ValueError('Wrong fixed ablation variant')
    if plan['primary_plan_sha256']!=e.sha(e.FORMAL/'plan.json') or plan['protocol_sha256']!=e.sha(PROTOCOL):
        raise ValueError('Fixed case/protocol registration changed')
    primary=e.bound(e.FORMAL,False)
    if plan['cases']!=primary['cases']['confirmation'] or plan['methods']!=['4703','4704','4705'] or plan['gain_floor']!=.0025:
        raise ValueError('Must retain every original confirmation initial and old fixed model')
    if plan['source_sha256']!=primary['source_sha256'] or plan['expected_additional_sumo_executions']!=3072:
        raise ValueError('Frozen comparison source/scope changed')
    if len(plan['cases'])!=1024 or len({c['cluster_id'] for c in plan['cases']})!=1024:
        raise ValueError('Exactly all1024 independent primary confirmation initials required')
    if plan['actor_identities']!=e.design.validate_candidate()['actor_identities']:
        raise ValueError('Old/new gate must use exactly the same fixed actors')
    if require_complete:complete_evidence()
    return plan


def bind():
    plan=guard();source=Path(__file__).resolve()
    committed=subprocess.check_output(['git','show',plan['wrapper_source_commit']+':'+source.relative_to(e.ROOT).as_posix()],cwd=e.ROOT)
    import hashlib
    if hashlib.sha256(committed).hexdigest()!=e.sha(source):raise ValueError('Commit wrapper before binding')
    e.write(OUT/'implementation_binding.json',dict(plan_sha256=e.sha(OUT/'plan.json'),
        wrapper_sha256=e.sha(source),old_candidate_sha256=e.sha(e.BASE/'compact_coupled_fixed_candidate_20261102.json'),
        completion_guard_sha256=e.sha(e.ROOT/'tools/mappo_completed_powered_attempt5.py'),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=e.ROOT,text=True).strip(),
        bound_utc=datetime.now(timezone.utc).isoformat(),additional_sumo_runs_at_binding=0,
        new_error_allocated=False,primary_target_changed=False))


def bound(require_complete=False):
    plan=guard(require_complete);binding=e.read(OUT/'implementation_binding.json')
    if (binding['plan_sha256']!=e.sha(OUT/'plan.json') or binding['wrapper_sha256']!=e.sha(__file__)
        or binding['old_candidate_sha256']!=e.sha(e.BASE/'compact_coupled_fixed_candidate_20261102.json')):
        raise ValueError('Bound old-gate wrapper changed')
    if binding['completion_guard_sha256']!=e.sha(e.ROOT/'tools/mappo_completed_powered_attempt5.py'):
        raise ValueError('Primary actual-terminal guard changed')
    return plan


def old_factory(method,model,case):
    if method not in ('4703','4704','4705'):raise ValueError('Only the three fixed old gates are additional methods')
    cls=CompactCoupledGate if VARIANT=='compact' else StrictPhaseOnlyGate
    return cls(model,case['plant'],runtime.COUPLED_BANK)


def worker(shard):
    torch.set_num_threads(1);plan=bound(True);models=e.load_models();started=time.perf_counter()
    root=OUT/f'shard{shard}';root.mkdir(exist_ok=False);cases=plan['cases'][shard::4];blocks=[]
    warm=np.array([[-140.,12.],[-220.,10.],[-190.,12.]])
    for profile in e.design.PLANTS:
        for method in plan['methods']:old_factory(method,models[method],dict(plant=profile)).select(warm,np.zeros(3))
    with ExitStack() as stack:
        envs={p:stack.enter_context(runtime.MatchedShiftedMergeEnv(plant=p)) for p in e.design.PLANTS}
        for index in range(0,len(cases),8):
            bound();batch=cases[index:index+8];directory=root/f'block{index//8:03d}';directory.mkdir(exist_ok=False)
            expected=[]
            for case in batch:
                offset=case['index']%3;methods=plan['methods'][offset:]+plan['methods'][:offset]
                expected.extend([[case['cluster_id'],m] for m in methods])
            metadata=dict(plan_sha256=e.sha(OUT/'plan.json'),shard=shard,index=index//8,cases=batch)
            store=EpisodeStore(directory/'observations.sqlite',metadata)
            try:
                for cluster,method in expected:
                    case=next(c for c in batch if c['cluster_id']==cluster)
                    record=runtime.execute_episode(case,method,models[method],envs[case['plant']],controller_factory=old_factory)
                    payload={k:v for k,v in record.items() if k!='trace'};payload['variant']=VARIANT
                    store.append([cluster,method],payload,record['trace'])
                store.seal(expected)
            finally:store.close()
            for key,record,trace in observations(directory/'observations.sqlite',metadata,expected):
                record['trace']=trace;case=next(c for c in batch if c['cluster_id']==key[0])
                runtime.audit_record(case,record,models[record['method']],reference_factory=old_factory)
                episode_outcome(record['row'])
            proof=dict(block=directory.name,metadata=metadata,keys=expected,
                database_sha256=e.sha(directory/'observations.sqlite'),seal_sha256=e.sha(directory/'observations.sqlite.seal.json'),
                full_controller_and_physics_replay=True)
            e.write(directory/'reviewed.json',proof);blocks.append(proof)
            print(json.dumps(dict(shard=shard,reviewed_executions=sum(len(b['keys']) for b in blocks),expected=768,
                elapsed_s=time.perf_counter()-started)),flush=True)
    e.write(root/'report.json',dict(native_pid=os.getpid(),blocks=blocks,executions=768,elapsed_s=time.perf_counter()-started))


def motion(trace):return [(t['t'],t['command'],t['x']) for t in trace]


def finalize(receipts):
    plan=bound(True);old={};primary={}
    for shard in range(4):
        summary=e.read(OUT/f'shard{shard}/report.json')
        for block in summary['blocks']:
            db=OUT/f'shard{shard}'/block['block']/'observations.sqlite'
            if e.sha(db)!=block['database_sha256']:raise ValueError('Reviewed old-gate data changed')
            for key,record,trace in observations(db,block['metadata'],block['keys']):
                if key in old:raise ValueError('Duplicate old-gate observation')
                old[key]=(record['row'],motion(trace))
        for case,record in e.reviewed_records(e.FORMAL,'confirmation',shard,e.bound(e.FORMAL,False)):
            if record['method'] not in ('baseline','4703','4704','4705'):continue
            key=(case['cluster_id'],record['method'])
            if key in primary:raise ValueError('Duplicate primary paired observation')
            primary[key]=(record['row'],motion(record['trace']))
    wanted={(c['cluster_id'],m) for c in plan['cases'] for m in plan['methods']}
    if set(old)!=wanted or set(primary)!=(wanted|{(c['cluster_id'],'baseline') for c in plan['cases']}):
        raise ValueError('Every fixed heldout initial/model must be paired,including adverse outcomes')
    def outcome(row,base):
        q=episode_outcome(row);b=episode_outcome(base)
        harm=bool(q['failed'] or b['failed'] or np.any(q['costs']>b['costs']+1e-9) or np.any(q['arrivals']>b['arrivals']+1e-9))
        return dict(adopted=bool(q['adopted']),harm=harm,harmful_adopted=bool(q['adopted'] and harm),
                    execution_failed=bool(q['failed']),gain=100*(b['team_cost']-q['team_cost'])/b['team_cost'])
    results={};paired=[];zero_equal=0
    for method in plan['methods']:
        counts={v:dict(initials=0,adopted=0,harmful=0,harmful_adopted=0,execution_failures=0,gain_sum=0.) for v in ('comparator','history')}
        discordance=dict(harm_removed=0,harm_introduced=0,harm_retained=0,harmful_adoption_removed=0,harmful_adoption_introduced=0)
        for case in plan['cases']:
            key=(case['cluster_id'],method);base,bmotion=primary[case['cluster_id'],'baseline'];orow,omotion=old[key];nrow,_=primary[key]
            a=outcome(orow,base);b=outcome(nrow,base)
            if not a['adopted'] and not a['execution_failed'] and not episode_outcome(base)['failed']:
                if omotion!=bmotion:raise ValueError('Old zero-adoption trajectory differs from baseline')
                zero_equal+=1
            for variant,q in (('comparator',a),('history',b)):
                c=counts[variant];c['initials']+=1;c['adopted']+=int(q['adopted']);c['harmful']+=int(q['harm'])
                c['harmful_adopted']+=int(q['harmful_adopted']);c['execution_failures']+=int(q['execution_failed']);c['gain_sum']+=q['gain']
            discordance['harm_removed']+=int(a['harm'] and not b['harm']);discordance['harm_introduced']+=int(not a['harm'] and b['harm'])
            discordance['harm_retained']+=int(a['harm'] and b['harm'])
            discordance['harmful_adoption_removed']+=int(a['harmful_adopted'] and not b['harmful_adopted'])
            discordance['harmful_adoption_introduced']+=int(not a['harmful_adopted'] and b['harmful_adopted'])
            paired.append(dict(cluster_id=case['cluster_id'],method=method,plant=case['plant'],scenario=case['scenario'],
                disturbance=case['disturbance'],comparator=a,history=b))
        for q in counts.values():q['mean_paired_percentage_gain']=q.pop('gain_sum')/q['initials']
        results[method]=dict(methods=counts,paired_discordance=discordance,
            mean_paired_gain_difference_history_minus_comparator=counts['history']['mean_paired_percentage_gain']-counts['comparator']['mean_paired_percentage_gain'])
    e.write(OUT/'paired_outcomes.json',paired)
    e.write(OUT/'report.json',dict(completed=True,actual_additional_sumo_executions=3072,original_confirmation_initials=1024,
        new_independent_initials=0,new_error_allocated=False,statistical_certificate=False,primary_study_modified=False,
        descriptive_comparison_only=True,no_paired_wall_time_speedup_claim=True,comparator_variant=VARIANT,children=receipts,
        results=results,old_successful_zero_adoption_physical_fallbacks_exact=zero_equal,
        primary_certificate_sha256=e.sha(e.FORMAL/'analysis/certificate.json'),plan_sha256=e.sha(OUT/'plan.json'),
        paired_outcomes_sha256=e.sha(OUT/'paired_outcomes.json')))
    print(json.dumps(dict(completed=True,actual_additional_sumo_executions=3072,results=results)),flush=True)


def run():
    bound(True);execution=OUT/'execution';execution.mkdir(exist_ok=False);children=[];receipts=[]
    for shard in range(4):
        args=command(Path(__file__),'worker','--shard',shard,'--variant',VARIANT);stream=(execution/f'shard{shard}.log').open('xb')
        child=subprocess.Popen(args,cwd=e.ROOT,stdout=stream,stderr=subprocess.STDOUT)
        e.write(execution/f'shard{shard}_started.json',dict(native_pid=child.pid,command=args))
        children.append((shard,child,stream))
    for shard,child,stream in children:
        code=child.wait();stream.close();receipt=dict(shard=shard,native_pid=child.pid,actual_native_exit_code=code)
        if code==0:
            if e.read(OUT/f'shard{shard}/report.json')['native_pid']!=child.pid:raise ValueError('Native PID differs')
            receipt['report_sha256']=e.sha(OUT/f'shard{shard}/report.json')
        e.write(execution/f'shard{shard}_native_exit.json',receipt);receipts.append(receipt)
    if any(r['actual_native_exit_code']!=0 for r in receipts):raise RuntimeError('Retain incomplete secondary study;never replace samples')
    finalize(receipts)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('bind','run','worker'));p.add_argument('--shard',type=int,choices=range(4));p.add_argument('--variant',choices=('compact','phase'),required=True);a=p.parse_args()
    VARIANT=a.variant;OUT=e.BASE/('powered_history_'+VARIANT+'_holdout_ablation_20261103')
    if a.action=='bind':bind()
    elif a.action=='run':run()
    else:worker(a.shard)
