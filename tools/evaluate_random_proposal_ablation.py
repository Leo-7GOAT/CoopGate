"""Fixed policy-free development control;one owned low-priority SUMO worker."""
import argparse,json,os,subprocess,time
from contextlib import ExitStack
from pathlib import Path
import numpy as np
import torch
from coopgate.episode_store_v21 import EpisodeStore,observations
from coopgate.observed_history_gate import ObservedHistoryGate
import mappo_coupled_formal_experiment as e
import mappo_random_proposal_runtime as runtime
from mappo_native_launch import command
from mappo_coupled_joint_data import episode_outcome

PARENT=e.BASE/'development/history_power_pilot_20261103'
OUT=e.BASE/'development/random_proposal_ablation_20261103'
METHODS=('random_0','random_1','random_2')
EXTRA=('tools/evaluate_random_proposal_ablation.py','tools/mappo_random_proposal_runtime.py','docs/random_proposal_ablation_20261103_protocol.md')


class RandomLocalProposals:
    def __init__(self,controller_seed):
        self.rngs=[np.random.default_rng(np.random.SeedSequence([int(controller_seed),role])) for role in range(3)]
    def proposal(self,observations,mask):
        return np.array([rng.choice(np.flatnonzero(mask[role])) for role,rng in enumerate(self.rngs)],dtype=np.int64)


def factory(method,unused,case):
    if method not in METHODS:raise ValueError('Only explicitly random controls are collected here')
    return ObservedHistoryGate(RandomLocalProposals(case['controller_seeds'][method]),case['plant'],runtime.COUPLED_BANK)


def prepare():
    owner=e.read(PARENT/'owner_native_exit.json')
    if owner['actual_native_exit_code']!=0 or owner['report_sha256']!=e.sha(PARENT/'report.json'):raise ValueError('Completed untouched parent required')
    original=e.read(PARENT/'plan.json');cases=[];rng=np.random.default_rng(550701)
    for case in original['cases']:
        case=dict(case);case['controller_seeds']={m:int(rng.integers(0,2**32)) for m in METHODS};cases.append(case)
    if len(cases)!=320:raise ValueError('All320 reused initials required')
    OUT.mkdir(exist_ok=False);sources=dict(original['sources']);sources.update({p:e.sha(e.ROOT/p) for p in EXTRA})
    e.write(OUT/'plan.json',dict(cases=cases,methods=METHODS,controller_stream=550701,sources=sources,
        parent_report_sha256=e.sha(PARENT/'report.json'),expected_actual_sumo_executions=960,
        new_independent_initials=0,error_allocated=False,actor_checkpoints_used=False,
        policy_kind='Independent per-role uniform draws from legal masks',primary_study_changed=False))


def guard():
    plan=e.read(OUT/'plan.json')
    if any(e.sha(e.ROOT/p)!=h for p,h in plan['sources'].items()) or plan['parent_report_sha256']!=e.sha(PARENT/'report.json'):
        raise ValueError('Frozen random control or completed parent changed')
    return plan


def worker():
    torch.set_num_threads(1);plan=guard();blocks=[];started=time.perf_counter();root=OUT/'worker';root.mkdir(exist_ok=False)
    with ExitStack() as stack:
        envs={p:stack.enter_context(runtime.MatchedShiftedMergeEnv(plant=p)) for p in e.design.PLANTS}
        for i,case in enumerate(plan['cases']):
            guard();directory=root/f'block{i:03d}';directory.mkdir(exist_ok=False)
            methods=METHODS[i%3:]+METHODS[:i%3];keys=[[case['cluster_id'],m] for m in methods]
            metadata=dict(plan_sha256=e.sha(OUT/'plan.json'),case=case,index=i)
            store=EpisodeStore(directory/'observations.sqlite',metadata)
            try:
                for method in methods:
                    r=runtime.execute_episode(case,method,None,envs[case['plant']],controller_factory=factory)
                    row={k:v for k,v in r.items() if k!='trace'};row['policy_kind']='uniform_random_legal_proposals'
                    store.append([case['cluster_id'],method],row,r['trace'])
                store.seal(keys)
            finally:store.close()
            for _,r,trace in observations(directory/'observations.sqlite',metadata,keys):
                r['trace']=trace;runtime.audit_record(case,r,None,reference_factory=factory);episode_outcome(r['row'])
            entry=dict(block=directory.name,metadata=metadata,keys=keys,database_sha256=e.sha(directory/'observations.sqlite'),seal_sha256=e.sha(directory/'observations.sqlite.seal.json'))
            e.write(directory/'reviewed.json',entry);blocks.append(entry)
            print(json.dumps(dict(complete_initials=i+1,expected_initials=320,actual_sumo_executions=3*(i+1))),flush=True)
    e.write(root/'report.json',dict(native_pid=os.getpid(),blocks=blocks,actual_sumo_executions=960,elapsed_s=time.perf_counter()-started))


def finalize(receipt):
    plan=guard();parent={};totals={};zero=records=0;adverse=[]
    for shard in range(4):
        for b in e.read(PARENT/f'shard{shard}/report.json')['blocks']:
            db=PARENT/f'shard{shard}'/b['block']/'observations.sqlite'
            if e.sha(db)!=b['database_sha256']:raise ValueError('Parent data changed')
            for _,r,t in observations(db,b['metadata'],b['keys']):
                if r['method']=='baseline':parent[r['cluster_id']]=(r['row'],t)
    for b in e.read(OUT/'worker/report.json')['blocks']:
        db=OUT/'worker'/b['block']/'observations.sqlite'
        if e.sha(db)!=b['database_sha256']:raise ValueError('Reviewed random data changed')
        for _,r,trace in observations(db,b['metadata'],b['keys']):
            records+=1;base,bt=parent[r['cluster_id']];q=episode_outcome(r['row']);z=episode_outcome(base)
            harm=bool(q['failed'] or z['failed'] or np.any(q['costs']>z['costs']+1e-9) or np.any(q['arrivals']>z['arrivals']+1e-9))
            result=totals.setdefault(r['method'],dict(initials=0,adopted=0,harmful_adopted=0,execution_failures=0,gain_sum=0.,callback_ms=[]))
            result['initials']+=1;result['adopted']+=int(q['adopted']);result['harmful_adopted']+=int(q['adopted'] and harm);result['execution_failures']+=int(q['failed'])
            result['gain_sum']+=100*(z['team_cost']-q['team_cost'])/z['team_cost'];result['callback_ms'].extend(c['wall_ms'] for c in r['calls'])
            if not q['adopted'] and not q['failed'] and not z['failed']:
                if [(t['t'],t['command'],t['x']) for t in trace]!=[(t['t'],t['command'],t['x']) for t in bt]:raise ValueError('Zero-adoption random trajectory differs from actual baseline')
                zero+=1
            if harm:adverse.append(dict(case=b['metadata']['case'],method=r['method'],adopted=q['adopted'],failed=q['failed'],cost_differences=(q['costs']-z['costs']).tolist(),arrival_differences=(q['arrivals']-z['arrivals']).tolist()))
    if records!=960 or len(parent)!=320 or any(x['initials']!=320 for x in totals.values()):raise ValueError('Missing complete random control scope')
    for r in totals.values():
        values=np.array(r.pop('callback_ms'));r['mean_paired_percentage_gain']=r.pop('gain_sum')/r['initials']
        r.update(callbacks=len(values),callback_p95_ms=float(np.quantile(values,.95)),callback_max_ms=float(values.max()),callback_over_500ms=int((values>500).sum()))
    e.write(OUT/'all_adverse.json',adverse)
    e.write(OUT/'report.json',dict(completed=True,actual_additional_sumo_executions=960,reused_independent_initials=320,new_independent_initials=0,
        new_error_allocated=False,statistical_certificate=False,actor_checkpoints_used=False,results=totals,
        original_mappo_results=e.read(PARENT/'report.json')['results'],actual_native_worker=receipt,
        successful_zero_adoption_paths_exact_baseline=zero,no_paired_wall_time_speedup_claim=True))
    print(json.dumps(dict(completed=True,actual_sumo_executions=960,results=totals)),flush=True)


def run():
    prepare()
    with (OUT/'worker.log').open('xb') as log:
        child=subprocess.Popen(command(__file__,'--worker'),cwd=e.ROOT,stdout=log,stderr=subprocess.STDOUT,
            creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name=='nt' else 0)
        e.write(OUT/'worker_started.json',dict(native_pid=child.pid,priority='below_normal',worker_count=1));code=child.wait()
    receipt=dict(native_pid=child.pid,actual_native_exit_code=code);e.write(OUT/'worker_native_exit.json',receipt)
    if code:raise RuntimeError('Retain incomplete random control;do not resample or replace')
    finalize(receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');a=p.parse_args();worker() if a.worker else run()
