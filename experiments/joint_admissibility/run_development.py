"""Independent integrated-search development: all known adverse initials plus80 new cases.

Actual SUMO, baseline plus both gates with all three unchanged MAPPO actors.
Every episode is sealed before independently replaying its complete controller
and physical trajectory. Adverse/aborted outcomes are never dropped.
"""
import argparse
from contextlib import ExitStack
from itertools import product
import os
from pathlib import Path
import subprocess
import sys
import time
import json
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from coopgate.observed_history_gate import ObservedHistoryGate
from experiments.joint_admissibility.gate import JointAdmissibilityGate
from coopgate.episode_store_v21 import EpisodeStore,observations
from coopgate.robust_v9 import robust_certificate
from coopgate.fast_v8 import parameters
from coopgate.merge_v4 import MergeConfig
from coopgate.disturbance_v9 import disturbance_tape
from coopgate.matched_shift_v21 import matched_plant_parameters
import mappo_coupled_formal_experiment as e
import mappo_coupled_formal_runtime as runtime
from mappo_formal_common import planned_initial,array_sha256,canonical_sha256
from mappo_coupled_joint_data import episode_outcome
from select_mappo_coupled_bank import DEST as BANK
DIAG=e.BASE/'formal_mappo_attempt4_diagnosis_20261102/all_false_admissions.json'
from mappo_native_launch import command
from datetime import datetime,timezone

OUT=e.BASE/'development/joint_admissibility_development_20261103'
VARIANTS=[('baseline','baseline')]+[(v,s) for v in ('history','joint') for s in ('4703','4704','4705')]
EXTRA=['experiments/joint_admissibility/run_development.py',
       'experiments/joint_admissibility/gate.py',
       'experiments/joint_admissibility/__init__.py',
       'tests/test_joint_candidate_admissibility.py',
       'src/coopgate/observed_history_gate.py',
       'src/coopgate/observed_lag_bias_fit.py',
       'docs/joint_admissibility_development_20261103_protocol.md']
DIAG5=e.BASE/'formal_mappo_attempt5_powered_diagnosis_20261103/all_false_admissions.json'



def cases():
    values=[]
    for index,(scene,plant,mode) in enumerate(product(e.design.SCENES,e.design.PLANTS,e.design.MODES)):
        values.append(dict(stage='fresh',cluster_id=f'joint:fresh:{index:03d}',seed=879000001+index,
                           scenario=scene,plant=plant,disturbance=mode))
    seen=set()
    for diagnosis in (DIAG,DIAG5):
        for issue in e.read(diagnosis)['all_target_harmful_adoptions']:
            key=(issue['seed'],issue['scenario'],issue['plant'],issue['disturbance'])
            if key in seen:continue
            seen.add(key)
            values.append(dict(stage='regression',cluster_id=f'joint:regression:{len(seen):03d}',
                seed=issue['seed'],scenario=issue['scenario'],plant=issue['plant'],disturbance=issue['disturbance']))
    for case in values:
        case.update(initial_sha256=array_sha256(planned_initial(case['seed'],case['scenario']),decimals=8),
            dynamics_sha256=canonical_sha256(matched_plant_parameters(case['seed'],case['plant'],.1)),
            disturbance_sha256=array_sha256(disturbance_tape(case['seed'],case['disturbance'],.1,250)))
    if len(values)!=80+len(seen) or len(seen)<12:raise ValueError('Every known adverse initial and all80 fresh cases required')
    return values


def freeze():
    from mappo_completed_powered_attempt5 import complete_evidence
    complete_evidence()
    previous=e.bound(e.FORMAL,False)
    sources=dict(previous['source_sha256'])
    sources.update({p:e.sha(e.ROOT/p) for p in EXTRA})
    items=cases();packed=parameters(MergeConfig())
    invalid=[c['cluster_id'] for c in items if not robust_certificate(planned_initial(c['seed'],c['scenario']),np.zeros(3),np.full(3,-2.5),packed,.5)]
    OUT.mkdir(parents=True,exist_ok=False)
    e.write(OUT/'plan.json',dict(kind='fixed_whole_closed_loop_development',cases=items,variants=VARIANTS,
        sources=sources,bank_sha256=e.sha(BANK),diagnosis_sha256=e.sha(DIAG),diagnosis5_sha256=e.sha(DIAG5),
        expected_executions=7*len(items),fresh_initials=80,known_regression_initials=len(items)-80,invalid_reset_initials=invalid,
        error_allocated=False,new_inference=False,deployment_certified=False,registered_utc=datetime.now(timezone.utc).isoformat()))
    if invalid:raise ValueError('Invalid reset;retained full schedule,no seed replacement or SUMO started')


def guard():
    plan=e.read(OUT/'plan.json')
    if any(e.sha(e.ROOT/p)!=h for p,h in plan['sources'].items()) or e.sha(BANK)!=plan['bank_sha256'] or e.sha(DIAG)!=plan['diagnosis_sha256'] or e.sha(DIAG5)!=plan['diagnosis5_sha256']:
        raise ValueError('Frozen source/bank/diagnosis changed')
    return plan


def factory(variant):
    def make(method,model,case):
        if variant=='history':return ObservedHistoryGate(model,case['plant'],BANK)
        if variant=='joint':return JointAdmissibilityGate(model,case['plant'],BANK)
        return runtime.make_controller(method,model,case)
    return make


def worker(shard):
    torch.set_num_threads(1);plan=guard();models=e.load_models();started=time.perf_counter()
    root=OUT/f'shard{shard}';root.mkdir(exist_ok=False);assigned=plan['cases'][shard::4];blocks=[]
    warm=np.array([[-140.,12.],[-220.,10.],[-190.,12.]])
    for profile in e.design.PLANTS:
        for variant,method in VARIANTS:
            factory(variant)(method,None if method=='baseline' else models[method],dict(plant=profile)).select(warm,np.zeros(3))
    with ExitStack() as stack:
        envs={p:stack.enter_context(runtime.MatchedShiftedMergeEnv(plant=p)) for p in e.design.PLANTS}
        for index,case in enumerate(assigned):
            guard();directory=root/f'block{index:03d}';directory.mkdir(exist_ok=False)
            # Rotate order without outcome-dependent scheduling.
            offset=index%len(VARIANTS);ordered=VARIANTS[offset:]+VARIANTS[:offset]
            expected=[[case['cluster_id'],v,m] for v,m in ordered]
            metadata=dict(plan_sha256=e.sha(OUT/'plan.json'),shard=shard,index=index,case=case)
            store=EpisodeStore(directory/'observations.sqlite',metadata)
            try:
                for variant,method in ordered:
                    model=None if method=='baseline' else models[method]
                    record=runtime.execute_episode(case,method,model,envs[case['plant']],controller_factory=factory(variant))
                    payload={k:v for k,v in record.items() if k!='trace'};payload['variant']=variant
                    store.append([case['cluster_id'],variant,method],payload,record['trace'])
                store.seal(expected)
            finally:store.close()
            # Independent exact replay after the entire paired block is durable.
            for key,record,trace in observations(directory/'observations.sqlite',metadata,expected):
                record['trace']=trace;model=None if record['method']=='baseline' else models[record['method']]
                runtime.audit_record(case,record,model,reference_factory=factory(record['variant']))
                episode_outcome(record['row'])
            blocks.append(dict(block=directory.name,metadata=metadata,keys=expected,
                database_sha256=e.sha(directory/'observations.sqlite'),seal_sha256=e.sha(directory/'observations.sqlite.seal.json'),
                complete_controller_and_physics_replay=True))
            e.write(directory/'reviewed.json',blocks[-1])
            print(json.dumps(dict(shard=shard,completed_initials=index+1,expected=len(assigned),sumo_executions=7*(index+1),elapsed_s=time.perf_counter()-started)),flush=True)
    e.write(root/'report.json',dict(native_pid=os.getpid(),blocks=blocks,sumo_executions=len(assigned)*7,elapsed_s=time.perf_counter()-started))


def finalize(receipts):
    plan=guard();totals={};timings={};adverse=[];zero_equal=0;executions=0
    for shard in range(4):
        shard_report=e.read(OUT/f'shard{shard}/report.json')
        for block in shard_report['blocks']:
            db=OUT/f'shard{shard}'/block['block']/'observations.sqlite'
            if e.sha(db)!=block['database_sha256']:raise ValueError('Reviewed raw block changed')
            records={}
            for key,record,trace in observations(db,block['metadata'],block['keys']):
                record['trace']=trace;records[(record['variant'],record['method'])]=record;executions+=1
            case=block['metadata']['case'];base=records['baseline','baseline'];b=episode_outcome(base['row'])
            for (variant,method),record in records.items():
                row=record['row'];actual=episode_outcome(row)
                harm=bool(actual['failed'] or b['failed'] or np.any(actual['costs']>b['costs']+1e-9) or np.any(actual['arrivals']>b['arrivals']+1e-9))
                gain=100*(b['team_cost']-actual['team_cost'])/b['team_cost']
                name=case['stage']+':'+variant+':'+method
                r=totals.setdefault(name,dict(initials=0,adopted=0,harmful_adopted=0,execution_failures=0,gain_sum=0.))
                r['initials']+=1;r['adopted']+=int(actual['adopted']);r['harmful_adopted']+=int(actual['adopted'] and harm)
                r['execution_failures']+=int(actual['failed']);r['gain_sum']+=gain
                timings.setdefault(name,[]).extend(c['wall_ms'] for c in record['calls'])
                if variant!='baseline' and not actual['adopted'] and not actual['failed'] and not b['failed']:
                    if len(record['trace'])!=len(base['trace']) or any(a['t']!=z['t'] or a['command']!=z['command'] or a['x']!=z['x'] for a,z in zip(record['trace'],base['trace'])):
                        raise ValueError('Zero-adoption fallback differs physically from baseline')
                    zero_equal+=1
                if variant!='baseline' and (actual['failed'] or harm):
                    adverse.append(dict(case=case,variant=variant,method=method,adopted=actual['adopted'],failed=actual['failed'],
                        cost_differences=(actual['costs']-b['costs']).tolist(),arrival_differences=(actual['arrivals']-b['arrivals']).tolist(),
                        source_shard=shard,source_block=block['block']))
    if executions!=plan['expected_executions']:raise ValueError('Missing fixed executions')
    for name,r in totals.items():
        r['mean_paired_percentage_gain']=r.pop('gain_sum')/r['initials'];t=np.asarray(timings[name]);r['callback_calls']=len(t)
        r.update(callback_p95_ms=float(np.quantile(t,.95)),callback_max_ms=float(t.max()),callback_over_500ms=int((t>500).sum()))
    e.write(OUT/'all_adverse.json',adverse)
    e.write(OUT/'report.json',dict(completed=True,actual_sumo_executions=executions,children=receipts,
        results=totals,zero_adoption_exact_physical_fallbacks=zero_equal,all_raw_blocks_sealed_and_exact_replayed=True,
        new_independent_development_initials=80,known_regression_initials=plan['known_regression_initials'],
        statistical_certificate=False,alpha_allocated=False,deployment_certified=False))
    print(json.dumps(dict(completed=True,executions=executions,results=totals)),flush=True)


def run():
    freeze();children=[]
    for shard in range(4):
        stream=(OUT/f'shard{shard}.log').open('xb')
        child=subprocess.Popen(command(__file__,'--shard',shard),cwd=e.ROOT,stdout=stream,stderr=subprocess.STDOUT)
        e.write(OUT/f'shard{shard}_started.json',dict(native_pid=child.pid,owner_pid=os.getpid()))
        children.append((shard,child,stream))
    receipts=[]
    for shard,child,stream in children:
        code=child.wait();stream.close();receipt=dict(shard=shard,native_pid=child.pid,actual_native_exit_code=code)
        e.write(OUT/f'shard{shard}_native_exit.json',receipt);receipts.append(receipt)
    if any(r['actual_native_exit_code']!=0 for r in receipts):raise RuntimeError('Development failed;every partial block and log retained')
    finalize(receipts)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--shard',type=int,choices=range(4));args=parser.parse_args()
    run() if args.shard is None else worker(args.shard)
