"""Sealed-block fixed-N execution;adverse controller outcomes never shorten N.

Formal execution remains disabled until a separate complete registration and
allocation binding exists. Validation uses new explicit nonformal fixtures.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from coopgate.episode_store_v21 import EpisodeStore,observations,inventory
from coopgate.mappo_deployment import load_deployment_actor
from coopgate.episode_thinning_control import matching_probability
from coopgate.robust_v9 import robust_certificate
from coopgate.fast_v8 import parameters
from coopgate.merge_v4 import MergeConfig
import mappo_formal_design as design
import mappo_formal_runtime as runtime
from mappo_formal_common import planned_initial,canonical_sha256
from mappo_joint_data import episode_outcome

ROOT=design.ROOT;BASE=design.BASE
FORMAL=BASE/'formal_mappo_attempt3_20261102'
VALIDATION=BASE/'development/formal_adapter_validation_20261102'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def load_models():
    candidate=design.validate_candidate();models={}
    for seed in ('4703','4704','4705'):
        model,identity=load_deployment_actor(BASE/f'development/joint_{seed}/policy.pt')
        if identity!=candidate['actor_identities'][seed]:raise ValueError('Actor or on-policy audit identity changed')
        models[seed]=model
    return models


def source_files():
    sources=list((ROOT/'src/coopgate').glob('*.py'))
    sources+=[ROOT/'tools'/n for n in ('mappo_formal_common.py','mappo_formal_design.py','mappo_formal_runtime.py',
        'mappo_formal_experiment.py','mappo_risk_attempt3.py','mappo_joint_data.py','build_mappo_formal_adapters.py')]
    sources += [ROOT/'tools/mappo_native_launch.py',ROOT/'tools/run_mappo_formal_validation.py']
    sources += [ROOT/'tests/test_mappo_formal_adapters.py',ROOT/'tests/test_mappo_formal_failures.py',ROOT/'docs/mappo_formal_attempt3_20261102_protocol.md']
    return sources


def build(validation=False):
    out=VALIDATION if validation else FORMAL;out.mkdir(parents=True,exist_ok=False)
    candidate=design.validate_candidate();models=load_models()
    cases={'validation':design.validation_cases()} if validation else {s:design.stage_cases(s) for s in design.COUNTS}
    invalid=[];packed=parameters(MergeConfig())
    for stage,items in cases.items():
        for c in items:
            if not robust_certificate(planned_initial(c['seed'],c['scenario']),np.zeros(3),np.full(3,-2.5),packed,.5):
                invalid.append([stage,c['index'],c['seed']])
    sources={p.relative_to(ROOT).as_posix():sha(p) for p in source_files()}
    for name,digest in sources.items():
        data=subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)
        if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Commit frozen source before registration: '+name)
    plan=dict(schema='coopgate.genuine-mappo.formal-v1',candidate_sha256=sha(BASE/'gain_margin_fixed_candidate_20261102.json'),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_sha256=sources,
        cases=cases,validation_only=validation,error_allocated=False,allocation_required_before_formal_collection=True,
        reset_domain=dict(checked=sum(map(len,cases.values())),invalid=invalid,rejected_or_replaced=0),
        main_sumo_runs=0 if validation else 16384,total_sumo_runs=sum(len(c['methods']) for items in cases.values() for c in items),
        formal_contract=dict(calibration_initials=3072,confirmation_initials=1024,mixture_risk=.05,strong_lag_risk=.05,
            adoption_lower=.20,paired_gain_lower_strict_percent=.5,zero_observed_failures=True,
            old_stratified_joint_bound_preserved=True,confirmation_runs_if_calibration_fails=True,
            controller_selection_on_formal_data=False,full_callback_cost=True,communication_latency=False,
            hard_real_time_claim=False),
        matching='Calibration changes independent thinning probabilities only;never target policy/threshold. Invalid matching is marked inactive,never clipped;confirmation main runs still complete.')
    write(out/'plan.json',plan)
    if invalid:raise ValueError('Reset-domain check failed;NO seed was rejected/replaced and NO formal run started')
    print(json.dumps(dict(root=str(out),executions=plan['total_sumo_runs'],validation_only=validation,error_allocated=False)),flush=True)


def bound(root,validation):
    plan=read(root/'plan.json')
    if plan['validation_only']!=validation or plan['reset_domain']['invalid']:raise ValueError('Invalid study domain')
    if plan['candidate_sha256']!=sha(BASE/'gain_margin_fixed_candidate_20261102.json'):raise ValueError('Candidate binding changed')
    design.validate_candidate()
    for name,digest in plan['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen study source changed: '+name)
    if not validation:
        binding=read(root/'binding.json');allocation=read(root/'allocation.json')
        if binding['plan_sha256']!=sha(root/'plan.json') or binding['allocation_sha256']!=sha(root/'allocation.json'):
            raise ValueError('Formal plan/allocation binding changed')
        if allocation['attempt']!=3 or allocation['allocated_error']!=.05/12 or not allocation['predecessor_verified']:
            raise ValueError('No independently audited new allocation')
    return plan


def stage_cases(root,stage,plan):
    if stage not in plan['cases']:raise ValueError('Stage not in frozen plan')
    cases=plan['cases'][stage]
    if stage=='confirmation':
        controls=read(root/'matching_controls.json')
        if controls['calibration_terminal_verified'] is not True:raise ValueError('Matching controls precede completed calibration')
        cases=design.stage_cases('confirmation',controls['probabilities'])
        if [{k:v for k,v in c.items() if k!='controller_coins'} for c in cases]!=plan['cases'][stage]:
            raise ValueError('Confirmation case changed beyond independent controller coins')
    return cases


def stage_path(root,stage,shard):return root/stage/f'shard{shard}'


def blocks_for(cases,shard,shards):
    if shards not in (1,4) or not 0<=shard<shards:raise ValueError('Explicit one/four-way fixed partition required')
    selected=cases[shard::shards]
    return [selected[i:i+8] for i in range(0,len(selected),8)]


def collect(stage,shard,validation=False):
    root=VALIDATION if validation else FORMAL;plan=bound(root,validation);torch.set_num_threads(1)
    models=load_models();cases=stage_cases(root,stage,plan);shards=1 if validation else 4
    out=stage_path(root,stage,shard);out.mkdir(parents=True,exist_ok=False)
    assigned=blocks_for(cases,shard,shards);manifest=[];count=0;started=time.perf_counter()
    profiles=sorted({c['plant'] for block in assigned for c in block})
    warm=np.array([[-140.,12.],[-220.,10.],[-190.,12.]])
    for profile in profiles:
        sample=dict(plant=profile,controller_coins=dict.fromkeys(models,True))
        for method in cases[0]['methods']:
            model=None if method=='baseline' else models[method.split('_')[-1]]
            runtime.make_controller(method,model,sample).select(warm,np.zeros(3))
    with ExitStack() as stack:
        envs={p:stack.enter_context(runtime.MatchedShiftedMergeEnv(plant=p)) for p in profiles}
        for block_index,block in enumerate(assigned):
            directory=out/f'block{block_index:04d}';directory.mkdir(exist_ok=False)
            metadata=dict(plan_sha256=sha(root/'plan.json'),stage=stage,shard=shard,index=block_index,
                cases_sha256=canonical_sha256(block))
            store=EpisodeStore(directory/'observations.sqlite',metadata)
            expected=[[c['cluster_id'],m] for c in block for m in c['methods']]
            try:
                for c in block:
                    for method in c['methods']:
                        model=None if method=='baseline' else models[method.split('_')[-1]]
                        record=runtime.execute_episode(c,method,model,envs[c['plant']])
                        # Commit adverse/capped outcomes BEFORE checking or summarizing.
                        store.append([c['cluster_id'],method],{k:v for k,v in record.items() if k!='trace'},record['trace'])
                        runtime.check_calls(record['row'],record['trace'],record['calls']);episode_outcome(record['row']);count+=1
                store.seal(expected)
            finally:store.close()
            manifest.append(dict(block=directory.name,metadata=metadata,keys=expected,database_sha256=sha(directory/'observations.sqlite'),
                seal_sha256=sha(directory/'observations.sqlite.seal.json')))
            write(directory/'collection_complete.json',manifest[-1])
            bound(root,validation)
            print(json.dumps(dict(stage=stage,shard=shard,completed=count,blocks=block_index+1,elapsed_s=time.perf_counter()-started)),flush=True)
    write(out/'collection_complete.json',dict(blocks=manifest,executions=count,expected=sum(len(c['methods']) for b in assigned for c in b),native_pid=os.getpid(),
        plan_sha256=sha(root/'plan.json'),elapsed_s=time.perf_counter()-started))


def reviewed_records(root,stage,shard,plan):
    cases=stage_cases(root,stage,plan);by_id={c['cluster_id']:c for c in cases};out=stage_path(root,stage,shard)
    complete=read(out/'collection_complete.json')
    assigned=blocks_for(cases,shard,1 if plan['validation_only'] else 4)
    if len(complete['blocks'])!=len(assigned):raise ValueError('Missing/duplicate scheduled block')
    for block_index,(block,planned) in enumerate(zip(complete['blocks'],assigned)):
        expected=[[c['cluster_id'],m] for c in planned for m in c['methods']]
        expected_metadata=dict(plan_sha256=sha(root/'plan.json'),stage=stage,shard=shard,index=block_index,cases_sha256=canonical_sha256(planned))
        if block['block']!=f'block{block_index:04d}' or block['keys']!=expected or block['metadata']!=expected_metadata:
            raise ValueError('Block differs from the complete outcome-free case/method schedule')
        directory=out/block['block']
        if sha(directory/'observations.sqlite')!=block['database_sha256'] or sha(directory/'observations.sqlite.seal.json')!=block['seal_sha256']:
            raise ValueError('Sealed episode block changed')
        for _,record,trace in observations(directory/'observations.sqlite',block['metadata'],block['keys']):
            record['trace']=trace;yield by_id[record['cluster_id']],record


def review(stage,shard,validation=False):
    root=VALIDATION if validation else FORMAL;plan=bound(root,validation);torch.set_num_threads(1)
    # Owning launcher must publish an actual native collection exit witness.
    terminal=read(stage_path(root,stage,shard)/'collect_native_exit.json')
    collected=read(stage_path(root,stage,shard)/'collection_complete.json')
    if (terminal['actual_native_exit_code']!=0 or terminal['native_pid']!=collected['native_pid']
            or terminal['plan_sha256']!=sha(root/'plan.json')
            or terminal['collection_complete_sha256']!=sha(stage_path(root,stage,shard)/'collection_complete.json')):
        raise ValueError('No matching actual zero native exit for this exact collected stage')
    models=load_models();count=commands=substeps=failures=0
    for c,r in reviewed_records(root,stage,shard,plan):
        method=r['method'];model=None if method=='baseline' else models[method.split('_')[-1]]
        if method not in c['methods']:raise ValueError('Method absent from exact stage case')
        proof=runtime.audit_record(c,r,model);count+=1;commands+=proof['reference_replayed_commands'];substeps+=proof['physical_substeps'];failures+=proof['recorded_failure']
        if count%128==0:print(json.dumps(dict(stage=stage,shard=shard,reviewed=count)),flush=True)
    out=stage_path(root,stage,shard);complete=read(out/'collection_complete.json')
    if count!=complete['expected'] or count!=complete['executions']:raise ValueError('Incomplete fixed-N review')
    bound(root,validation)
    write(out/'verification.json',dict(all_recorded_outcomes_replay=True,executions=count,commands=commands,physical_substeps=substeps,
        accurately_recorded_failures_retained=failures,allocation_spent_by_validation=False if validation else None,
        collection_complete_sha256=sha(out/'collection_complete.json')))
    print(json.dumps(dict(stage=stage,shard=shard,reviewed=count,recorded_failures=failures)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('build','collect','review'));p.add_argument('--stage');p.add_argument('--shard',type=int,default=0);p.add_argument('--validation',action='store_true');a=p.parse_args()
    if a.action=='build':build(a.validation)
    else:(collect if a.action=='collect' else review)(a.stage,a.shard,a.validation)
