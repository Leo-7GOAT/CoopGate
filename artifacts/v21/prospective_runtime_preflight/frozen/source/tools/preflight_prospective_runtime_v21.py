"""Actual SUMO integration on one previously used development initial.

64 collection runs, 64 unchanged legacy-reference runs and 16 observed-state
nominal shadow runs. These are implementation checks, not fresh calibration.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from coopgate.disturbance_v9 import DisturbedMergeEnv
from coopgate.episode_store_v21 import EpisodeStore, observations
from coopgate.observable_initial_v19 import ObservableInitialMergeEnv
from experiment_ego_v13 import run_episode
from probe_disturbed_gate_v19 import digest,save
from recover_checkpoint_v19 import atomic_bytes
from recover_interrupted_v21 import compare_replay
from resume_frozen_risk_v19 import build_record
from start_final_v19 import require_committed,verify_manifest
from study_gate_v19 import read_protocol,regime_for
from study_matched_transfer_v21 import factories
from study_phase_repair_v21 import validate_worker_exits
from study_repair_v20 import trace_metadata
from prospective_runtime_v21 import IndependentReview,execute,method_definitions,record,verify_record

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'artifacts/v21/prospective_runtime_preflight'
FROZEN=OUTPUT/'frozen'
EXECUTION=OUTPUT/'execution'
SOURCES=('tools/prospective_runtime_v21.py','tools/preflight_prospective_runtime_v21.py',
         'src/coopgate/episode_store_v21.py','tests/test_prospective_runtime_v21.py',
         'tools/recover_interrupted_v21.py')


def utc():return datetime.now(timezone.utc).isoformat()


def read(path):return json.loads(Path(path).read_text())


def manifest(path):
    return {p.relative_to(path).as_posix():digest(p) for p in path.rglob('*') if p.is_file() and p.name!='sha256.json'}


def cases(plan,shard):
    spec=plan['protocol']['spec']
    return [(0,mode,name) for mi,mode in enumerate(spec['modes']) if mi%3==shard for name in spec['methods']]


def freeze():
    if OUTPUT.exists():raise FileExistsError('Preserve previous integration execution')
    _,old=read_protocol(ROOT/'artifacts/v21/matched_transfer')
    verify_manifest(ROOT/'artifacts/v20/temporal_repair_extension')
    historical=read(ROOT/'artifacts/v20/temporal_repair_extension/protocol.json')
    if historical['spec']['seed_base']!=63100001:raise ValueError('Integration initial must already have been used')
    sources=dict(old['source_sha256'])
    sources.update({p.relative_to(ROOT).as_posix():digest(p) for p in (ROOT/'src/coopgate').glob('*.py')})
    sources.update({name:digest(ROOT/name) for name in SOURCES})
    require_committed([ROOT/name for name in sources])
    spec=dict(data_role='integration_only',seed_base=63100001,n_initials=1,sampling='iid_mixture',
        modes=['clean','uniform','bias','pulse'],methods=method_definitions(),all_traces_saved=True,audit_initial_indices=[])
    plan=dict(purpose='Actual common-profile collection, storage, independent review and observed-only nominal prediction',
        protocol=dict(spec=spec,models=old['models']),source_sha256=sources,shards=3,
        collection_runs=64,legacy_reference_runs=64,observed_shadow_runs=16,actual_sumo_runs=144,
        independent_initials_added=0,new_calibration_data=False,new_risk_certified=False,
        historical_initial_protocol_sha256=digest(ROOT/'artifacts/v20/temporal_repair_extension/protocol.json'),
        source_rule='All sixteen methods and four modes, fixed before execution; retain failures and errors',
        shadow_rule='Clean mode only, initialize from observed initial, fixed internal engine seed; no actual future disturbance')
    FROZEN.mkdir(parents=True)
    save(FROZEN/'plan.json',plan)
    for name in sources:
        destination=FROZEN/'source'/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,destination)
    save(FROZEN/'sha256.json',manifest(FROZEN))
    print(json.dumps(dict(actual_sumo_runs=144,new_independent_initials=0,plan_sha256=digest(FROZEN/'plan.json'))),flush=True)


def inputs():
    verify_manifest(FROZEN)
    plan=read(FROZEN/'plan.json')
    require_committed([FROZEN/'plan.json',FROZEN/'sha256.json'])
    for name,expected in plan['source_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Frozen integration source changed')
    return plan


def worker(shard):
    torch.set_num_threads(1)
    plan=inputs()
    if shard not in range(3):raise ValueError('Unknown worker')
    protocol,spec=plan['protocol'],plan['protocol']['spec']
    out=OUTPUT/f'shard_{shard}'
    out.mkdir(exist_ok=False)
    fs=factories(protocol)
    warm=np.array([[-80.,12.],[-85.,13.],[-120.,11.]])
    for factory in fs.values():factory(warm).select(warm,np.zeros(3))
    assigned=cases(plan,shard)
    metadata=dict(role='integration_only',plan_sha256=digest(FROZEN/'plan.json'),shard=shard)
    store=EpisodeStore(out/'observations.sqlite',metadata)
    expected=[]
    review=IndependentReview(protocol)
    shadow_runs=0
    try:
        with ExitStack() as stack:
            env=stack.enter_context(DisturbedMergeEnv())
            reference_env=stack.enter_context(DisturbedMergeEnv())
            shadow_env=None
            for number,case in enumerate(assigned):
                index,mode,name=case
                actual,trace=execute(env,case,spec,fs[name])
                key=(*case,'collection')
                store.append(key,actual,trace);expected.append(key)
                review.check(actual,trace)
                # Direct old execution/accounting is independent of execute().
                raw,reference,initial=run_episode(reference_env,spec['seed_base'],regime_for(spec,index),mode,fs[name],True)
                legacy,_,_,_=build_record(raw,reference,initial,index,name,spec['methods'][name],spec)
                legacy.update(trace_metadata(reference))
                key=(*case,'legacy_reference')
                store.append(key,legacy,reference);expected.append(key)
                verify_record(legacy,reference,spec)
                compare_replay(legacy,actual,reference,trace)
                if mode=='clean':
                    if shadow_env is None:
                        shadow_env=stack.enter_context(ObservableInitialMergeEnv(np.asarray(actual['initial'])))
                    shadow,shadow_trace,observed=run_episode(shadow_env,1,'unused','clean',fs[name],True)
                    shadow.update(initial=observed.tolist(),controller=name)
                    key=(*case,'observed_shadow')
                    store.append(key,shadow,shadow_trace);expected.append(key)
                    # Only observable physics and commands are compared; the
                    # shadow deliberately has different engine/scenario labels.
                    for field in ('per_vehicle_finish','per_vehicle_variation','completed','collision','aborted','contract_violations'):
                        np.testing.assert_equal(shadow[field],actual[field])
                    if len(shadow_trace)!=len(trace):raise ValueError('Shadow trajectory length differs')
                    for a,b in zip(shadow_trace,trace):
                        np.testing.assert_array_equal(a['command'],b['command'])
                        np.testing.assert_allclose(a['x'],b['x'],atol=1e-8,rtol=0)
                    shadow_runs+=1
                print(json.dumps(dict(shard=shard,verified_pairs=number+1,expected_pairs=len(assigned),
                    actual_sumo_runs=len(expected),shadow_runs=shadow_runs,pid=os.getpid())),flush=True)
        store.seal(expected)
        reloaded=list(observations(out/'observations.sqlite',metadata,expected))
        if len(reloaded)!=len(expected):raise ValueError('Stored execution count differs')
        save(out/'complete.json',dict(pid=os.getpid(),pairs=len(assigned),actual_sumo_runs=len(expected),
            shadow_runs=shadow_runs,review_counts=review.counts,retained_error_cases=review.error_cases,
            plan_sha256=digest(FROZEN/'plan.json'),all_non_timing_reference_fields_exact=True))
        save(out/'sha256.json',manifest(out))
    finally:
        store.close()


def run():
    plan=inputs()
    EXECUTION.mkdir(exist_ok=False)
    processes,handles=[],[]
    try:
        for shard in range(3):
            stdout=(EXECUTION/f'worker_{shard}.stdout.log').open('x',encoding='utf-8')
            stderr=(EXECUTION/f'worker_{shard}.stderr.log').open('x',encoding='utf-8')
            handles.extend((stdout,stderr))
            processes.append(subprocess.Popen([sys.executable,'-X','utf8','tools/preflight_prospective_runtime_v21.py','worker',str(shard)],
                cwd=ROOT,stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)))
        launch=dict(utc=utc(),supervisor_pid=os.getpid(),worker_pids=[p.pid for p in processes],plan_sha256=digest(FROZEN/'plan.json'))
        save(EXECUTION/'launch.json',launch)
        while True:
            codes=[p.poll() for p in processes]
            atomic_bytes(EXECUTION/'state.json',(json.dumps(dict(utc=utc(),worker_pids=launch['worker_pids'],exit_codes=codes))+'\n').encode())
            if all(code is not None for code in codes):break
            time.sleep(10)
        exits=[dict(pid=p.pid,exit_code=p.returncode) for p in processes]
        save(EXECUTION/'worker_exits.json',dict(utc=utc(),workers=exits))
        validate_worker_exits(exits,launch,3)
        receipts=[]
        for shard in range(3):
            directory=OUTPUT/f'shard_{shard}'
            verify_manifest(directory)
            receipt=read(directory/'complete.json')
            if receipt['pairs']!=len(cases(plan,shard)) or receipt['plan_sha256']!=digest(FROZEN/'plan.json'):
                raise ValueError('Complete integration receipt differs')
            receipts.append(receipt)
        total=sum(r['actual_sumo_runs'] for r in receipts)
        if total!=plan['actual_sumo_runs'] or sum(r['shadow_runs'] for r in receipts)!=16:
            raise ValueError('Integration budget is incomplete')
        save(OUTPUT/'audit.json',dict(actual_sumo_runs=total,collection_reference_pairs=64,observed_shadow_runs=16,
            all_reference_pairs_exact=True,all_observed_shadow_commands_and_outcomes_match=True,
            independent_review_counts={k:sum(r['review_counts'][k] for r in receipts) for k in receipts[0]['review_counts']},
            retained_error_cases=[c for r in receipts for c in r['retained_error_cases']],
            new_independent_initials=0,new_risk_certified=False,plan_sha256=digest(FROZEN/'plan.json')))
        save(EXECUTION/'completed.json',dict(utc=utc(),audit_sha256=digest(OUTPUT/'audit.json')))
    except Exception as error:
        save(EXECUTION/'failure.json',dict(utc=utc(),type=type(error).__name__,error=str(error)))
        raise
    finally:
        for handle in handles:handle.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('freeze');sub.add_parser('run')
    sub.add_parser('worker').add_argument('shard',type=int)
    args=parser.parse_args()
    if args.command=='freeze':freeze()
    elif args.command=='worker':worker(args.shard)
    else:run()
