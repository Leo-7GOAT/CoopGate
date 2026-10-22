"""Complete common-condition MAMPS/MPS adapters, without reading live calibration."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.disturbance_v9 import DisturbedMergeEnv
from coopgate.episode_store_v21 import EpisodeStore, encode, observations
from coopgate.matched_shift_v21 import MatchedShiftedMergeEnv, audit_matched_trace
from coopgate.physical_switch_v27 import PhysicalSwitchGate
from audit_buffered_scope_v25 import TimedGate, check_calls, validate_completion
from execute_prospective_v23 import (acknowledge_child, check_acknowledgment, exclusive_json,
                                    runtime_identity, sources_committed, utc)
from experiment_ego_v13 import run_episode
from present_completed_v21 import load_complete
from profile_phase_runtime_v24 import committed, digest
from recover_checkpoint_v19 import atomic_bytes
import recover_interrupted_v21 as original
from verify_development_v14 import trace_audit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/v27/physical_switch_comparison'
SCOPE = ROOT/'artifacts/v25/buffered_complete_scope/plan.json'
OLD = ROOT/'artifacts/v21/interrupted_recovery_20261022/results'
RULES = ('mamps_shared','mps_shared')
SCRIPT = 'tools/audit_physical_switch_v27.py'
NEW_SOURCES = ('src/coopgate/physical_switch_v27.py','tests/test_physical_switch_v27.py',
               SCRIPT,'tests/test_physical_switch_audit_v27.py')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def groups_for(plan, stage):
    if stage not in ('preflight','full'):
        raise ValueError('Unknown stage')
    return [g for g in plan['groups'] if stage=='full' or g['local_index']==0]


def schedule(group):
    result=[]
    for i, entry in enumerate(group['cases']):
        for rule in (RULES if i%2==0 else RULES[::-1]):
            result.append((*entry['case'],rule))
    return result


def validate_scope(base, groups):
    if groups != base['groups'] or len(groups)!=288 or sum(len(g['cases']) for g in groups)!=3456:
        raise ValueError('Every original development/transfer case is required')
    selected=groups_for(dict(groups=groups),'preflight')
    if len(selected)!=6 or sum(len(schedule(g)) for g in selected)!=144:
        raise ValueError('Complete six-initial integration required')


def freeze():
    load_complete()
    base=read(SCOPE)
    validate_scope(base,base['groups'])
    sources=dict(base['source_sha256'])
    for name,expected in sources.items():
        if digest(ROOT/name)!=expected:
            raise ValueError('Frozen source changed: '+name)
    sources.update({name:digest(ROOT/name) for name in NEW_SOURCES})
    sources_committed([ROOT/name for name in sources])
    plan=dict(purpose=__doc__,scope_plan_sha256=digest(SCOPE),source_sha256=sources,
        inputs_sha256=base['inputs_sha256'],groups=base['groups'],protocols=base['protocols'],
        models=base['models'],runtime=runtime_identity(),rules=list(RULES),
        stages=dict(preflight=dict(blocks=6,actual_sumo_runs=144),full=dict(blocks=288,actual_sumo_runs=6912)),
        old_independent_initials=dict(phase=128,transfer=160),new_independent_initials=0,
        prior_source='https://arxiv.org/html/1910.12639v2',
        adaptation='MAMPS Algorithm 1 monotone masks and all-or-none MPS, with the same strong baseline and existing bounded-error stopping certificate; not the original deterministic robot benchmark or its theorem',
        comparison='All previous completed phase/transfer controllers retained; primary contrasts compare both physical adapters with phase05 on identical model/initial/noise/plant cases',
        primary_risk_certificate_unchanged=True,active_calibration_outcomes_read=False,
        adoption='Any applied command differs from the same-state strong baseline by >1e-9 on at least one not-yet-arrived vehicle; episode adoption is any such step',
        harm='Any car cost or arrival exceeds the matched full-baseline outcome by >1e-9, or execution failure; baseline failures retained and reported separately',
        statistical_scope='Old development/transfer initials only. Phase actual modes use the original mode_selection_stream; four-mode bundle counts separate. Transfer 32 paired initials per scene/plant. Pointwise paired bootstrap only, no independent confirmation or new risk certificate.',
        bootstrap_samples=5000,bootstrap_stream=27113,
        runtime_scope='One extra worker alongside existing studies. New full callback timing includes baseline and all physical checks; historical wall times are not a paired speed comparison.',
        failures='Persist raw rows/traces before review, retain aborts and bound violations; no silent resume, overwrite or outcome-selected rerun',
        workers=1)
    OUT.mkdir(parents=True,exist_ok=False)
    exclusive_json(OUT/'plan.json',plan)
    print(json.dumps(dict(plan_sha256=digest(OUT/'plan.json'),stages=plan['stages'])),flush=True)


def inputs(stage):
    committed(OUT/'plan.json')
    plan=read(OUT/'plan.json')
    if digest(SCOPE)!=plan['scope_plan_sha256'] or runtime_identity()!=plan['runtime']:
        raise ValueError('Frozen scope/runtime differs')
    validate_scope(read(SCOPE),plan['groups'])
    for name,expected in {**plan['source_sha256'],**plan['inputs_sha256']}.items():
        if digest(ROOT/name)!=expected:
            raise ValueError('Frozen input/source differs: '+name)
    sources_committed([ROOT/name for name in plan['source_sha256']])
    for meta in plan['models'].values():
        if digest(ROOT/meta['path'])!=meta['sha256']:
            raise ValueError('Frozen actor changed')
    if stage=='full':
        proof=OUT/'preflight/analysis/verification.json'
        committed(proof)
        checked=read(proof)
        if checked['plan_sha256']!=digest(OUT/'plan.json') or checked['actual_sumo_runs']!=144:
            raise ValueError('Complete committed preflight verification required')
        if digest(OUT/'preflight/execution/exit.json')!=checked['exit_receipt_sha256']:
            raise ValueError('Preflight execution evidence changed')
    return plan


def metadata(stage,group):
    return dict(plan_sha256=digest(OUT/'plan.json'),stage=stage,
                original_block_index=group['block_index'],study=group['study'])


def build_row(row,trace,initial,old_row,case,rule):
    row.update(initial=initial.tolist(),controller=rule+'_'+case[-1].rsplit('_',1)[1],
               model=case[-1].rsplit('_',1)[1],physical_switch_rule=rule,
               initial_index=old_row['initial_index'],control_steps=len(trace),
               learned_action_steps=sum(t['actual_learning_used'] for t in trace))
    if 'global_initial_index' in old_row:
        row.update(global_initial_index=old_row['global_initial_index'],plant_name=old_row['plant_name'])
    costs=np.asarray(row['per_vehicle_finish'])+.05*np.asarray(row['per_vehicle_variation'])
    row.update(per_vehicle_cost=costs.tolist(),team_cost=float(costs.mean()))
    return row


def verify_record(record,trace,model,study,old_row):
    row=record['row']
    check_calls(row,trace,record['calls'])
    for key in ('seed','scenario','disturbance','radius','initial'):
        np.testing.assert_equal(row[key],old_row[key])
    if study=='transfer':
        for key in ('plant','external_innovation_radius','physical_error_radius','innovation_tape_sha256'):
            np.testing.assert_equal(row[key],old_row[key])
        audit_matched_trace(trace,row)
    else:
        trace_audit(trace,row)
    gate=PhysicalSwitchGate(model,row['physical_switch_rule'])
    state,previous=np.array(row['initial']),np.zeros(3)
    for step,item in enumerate(trace):
        command,info=gate.select(state,previous)
        np.testing.assert_array_equal(command,item['command'])
        expected={k:v for k,v in item.items() if k not in ('t','x','command','disturbance','innovation','actuator_states','realized_error_in_contract','gate_ms')}
        # Transfer environments add diagnostic fields; compare every controller
        # field, then separately reconstruct all plant fields above.
        for key,value in info.items():
            if key!='gate_ms':
                np.testing.assert_equal(value,expected[key])
        state,previous=np.array(item['x']),command
    if row['aborted']:
        try:
            gate.select(state,previous)
        except Exception as exc:
            if str(exc)!=row['abort_reason']:
                raise ValueError('Terminal failure did not reproduce') from exc
        else:
            raise ValueError('Recorded terminal failure did not reproduce')
    if row['learned_action_steps']!=sum(t['actual_learning_used'] for t in trace):
        raise ValueError('Actual adoption accounting differs')
    return dict(physical_traces=1,reconstructed_commands=len(trace),terminal_failures=int(row['aborted']),
                initial_and_external_noise_matched=True,actual_learning_steps=row['learned_action_steps'])


def worker(stage):
    execution=OUT/stage/'execution'
    startup=execution/'worker.native.json'
    atomic_bytes(startup,encode(dict(native_pid=os.getpid(),parent_pid=os.getppid(),utc=utc())))
    limit=time.monotonic()+180
    ack=execution/'worker.acknowledged.json'
    while not ack.exists():
        if time.monotonic()>limit:
            raise TimeoutError('No native worker acknowledgment')
        time.sleep(.2)
    proof=read(ack)
    if (proof['native_pid']!=os.getpid() or proof['parent_pid']!=os.getppid()
            or proof['native_start_sha256']!=digest(startup)):
        raise ValueError('Native identity acknowledgment differs')
    plan=inputs(stage)
    torch.set_num_threads(1)
    models={s:load_coupled(ROOT/m['path'])[0] for s,m in plan['models'].items()}
    old_rows={s:{original.identity(s,r):r for r in read(OLD/s/'episodes.json')} for s in plan['protocols']}
    warm=np.array([[-80.,12.],[-85.,13.],[-120.,11.]])
    for model in models.values():
        for rule in RULES:
            PhysicalSwitchGate(model,rule).select(warm,np.zeros(3))
    blocks=OUT/stage/'blocks'
    blocks.mkdir(exist_ok=False)
    completed=0
    with ExitStack() as stack:
        additive=stack.enter_context(DisturbedMergeEnv())
        shifted={p:stack.enter_context(MatchedShiftedMergeEnv(plant=p)) for p in plan['protocols']['transfer']['spec']['plants']}
        for number,group in enumerate(groups_for(plan,stage)):
            directory=blocks/f"{group['block_index']:05d}"
            directory.mkdir()
            store=EpisodeStore(directory/'observations.sqlite',metadata(stage,group))
            reviews=[]
            try:
                for case_with_rule in schedule(group):
                    case,rule=tuple(case_with_rule[:-1]),case_with_rule[-1]
                    study=group['study']
                    old_row=old_rows[study][case]
                    model=models[case[-1].rsplit('_',1)[1]]
                    wrapped=[]
                    def factory(state):
                        controller=TimedGate(PhysicalSwitchGate(model,rule))
                        wrapped.append(controller)
                        return controller
                    env=additive if study=='phase' else shifted[case[4]]
                    started=time.perf_counter()
                    row,trace,initial=run_episode(env,old_row['seed'],old_row['scenario'],old_row['disturbance'],factory,True)
                    row=build_row(row,trace,initial,old_row,case,rule)
                    record=dict(row=row,calls=wrapped[0].calls,episode_wall_seconds=time.perf_counter()-started)
                    store.append(case_with_rule,record,trace)
                    reviews.append(verify_record(record,trace,model,study,old_row))
                    completed+=1
                store.seal(schedule(group))
            finally:
                store.close()
            exclusive_json(directory/'complete.json',dict(**metadata(stage,group),actual_sumo_runs=len(reviews),
                reviews=reviews,database_sha256=digest(directory/'observations.sqlite'),
                seal_sha256=digest(directory/'observations.sqlite.seal.json')))
            atomic_bytes(execution/'progress.json',encode(dict(utc=utc(),native_pid=os.getpid(),
                completed_blocks=number+1,expected_blocks=plan['stages'][stage]['blocks'],
                completed_runs=completed,expected_runs=plan['stages'][stage]['actual_sumo_runs'])))
            print(json.dumps(dict(stage=stage,blocks=number+1,completed_runs=completed)),flush=True)
    assert completed==plan['stages'][stage]['actual_sumo_runs']
    exclusive_json(execution/'collection_complete.json',dict(plan_sha256=digest(OUT/'plan.json'),native_pid=os.getpid(),
        actual_sumo_runs=completed,blocks=plan['stages'][stage]['blocks'],utc=utc()))


def watch(stage):
    inputs(stage)
    execution=OUT/stage/'execution'
    execution.mkdir(parents=True,exist_ok=False)
    with (execution/'worker.stdout.log').open('x') as stdout,(execution/'worker.stderr.log').open('x') as stderr:
        process=subprocess.Popen([sys.executable,'-X','utf8',SCRIPT,'worker',stage],cwd=ROOT,stdout=stdout,stderr=stderr,
                                 creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        exclusive_json(execution/'spawn.json',dict(launcher_pid=process.pid,watcher_pid=os.getpid(),utc=utc()))
        try:
            native=acknowledge_child(process,execution,'worker')
            exclusive_json(execution/'launch.json',dict(**native,plan_sha256=digest(OUT/'plan.json'),utc=utc()))
        except Exception as exc:
            exclusive_json(execution/'observation_error.json',dict(type=type(exc).__name__,message=str(exc),utc=utc()))
            code=process.wait()
            exclusive_json(execution/'exit.json',dict(launcher_pid=process.pid,exit_code=code,plan_sha256=digest(OUT/'plan.json'),utc=utc()))
            raise
        code=process.wait()
        exclusive_json(execution/'exit.json',dict(**native,exit_code=code,plan_sha256=digest(OUT/'plan.json'),utc=utc()))
    if code:
        raise SystemExit(code)


def verify_stage(stage):
    plan=inputs(stage)
    execution=OUT/stage/'execution'
    receipt,launch=read(execution/'exit.json'),read(execution/'launch.json')
    validate_completion(plan['stages'][stage],digest(OUT/'plan.json'),receipt,launch,read(execution/'collection_complete.json'))
    check_acknowledgment(execution,'worker',launch['launcher_pid'],launch['native_pid'])
    runs=commands=failures=0
    for group in groups_for(plan,stage):
        directory=OUT/stage/'blocks'/f"{group['block_index']:05d}"
        done=read(directory/'complete.json')
        if (done['database_sha256']!=digest(directory/'observations.sqlite')
                or done['seal_sha256']!=digest(directory/'observations.sqlite.seal.json')
                or done['actual_sumo_runs']!=len(schedule(group))):
            raise ValueError('Completed block or seal differs')
        rows=list(observations(directory/'observations.sqlite',metadata(stage,group),schedule(group)))
        if len(rows)!=len(done['reviews']):
            raise ValueError('Missing per-case physical review')
        for (_,record,trace),review in zip(rows,done['reviews'],strict=True):
            check_calls(record['row'],trace,record['calls'])
            if review['physical_traces']!=1 or review['reconstructed_commands']!=len(trace):
                raise ValueError('Review scope differs')
            runs+=1;commands+=len(trace);failures+=review['terminal_failures']
    if runs!=plan['stages'][stage]['actual_sumo_runs']:
        raise ValueError('Incomplete stage')
    output=OUT/stage/'analysis'
    output.mkdir(exist_ok=False)
    verification=dict(plan_sha256=digest(OUT/'plan.json'),stage=stage,actual_sumo_runs=runs,
        independently_reconstructed_commands=commands,retained_terminal_failures=failures,
        exit_receipt_sha256=digest(execution/'exit.json'),new_independent_initials=0,
        new_risk_certificate=False,active_calibration_outcomes_read=False)
    exclusive_json(output/'verification.json',verification)
    print(json.dumps(verification),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('freeze','worker','watch','verify'))
    parser.add_argument('stage',nargs='?',choices=('preflight','full'))
    args=parser.parse_args()
    if args.action=='freeze':
        freeze()
    elif args.stage is None:
        parser.error('A stage is required')
    else:
        {'worker':worker,'watch':watch,'verify':verify_stage}[args.action](args.stage)
