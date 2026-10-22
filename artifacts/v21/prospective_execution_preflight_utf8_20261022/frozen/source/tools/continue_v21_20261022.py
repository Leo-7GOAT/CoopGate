"""Finish the eleven remaining frozen V21 cases after session interruption.

The October 19 observations and October 21 recovery outputs remain immutable.
Original unknown/abnormal exits are never rewritten. A detached watcher records
the new supervisor's actual exit; complete independent reviews precede selection.
"""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np
import torch

from coopgate.matched_shift_v21 import MatchedShiftedMergeEnv
from coopgate.prospective_validation_v21 import profile_names,select_development_family,validate_attempt_history
from prepare_prospective_validation_v21 import ATTEMPTS,RULES,read_rules
from probe_disturbed_gate_v19 import digest,save
from recover_checkpoint_v19 import atomic_bytes
import recover_interrupted_v21 as prior
from select_recovered_v21 import GUARD
from start_final_v19 import require_committed,verify_manifest

ROOT=prior.ROOT
OUTPUT=ROOT/'artifacts/v21/interrupted_recovery_20261022'
FROZEN=OUTPUT/'frozen'
EXECUTION=OUTPUT/'execution'
SOURCES=('tools/continue_v21_20261022.py','tests/test_continue_v21_20261022.py')


def read(path):return prior.read(path)


def split_prefix(expected,present):
    expected=list(map(tuple,expected));present=list(map(tuple,present))
    if present!=expected[:len(present)] or len(set(present))!=len(present):
        raise ValueError('Recovery observations must be the exact frozen prefix')
    return expected[len(present):]


def freeze():
    if OUTPUT.exists():raise FileExistsError('Preserve prior continuation evidence')
    processes=prior.native_snapshot();prior.assert_no_existing_workers(processes)
    old,protocols=prior.inputs(check_original=True)
    prior.require_stage('preflight')
    verify_manifest(prior.RECOVERY/'preflight_audit')
    read_rules();verify_manifest(GUARD)
    require_committed([ROOT/name for name in SOURCES])
    FROZEN.mkdir(parents=True)
    groups=[]
    for study,item in old['studies'].items():
        for shard,entry in item['shards'].items():
            directory=prior.RECOVERY/'suffix'/f'shard_{shard}'/study
            expected=list(map(tuple,entry['missing_cases']))
            existing=[]
            for case in expected:
                row_path=directory/prior.row_name(study,case)
                if row_path.exists():
                    row=read(row_path)
                    receipt=read(directory/prior.row_name(study,case).replace('row_','receipt_',1))
                    if (prior.identity(study,row)!=case or receipt['case']!=list(case)
                            or receipt['stage']!='suffix' or not (directory/prior.trace_name(study,case)).exists()):
                        raise ValueError('Previously submitted case identity differs')
                    existing.append(case)
            missing=split_prefix(expected,existing)
            if {p.name for p in directory.glob('row_*.json')}!={prior.row_name(study,c) for c in existing}:
                raise ValueError('Unexpected previous recovery observation')
            groups.append(dict(study=study,shard=int(shard),existing_cases=existing,missing_cases=missing,
                boundary_case=existing[-1] if missing and existing else None))
    if any(g['missing_cases'] and not g['boundary_case'] for g in groups):
        raise ValueError('This continuation requires an existing boundary observation')
    files={}
    for shard in range(3):
        directory=prior.RECOVERY/'suffix'/f'shard_{shard}'
        archive=FROZEN/f'previous_suffix_{shard}.zip'
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as target:
            for path in sorted(directory.rglob('*')):
                if path.is_file():
                    files[path.relative_to(ROOT).as_posix()]=digest(path)
                    target.write(path,path.relative_to(directory).as_posix())
        with zipfile.ZipFile(archive) as target:
            if target.testzip() is not None:raise ValueError('Prefix archive integrity failure')
    for path in sorted(prior.EXECUTION.iterdir()):
        if path.is_file():
            files[path.relative_to(ROOT).as_posix()]=digest(path)
            copy=FROZEN/'previous_execution'/path.name
            copy.parent.mkdir(exist_ok=True)
            shutil.copy2(path,copy)
    plan=dict(utc=prior.utc(),purpose='Exact continuation after the October 21 process interruption',
        previous_native_snapshot=processes,original_and_previous_supervisor_exit_status='unknown',
        groups=groups,previous_files=files,
        expected_missing_runs=sum(len(g['missing_cases']) for g in groups),
        expected_boundary_replays=sum(bool(g['missing_cases']) for g in groups),
        previous_recovery_plan_sha256=digest(prior.FROZEN/'plan.json'),
        original_rules_sha256=digest(RULES/'plan.json'),previous_admission_rules_sha256=digest(GUARD/'plan.json'),
        earlier_45_replays_audit_sha256=digest(prior.RECOVERY/'preflight_audit/audit.json'),
        policy_or_statistical_rules_changed=False,partial_effectiveness_statistics_read=False,
        selection='Use unchanged whole-family prospective selection after both complete independent reviews',
        source_sha256={name:digest(ROOT/name) for name in SOURCES})
    save(FROZEN/'plan.json',plan)
    for name in SOURCES:
        path=FROZEN/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,path)
    save(FROZEN/'sha256.json',prior.manifest(FROZEN))
    print(json.dumps(dict(missing=plan['expected_missing_runs'],boundary_replays=plan['expected_boundary_replays'],
                          plan_sha256=digest(FROZEN/'plan.json'))),flush=True)


def inputs(check_files=True):
    verify_manifest(FROZEN)
    plan=read(FROZEN/'plan.json')
    require_committed([FROZEN/'plan.json',FROZEN/'sha256.json',*[ROOT/name for name in SOURCES]])
    for name,expected in plan['source_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Continuation source changed')
    old,protocols=prior.inputs(check_original=True)
    if digest(prior.FROZEN/'plan.json')!=plan['previous_recovery_plan_sha256']:
        raise ValueError('Previous recovery plan changed')
    if check_files:
        for name,expected in plan['previous_files'].items():
            if digest(ROOT/name)!=expected:raise ValueError('Previously retained observation changed')
    return plan,old,protocols


def group_dir(group,new=False):
    root=OUTPUT/'work' if new else prior.RECOVERY/'suffix'
    return root/f"shard_{group['shard']}"/group['study']


def worker(group_index):
    plan,_,protocols=inputs()
    group=plan['groups'][group_index]
    study=group['study'];spec=protocols[study]['spec']
    if study!='transfer' or not group['missing_cases']:
        raise ValueError('Only the registered incomplete matched-transfer group may execute')
    torch.set_num_threads(1)
    out=group_dir(group,True);out.mkdir(parents=True,exist_ok=False)
    fs=prior.MODULES[study].factories(protocols[study])
    warm=np.array([[-80.,12.],[-85.,13.],[-120.,11.]])
    for factory in fs.values():factory(warm).select(warm,np.zeros(3))
    with ExitStack() as stack:
        envs={p:stack.enter_context(MatchedShiftedMergeEnv(plant=p)) for p in spec['plants']}
        boundary=group['boundary_case']
        row,trace=prior.execute(study,boundary,spec,fs[boundary[-1]],envs[boundary[4]])
        save(out/'boundary_row.json',row);save(out/'boundary_trace.json',trace)
        prior.verify_execution(study,row,trace,spec)
        previous=group_dir(group)
        prior.compare_replay(read(previous/prior.row_name(study,boundary)),row,
                             read(previous/prior.trace_name(study,boundary)),trace)
        for number,case in enumerate(group['missing_cases']):
            row,trace=prior.execute(study,case,spec,fs[case[-1]],envs[case[4]])
            save(out/prior.row_name(study,case),row);save(out/prior.trace_name(study,case),trace)
            prior.verify_execution(study,row,trace,spec)
            save(out/prior.row_name(study,case).replace('row_','receipt_',1),dict(case=case,verified=True,utc=prior.utc()))
            print(json.dumps(dict(group=group_index,completed=number+1,expected=len(group['missing_cases']),pid=os.getpid())),flush=True)
    save(out/'complete.json',dict(pid=os.getpid(),new_cases=len(group['missing_cases']),boundary_replays=1,
                                  plan_sha256=digest(FROZEN/'plan.json')))
    save(out/'sha256.json',prior.manifest(out))


def require_stage(stage,expected_count):
    launch=read(EXECUTION/f'{stage}_launch.json');exits=read(EXECUTION/f'{stage}_exits.json')
    if launch['plan_sha256']!=digest(FROZEN/'plan.json'):raise ValueError('Different launch plan')
    prior.phase.validate_worker_exits(exits['workers'],launch,expected_count)
    return dict(launch_sha256=digest(EXECUTION/f'{stage}_launch.json'),exits_sha256=digest(EXECUTION/f'{stage}_exits.json'))


def assembled(study,plan,old,protocol):
    rows,sources=[],{}
    for group in [g for g in plan['groups'] if g['study']==study]:
        shard=group['shard'];original=prior.STUDIES[study]/f'shard_{shard}'
        part=read(original/'episodes.json')
        for row in part:
            key=prior.identity(study,row)
            sources[key]=dict(path=(original/prior.trace_name(study,key)).relative_to(ROOT).as_posix(),epoch='original')
        for is_new,key_name in ((False,'existing_cases'),(True,'missing_cases')):
            directory=group_dir(group,is_new)
            if is_new and group[key_name]:
                verify_manifest(directory)
                receipt=read(directory/'complete.json')
                if receipt['new_cases']!=len(group[key_name]) or receipt['plan_sha256']!=digest(FROZEN/'plan.json'):
                    raise ValueError('Incomplete continuation group')
            for case in group[key_name]:
                row=read(directory/prior.row_name(study,case))
                if prior.identity(study,row)!=tuple(case):raise ValueError('Retained case identity changed')
                part.append(row)
                sources[tuple(case)]=dict(path=(directory/prior.trace_name(study,case)).relative_to(ROOT).as_posix(),
                                         epoch='continuation_20261022' if is_new else 'recovery_20261021')
        if [prior.identity(study,r) for r in part]!=prior.MODULES[study].schedule(protocol['spec'],shard):
            raise ValueError('Full original schedule is not complete')
        rows.extend(part)
    for source in sources.values():source['sha256']=digest(ROOT/source['path'])
    if len(rows)!=protocol['spec']['expected_actual_sumo_runs'] or len(sources)!=len(rows):
        raise ValueError('Incomplete or duplicated study')
    return rows,sources


def finalize(study):
    plan,old,protocols=inputs()
    require_stage('resume',sum(bool(g['missing_cases']) for g in plan['groups']))
    protocol=protocols[study];spec=protocol['spec']
    rows,sources=assembled(study,plan,old,protocol)
    out=OUTPUT/'results'/study;out.mkdir(parents=True,exist_ok=False)
    if study=='phase':
        view=ROOT/'runs/v21_review_20261022';view.mkdir(parents=True,exist_ok=False)
        for case,source in sources.items():
            target=view/f'shard_{case[0]%3}'/prior.trace_name(study,case)
            target.parent.mkdir(exist_ok=True);os.link(ROOT/source['path'],target)
        old_study=prior.phase.STUDY
        try:
            prior.phase.STUDY=view
            reviewed=prior.phase.independent_review(rows,protocol)
        finally:prior.phase.STUDY=old_study
        paired=prior.phase.paired_checks(rows,spec)
        results=prior.phase.summarize(rows,spec)
        recounted={name:prior.phase.outcome_counts(rows,spec,name) for name in spec['methods']}
        for name,counts in recounted.items():
            for key,value in counts.items():np.testing.assert_allclose(value,results[name][key],atol=1e-12,rtol=1e-12)
        summary=dict(independent_initials=spec['n_initials'],results=results,paired_results=paired,
                     independently_recounted_outcomes=recounted)
    else:
        reviewed=prior.transfer.independent_review(rows,protocol,lambda r:read(ROOT/sources[prior.identity(study,r)]['path']))
        summary=dict(independent_initials=len(spec['scenarios'])*spec['n_initials_per_scenario'],**prior.transfer.summarize(rows,spec))
    for source in sources.values():
        if digest(ROOT/source['path'])!=source['sha256']:raise ValueError('Trace changed during review')
    rows.sort(key=lambda row:prior.identity(study,row))
    save(out/'episodes.json',rows)
    save(out/'provenance.json',[dict(case=list(case),**source) for case,source in sources.items()])
    save(out/'summary.json',dict(**summary,actual_sumo_runs=len(rows),independent_review_counts=reviewed,
        protocol_sha256=old['studies'][study]['protocol_sha256'],episodes_sha256=digest(out/'episodes.json'),
        continuation_plan_sha256=digest(FROZEN/'plan.json'),original_and_previous_exit_status='unknown',
        risk_certified=False,hard_real_time_certified=False))
    save(out/'sha256.json',prior.manifest(out))
    print(json.dumps(dict(study=study,completed=True,actual_sumo_runs=len(rows))),flush=True)


def stage(name,commands):
    processes,handles=[],[]
    try:
        for i,command in enumerate(commands):
            out=(EXECUTION/f'{name}_{i}.stdout.log').open('x',encoding='utf-8')
            err=(EXECUTION/f'{name}_{i}.stderr.log').open('x',encoding='utf-8');handles.extend((out,err))
            processes.append(subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)))
        save(EXECUTION/f'{name}_launch.json',dict(utc=prior.utc(),supervisor_pid=os.getpid(),
            worker_pids=[p.pid for p in processes],commands=commands,plan_sha256=digest(FROZEN/'plan.json')))
        while True:
            codes=[p.poll() for p in processes]
            atomic_bytes(EXECUTION/'state.json',(json.dumps(dict(utc=prior.utc(),stage=name,
                worker_pids=[p.pid for p in processes],exit_codes=codes))+'\n').encode())
            if all(code is not None for code in codes):break
            time.sleep(10)
        save(EXECUTION/f'{name}_exits.json',dict(utc=prior.utc(),workers=[dict(pid=p.pid,exit_code=p.returncode) for p in processes]))
        if any(p.returncode!=0 for p in processes):raise RuntimeError('Stage failed; retain all outputs')
    finally:
        for handle in handles:handle.close()


def run():
    plan,_,_=inputs()
    base=[sys.executable,'-X','utf8','tools/continue_v21_20261022.py']
    try:
        stage('resume',[base+['worker',str(i)] for i,g in enumerate(plan['groups']) if g['missing_cases']])
        stage('review',[base+['finalize',study] for study in ('phase','transfer')])
        save(EXECUTION/'completed.json',dict(utc=prior.utc(),summary_sha256={s:digest(OUTPUT/'results'/s/'summary.json') for s in ('phase','transfer')}))
    except Exception as error:
        save(EXECUTION/'failure.json',dict(utc=prior.utc(),type=type(error).__name__,error=str(error)));raise


def watch():
    inputs();prior.assert_no_existing_workers(prior.native_snapshot())
    EXECUTION.mkdir(exist_ok=False)
    with (EXECUTION/'supervisor.stdout.log').open('x') as out,(EXECUTION/'supervisor.stderr.log').open('x') as err:
        command=[sys.executable,'-X','utf8','tools/continue_v21_20261022.py','run']
        process=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        save(EXECUTION/'watcher_launch.json',dict(utc=prior.utc(),watcher_pid=os.getpid(),supervisor_pid=process.pid,
                                               command=command,plan_sha256=digest(FROZEN/'plan.json')))
        code=process.wait()
        save(EXECUTION/'supervisor_exit.json',dict(utc=prior.utc(),supervisor_pid=process.pid,exit_code=code,
                                                 plan_sha256=digest(FROZEN/'plan.json')))
    if code:raise SystemExit(code)


def select():
    plan,old,protocols=inputs()
    receipt=read(EXECUTION/'supervisor_exit.json');launch=read(EXECUTION/'watcher_launch.json')
    if (receipt['exit_code']!=0 or receipt['supervisor_pid']!=launch['supervisor_pid']
            or receipt['plan_sha256']!=digest(FROZEN/'plan.json') or launch['plan_sha256']!=receipt['plan_sha256']):
        raise ValueError('The new supervised run has not completed successfully')
    evidence=dict(resume=require_stage('resume',sum(bool(g['missing_cases']) for g in plan['groups'])),review=require_stage('review',2),
                  supervisor_exit_sha256=digest(EXECUTION/'supervisor_exit.json'))
    prior.assert_no_existing_workers(prior.native_snapshot())
    rules=read_rules();verify_manifest(GUARD)
    if digest(RULES/'plan.json')!=plan['original_rules_sha256'] or digest(GUARD/'plan.json')!=plan['previous_admission_rules_sha256']:
        raise ValueError('Prospective selection rules changed')
    finished=read(EXECUTION/'completed.json')
    summaries={}
    for study,protocol in protocols.items():
        directory=OUTPUT/'results'/study;verify_manifest(directory)
        if digest(directory/'summary.json')!=finished['summary_sha256'][study]:raise ValueError('Different completed summary')
        summary=read(directory/'summary.json')
        if (summary['actual_sumo_runs']!=protocol['spec']['expected_actual_sumo_runs']
                or summary['independent_review_counts']['physical_traces']!=summary['actual_sumo_runs']
                or summary['protocol_sha256']!=old['studies'][study]['protocol_sha256']):
            raise ValueError('Complete independent review identity differs')
        rows,sources=assembled(study,plan,old,protocol);rows.sort(key=lambda row:prior.identity(study,row))
        np.testing.assert_equal(rows,read(directory/'episodes.json'))
        np.testing.assert_equal({tuple(r['case']):{k:v for k,v in r.items() if k!='case'} for r in read(directory/'provenance.json')},sources)
        summaries[study]=summary
    if protocols['phase']['models']!=rules['models'] or summaries['phase']['protocol_sha256']!=rules['development_protocol_sha256']:
        raise ValueError('Different development family')
    output=ATTEMPTS/'attempt_1'
    if output.exists():raise FileExistsError('Preserve prior selection')
    allocation=validate_attempt_history([],1)
    choice=select_development_family(summaries['phase']['results'],summaries['phase']['independent_initials'])
    output.mkdir(parents=True)
    save(output/'selection.json',dict(**choice,**allocation,continuation_completion=evidence,
        continuation_plan_sha256=digest(FROZEN/'plan.json'),rules_sha256=digest(RULES/'plan.json'),
        common_profiles=list(profile_names(choice['selected_family'])) if choice['selected_family'] else [],
        development_summary_sha256=digest(OUTPUT/'results/phase/summary.json'),
        calibration_execution_protocol_still_required=True,new_calibration_data_generated=False))
    print(json.dumps(dict(selected_family=choice['selected_family'],risk_certified=False)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    for name in ('freeze','watch','run','select'):sub.add_parser(name)
    sub.add_parser('worker').add_argument('group',type=int)
    sub.add_parser('finalize').add_argument('study',choices=['phase','transfer'])
    args=parser.parse_args()
    if args.command=='worker':worker(args.group)
    elif args.command=='finalize':finalize(args.study)
    else:globals()[args.command]()
