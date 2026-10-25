"""All five completed V36 bank regressions: true suffixes and 16 SUMO branches.

Post-hoc mechanism study, not fresh efficacy, tuning, or a risk certificate.
The original V31 telescoping identity and V34/V35 execution remain unchanged.
"""
import argparse
import json
import os
from pathlib import Path
from types import FunctionType

import numpy as np
import torch

from coopgate.actuator_recovery_v34 import class_bounds
from coopgate.episode_store_v21 import EpisodeStore,observations
from coopgate.recovery_forecast_v35 import forecast
from coopgate.recovery_intervention_diagnostic import ROLES,RecordedRecoveryBranch,compare_prediction,identity,pair,predictions
from coopgate.temporal_portfolio_v20 import temporal_stress_bank
import execute_fair_failure_v44 as frozen
import recover_block_grids_v38 as recovered
import report_recovery_grid_v36 as completed
import run_joint_risk_v42 as runtime

ROOT=runtime.design.ROOT
io=frozen.io
OUT=ROOT/'artifacts/diagnostics/v36_all_bank_regressions'
PLAN=OUT/'plan.json'
REPORT_SHA='1e13918534046bc71b3a39a0ee340c5852357778930dc5cbc8b96cf4eb175162'
SOURCES=('src/coopgate/recovery_intervention_diagnostic.py','tools/diagnose_recovery_regressions.py',
    'tests/test_recovery_intervention_diagnostic.py','docs/v36_regression_mechanism_protocol.md',
    'tools/run_joint_risk_v42.py','tools/recover_block_grids_v38.py','tools/report_recovery_grid_v36.py')


def source_cases():
    description=completed.derive()
    assert description['source_report_sha256']==REPORT_SHA and len(description['bank_harm_cases'])==5
    report=io.read(completed.OUT/'analysis/report.json')
    proof=io.read(completed.OUT/'completion_verification.json')
    original=io.read(recovered.v36.OUT/'plan.json');recovery=io.read(completed.OUT/'plan.json')
    selected=sorted((r for r in report['cases'] if r['row']['rule']=='bank_performance' and r['relative_harm']),key=lambda r:r['case'])
    lookup={tuple(key):group for group in original['groups'] for key in group['cases']}
    files={p.relative_to(ROOT).as_posix():io.digest(p) for p in [completed.OUT/'analysis/report.json',
        completed.OUT/'analysis/sha256.json',completed.OUT/'completion_verification.json',
        completed.OUT/'plan.json',recovered.v36.OUT/'plan.json']}
    records=[]
    for selected_case in selected:
        key=selected_case['case'];group=lookup[tuple(key)]
        folder,metadata,_=recovered.complete_directory('v36',recovery,group)
        expected=proof['block_files_sha256'][f"{group['block_index']:05d}"]
        for name,digest in expected.items():
            assert io.digest(folder/name)==digest;files[(folder/name).relative_to(ROOT).as_posix()]=digest
        matches=[(body,trace) for case,body,trace in observations(folder/'observations.sqlite',metadata,group['cases']) if case==tuple(key)]
        assert len(matches)==1
        body,trace=matches[0];assert body['row']==selected_case['row'] and body['calls']==selected_case['calls']
        steps=[k for k,t in enumerate(trace) if t['actual_learning_used']]
        assert len(steps)==body['row']['learned_action_steps']==selected_case['review']['adopted_steps']
        records.append(dict(case=key,source=body['row'],baseline=selected_case['baseline'],trace=trace,steps=steps))
    assert len(records)==5 and sum(len(r['steps']) for r in records)==8
    return records,files,original


def freeze():
    if OUT.exists():raise FileExistsError('Preserve first post-hoc mechanism plan')
    records,files,original=source_cases();_,binding=frozen.bound_inputs()
    hashes={**binding['source_sha256'],**original['source_sha256'],**{p:io.digest(ROOT/p) for p in SOURCES}}
    for name,expected in hashes.items():io.committed(ROOT/name);assert io.digest(ROOT/name)==expected
    OUT.mkdir(parents=True,exist_ok=False)
    metadata=dict(scope='all-five-completed-bank-regressions',source_report_sha256=REPORT_SHA)
    store=EpisodeStore(OUT/'inputs.sqlite',metadata)
    try:
        for record in records:store.append(record['case'],{k:v for k,v in record.items() if k!='trace'},record['trace'])
        store.seal([r['case'] for r in records])
    finally:store.close()
    jobs=[dict(case_index=i,step=k) for i,r in enumerate(records) for k in r['steps']]
    io.exclusive_json(PLAN,dict(source_sha256=hashes,source_files_sha256=files,runtime=io.runtime_identity(),
        input_metadata=metadata,input_cases=[r['case'] for r in records],
        input_database_sha256=io.digest(OUT/'inputs.sqlite'),input_seal_sha256=io.digest(OUT/'inputs.sqlite.seal.json'),
        jobs=jobs,roles=list(ROLES),expected_sumo_runs=16,selected_source_cases=5,
        distinct_source_initials=len({tuple(r['case'][:4]) for r in records}),new_independent_initials=0,
        posthoc_selection_on_completed_harms=True,source_scope_runs=1920,
        controller_changed=False,new_risk_certificate=False,model_selection=False,
        diagnostic='Every recorded adoption in every retained bank harm; true-plant realized-path suffixes, true-plant 33-path bank, and two physical prefix-preserving interventions per adoption',
        oracle_scope='True parameters/actuator memory/future tape used only offline; native controller sees state, commands and public class',
        incomplete_policy='Retain failed/capped branches; suppress complete-performance identity if any local branch incomplete',
        native_prefix_state_injection=False,active_partial_efficacy_read=False,utc=io.utc()))
    print(json.dumps(dict(plan_sha256=io.digest(PLAN),adoptions=8,sumo_replays=16)),flush=True)


def inputs():
    io.committed(PLAN);plan=io.read(PLAN)
    assert plan['runtime']==io.runtime_identity() and plan['expected_sumo_runs']==16 and len(plan['jobs'])==8
    for name,expected in {**plan['source_sha256'],**plan['source_files_sha256']}.items():assert io.digest(ROOT/name)==expected
    assert io.digest(OUT/'inputs.sqlite')==plan['input_database_sha256']
    assert io.digest(OUT/'inputs.sqlite.seal.json')==plan['input_seal_sha256']
    records=[]
    for key,body,trace in observations(OUT/'inputs.sqlite',plan['input_metadata'],plan['input_cases']):
        assert key==tuple(body['case']);records.append(dict(**body,trace=trace))
    assert [dict(case_index=i,step=k) for i,r in enumerate(records) for k in r['steps']]==plan['jobs']
    return plan,records


def coupled_functions(source,trace,step,role):
    def factory(unused_model,profile,unused_rule):
        assert profile==source['plant_name'];return RecordedRecoveryBranch(source,trace,step,role)
    run=FunctionType(runtime.execute_episode.__code__,{**runtime.execute_episode.__globals__,'MemoizedRecoveryGate':factory},
        argdefs=runtime.execute_episode.__defaults__)
    audit=FunctionType(runtime.audit_record.__code__,{**runtime.audit_record.__globals__,'RecoveryJointGate':factory},
        argdefs=runtime.audit_record.__defaults__)
    return run,audit


def physical_case(source,index,step,role):
    tape=runtime.disturbance_tape(source['seed'],source['disturbance'],.1,250)
    assert runtime.tape_fingerprint(tape)==source['innovation_tape_sha256']
    return dict(cluster_id=f'v36-regression-{index:02d}-{step:02d}-{role}',seed=source['seed'],
        scenario=source['scenario'],plant=source['plant_name'],disturbance=source['disturbance'],
        initial_sha256=runtime.design.shared.array_sha256(source['initial'],decimals=8),
        dynamics_sha256=runtime.design.shared.canonical_sha256(source['plant']),
        disturbance_sha256=runtime.design.shared.array_sha256(tape))


def offline(record,step,tape):
    source,trace=record['source'],record['trace'];actual=predictions(source,trace,step,tape,record_trace=True)
    # Same realized path with the ONLINE reference coefficients/state isolates
    # a diagnostic model contrast; it is never used as a deployable forecast.
    reference=predictions(source,trace,step,tape,plant=trace[step]['prediction_parameters'],
        memory=trace[step]['reconstructed_actuator'])
    _,bank=temporal_stress_bank('sampled');bank=np.ascontiguousarray(bank*(.1/.5))
    bank_pairs=[pair(predictions(source,trace,step,path)) for path in bank]
    complete=all(p['both_complete'] for p in bank_pairs)
    bank_max=np.max([p['difference'] for p in bank_pairs],axis=0) if complete else None
    phase_max=np.max([p['phase_difference'] for p in bank_pairs],axis=0) if complete else None
    return dict(actual_branches=actual,actual_pair=pair(actual),reference_parameter_realized_path_pair=pair(reference),
        true_parameter_bank_pairs=bank_pairs,true_parameter_bank_all_complete=complete,
        true_parameter_bank_maximum=bank_max.tolist() if complete else None,
        true_parameter_bank_phase_maximum=phase_max.tolist() if complete else None,
        true_parameter_bank_pass=bool(complete and np.all(bank_max<=1e-10) and np.all(phase_max<=.05+1e-10)),
        original_online_bank_maximum=np.max(trace[step]['selected_bank_differences'],axis=0).tolist())


def run():
    plan,records=inputs();torch.set_num_threads(1)
    (OUT/'execution').mkdir(exist_ok=False)
    io.exclusive_json(OUT/'execution/native_start.json',dict(native_pid=os.getpid(),parent_pid=os.getppid(),utc=io.utc(),plan_sha256=io.digest(PLAN)))
    (OUT/'jobs').mkdir();local={i:[] for i in range(5)};reviews=[]
    for number,job in enumerate(plan['jobs']):
        index,step=job['case_index'],job['step'];record=records[index];source,trace=record['source'],record['trace']
        tape=runtime.disturbance_tape(source['seed'],source['disturbance'],.1,250)
        folder=OUT/'jobs'/f'{number:02d}';folder.mkdir();diagnostic=offline(record,step,tape)
        io.exclusive_json(folder/'offline.json',diagnostic);actual={}
        for role in (ROLES if number%2==0 else ROLES[::-1]):
            case=physical_case(source,index,step,role);execute,audit=coupled_functions(source,trace,step,role)
            with runtime.MatchedShiftedMergeEnv(plant=source['plant_name']) as env:body=execute(case,'baseline',None,env)
            body.update(diagnostic_role=role,source_case=record['case'],intervention_step=step,
                wire_method_is_baseline_suffix_scaffold=True)
            io.exclusive_json(folder/(role+'.json'),body)
            replay=audit(case,body,None);comparison=compare_prediction(diagnostic['actual_branches'][role],body,source,trace,step)
            review=dict(job=job,role=role,physical_replay=replay,forecast_comparison=comparison)
            io.exclusive_json(folder/(role+'_review.json'),review);reviews.append(review);actual[role]=body
        both=all(actual[r]['row']['completed'] for r in ROLES)
        difference=(np.column_stack((actual[ROLES[1]]['row']['per_vehicle_cost'],actual[ROLES[1]]['row']['per_vehicle_finish']))
            -np.column_stack((actual[ROLES[0]]['row']['per_vehicle_cost'],actual[ROLES[0]]['row']['per_vehicle_finish']))) if both else None
        if both:np.testing.assert_allclose(difference,diagnostic['actual_pair']['difference'],atol=1e-9,rtol=0)
        local[index].append(dict(step=step,pair=dict(both_complete=both,difference=difference.tolist() if both else None)))
        print(json.dumps(dict(adoptions_replayed=number+1,expected_adoptions=8,actual_sumo_runs=len(reviews))),flush=True)
    identities=[]
    for index,record in enumerate(records):
        source=record['source'];tape=runtime.disturbance_tape(source['seed'],source['disturbance'],.1,250)
        initial=forecast(np.asarray(source['initial']),np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3),0,tape,
            source['plant'],class_bounds(source['plant_name']))
        expected=np.column_stack((record['baseline']['per_vehicle_cost'],record['baseline']['per_vehicle_finish']))
        assert initial['complete'];np.testing.assert_allclose(initial['values'],expected,atol=1e-9,rtol=0)
        identities.append(dict(case=record['case'],**identity(source,record['baseline'],local[index])))
    assert len(reviews)==16 and inputs()[0]==plan
    io.exclusive_json(OUT/'report.json',dict(cases=identities,physical_replays=16,adoptions=8,
        all_forecasts_match_physical_replays=True,new_independent_initials=0,posthoc_descriptive_only=True,
        identity_is_existing_v31_accounting=True,new_risk_certificate=False))
    io.exclusive_json(OUT/'complete.json',dict(plan_sha256=io.digest(PLAN),actual_sumo_runs=16,
        files_sha256={p.relative_to(OUT).as_posix():io.digest(p) for p in OUT.rglob('*') if p.is_file()}))
    print(json.dumps(dict(physical_replays=16,identity_assessable=sum(c['assessable'] for c in identities),
        all_assessable_identities_match=all(c['identity_matches'] for c in identities if c['assessable']))),flush=True)


def verify():
    plan,records=inputs();done=io.read(OUT/'complete.json');exit_proof=io.read(OUT/'observed_execution_exit.json')
    assert done['plan_sha256']==io.digest(PLAN) and done['actual_sumo_runs']==16
    assert exit_proof['exit_code']==0 and exit_proof['complete_sha256']==io.digest(OUT/'complete.json')
    assert exit_proof['native_pid']==io.read(OUT/'execution/native_start.json')['native_pid']
    for name,expected in done['files_sha256'].items():assert io.digest(OUT/name)==expected
    torch.set_num_threads(1);local={i:[] for i in range(5)};count=0
    for number,job in enumerate(plan['jobs']):
        index,step=job['case_index'],job['step'];record=records[index];source,trace=record['source'],record['trace']
        tape=runtime.disturbance_tape(source['seed'],source['disturbance'],.1,250)
        folder=OUT/'jobs'/f'{number:02d}';diagnostic=offline(record,step,tape)
        assert diagnostic==io.read(folder/'offline.json')
        actual={}
        for role in ROLES:
            body=io.read(folder/(role+'.json'));case=physical_case(source,index,step,role)
            _,audit=coupled_functions(source,trace,step,role)
            review=dict(job=job,role=role,physical_replay=audit(case,body,None),
                forecast_comparison=compare_prediction(diagnostic['actual_branches'][role],body,source,trace,step))
            assert review==io.read(folder/(role+'_review.json'));actual[role]=body;count+=1
        both=all(actual[r]['row']['completed'] for r in ROLES)
        difference=(np.column_stack((actual[ROLES[1]]['row']['per_vehicle_cost'],actual[ROLES[1]]['row']['per_vehicle_finish']))
            -np.column_stack((actual[ROLES[0]]['row']['per_vehicle_cost'],actual[ROLES[0]]['row']['per_vehicle_finish']))) if both else None
        local[index].append(dict(step=step,pair=dict(both_complete=both,difference=difference.tolist() if both else None)))
    report=io.read(OUT/'report.json')
    expected=[dict(case=r['case'],**identity(r['source'],r['baseline'],local[i])) for i,r in enumerate(records)]
    assert report['cases']==expected and count==report['physical_replays']==16
    io.exclusive_json(OUT/'verification.json',dict(complete_sha256=io.digest(OUT/'complete.json'),observed_exit=exit_proof,
        physical_replays_verified=16,offline_jobs_recomputed=8,all_source_and_file_hashes_verified=True,
        all_forecasts_match_physical_replays=True,all_reported_identities_recomputed=True,
        new_independent_initials=0,new_risk_certificate=False))
    print(json.dumps(dict(physical_replays_verified=16,offline_jobs_recomputed=8)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('freeze','run','verify'));args=parser.parse_args()
    globals()[args.command]()
