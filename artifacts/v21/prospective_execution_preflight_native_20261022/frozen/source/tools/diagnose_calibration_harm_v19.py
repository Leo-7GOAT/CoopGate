"""Replay every strict-gate calibration harm, then inspect hindsight potentials.

Outcome-dependent development diagnosis only. Realized future noise is used
after each closed-loop SUMO replay, never as an online controller input.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from coopgate.disturbance_v9 import DisturbedMergeEnv, disturbance_tape
from coopgate.disturbed_advantage_v19 import disturbed_suffix
from coopgate.fast_v8 import parameters
from coopgate.merge_v4 import MergeConfig
from compare_established_v19 import complete_study
from experiment_ego_v13 import run_episode
from probe_disturbed_gate_v19 import digest, save
from resume_frozen_risk_v19 import build_record, check_episode, compare_boundary
from start_final_v19 import require_committed, verify_manifest
from study_gate_v19 import factories, read_protocol

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'artifacts/v19/risk_validation'
PLAN = ROOT / 'artifacts/v19/calibration_harm_replay_plan'
OUTPUT = ROOT / 'artifacts/v19/calibration_harm_replay'


def identity(row):
    return row['initial_index'], row['disturbance'], row['controller']


def harmful(row, base):
    return bool(np.any(np.asarray(row['per_vehicle_cost']) - base['per_vehicle_cost'] > 1e-9) or
                np.any(np.asarray(row['per_vehicle_finish']) - base['per_vehicle_finish'] > 1e-9) or
                not row['completed'] or row['collision'] or row['aborted'] or row['contract_violations'])


def select_cases(rows):
    lookup = {identity(row):row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError('Duplicate original cases')
    selected = {}
    for row in rows:
        if not row['controller'].startswith('portfolio0_'):
            continue
        base_key = row['initial_index'], row['disturbance'], 'baseline'
        if harmful(row,lookup[base_key]):
            selected[identity(row)] = 'all_strict_gate_harmful_runs'
            selected[base_key] = 'paired_baseline_for_harmful_run'
    return [dict(initial_index=key[0], disturbance=key[1], controller=key[2], reason=selected[key])
            for key in sorted(selected, key=lambda key:(key[0],key[1],key[2]!='baseline',key[2]))]


def prepare():
    if PLAN.exists() or OUTPUT.exists():
        raise FileExistsError('Preserve the existing diagnostic plan or results')
    read_protocol(STUDY)
    protocol,lookup,_,_,_ = complete_study(STUDY,'risk_validation')
    rows = list(lookup.values())
    cases = select_cases(rows)
    harm_cases = [case for case in cases if case['controller'] != 'baseline']
    sources = [Path(__file__).resolve(), ROOT/'tests/test_calibration_harm_replay_v19.py',
               ROOT/'tools/resume_frozen_risk_v19.py', ROOT/'tools/experiment_ego_v13.py',
               ROOT/'tools/verify_development_v14.py', ROOT/'tools/study_gate_v19.py']
    plan = dict(purpose='Post-calibration mechanism diagnosis of all strict-gate harmful runs',
                selection='Every harmful strict-gate run, including any zero-adoption/physical failure, plus each distinct paired baseline',
                source_study_actual_sumo_runs=len(rows), independent_initials=protocol['spec']['n_initials'],
                harmful_runs=len(harm_cases), paired_baseline_runs=len(cases)-len(harm_cases), expected_sumo_replays=len(cases),
                cases=cases, training_models=protocol['models'],
                outcome_dependent_selection=True, additional_independent_risk_samples=0,
                final_confirmation_outcomes_read=False, online_controller_uses_realized_future=False,
                hindsight_scope='Evaluate the frozen feedback baseline along the realized full disturbance tape only AFTER simulation. This oracle is diagnostic, never deployable.',
                numeric_identity_tolerance=1e-9, gate_admission_tolerance=1e-10,
                study_protocol_sha256=digest(STUDY/'protocol.json'), study_episodes_sha256=digest(STUDY/'episodes.json'),
                source_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in sources})
    PLAN.mkdir(parents=True)
    save(PLAN/'plan.json',plan)
    for path in plan['source_sha256']:
        target=PLAN/'source'/path
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/path,target)
    save(PLAN/'sha256.json',{p.relative_to(PLAN).as_posix():digest(p) for p in PLAN.rglob('*') if p.is_file() and p.name!='sha256.json'})
    print(json.dumps({k:plan[k] for k in ('harmful_runs','paired_baseline_runs','expected_sumo_replays','outcome_dependent_selection')}),flush=True)


def hindsight(row,trace,base):
    if not row['completed'] or row['collision'] or row['aborted'] or row['contract_violations']:
        return dict(available=False,reason='Keep physical failures, but completed-trajectory telescoping is not applied to administrative penalties')
    config=MergeConfig()
    packed=parameters(config)
    tape=disturbance_tape(row['seed'],row['disturbance'],.5,round(config.horizon/config.dt))
    states=np.asarray([row['initial']]+[item['x'] for item in trace])
    commands=np.asarray([item['command'] for item in trace])
    previous=np.concatenate((np.zeros((1,3)),commands),axis=0)
    applied=np.asarray([error for item in trace for error in item['disturbance']])
    np.testing.assert_array_equal(applied,tape[:len(applied)])
    potentials=[]
    for step,state in enumerate(states):
        try:
            value,_,_=disturbed_suffix(state,previous[step],step,tape,packed)
        except RuntimeError as error:
            return dict(available=False,reason='counterfactual_baseline_forecast_exception',
                        step=step,error=str(error))
        potentials.append(value)
    potentials=np.asarray(potentials)
    active=states[:-1,:,0]<config.goal
    stages=np.stack((active*(config.decision_dt+.05*np.abs(commands-previous[:-1])),active*config.decision_dt),axis=-1)
    differences=stages+potentials[1:]-potentials[:-1]
    original=np.column_stack((row['per_vehicle_cost'],row['per_vehicle_finish']))-np.column_stack((base['per_vehicle_cost'],base['per_vehicle_finish']))
    baseline_value=np.column_stack((base['per_vehicle_cost'],base['per_vehicle_finish']))
    base_error=float(np.max(np.abs(potentials[0]-baseline_value)))
    identity_error=float(np.max(np.abs(differences.sum(0)-original)))
    steps=[]
    for k,item in enumerate(trace):
        admitted=bool(item.get('portfolio_joint_action_admitted',False))
        bank=np.asarray(item['selected_bank_differences']) if admitted else None
        if admitted and (bank.shape!=(17,3,2) or not np.all(bank<=1e-10)):
            raise ValueError('Original finite-bank admission invariant failed')
        steps.append(dict(step=k, time_before=k*config.decision_dt, admitted=admitted,
                          composition=item.get('composition',0), hindsight_bellman_difference=differences[k].tolist(),
                          baseline_potential=potentials[k].tolist(), stage_cost_time=stages[k].tolist(),
                          bank_componentwise_max=bank.max(0).tolist() if admitted else None,
                          positive_hindsight_residual=bool(np.any(differences[k]>1e-9))))
    return dict(available=True,baseline_forecast_matches_actual=base_error<=1e-9,
                telescoping_matches_recorded_regret=identity_error<=1e-9,
                baseline_forecast_max_error=base_error,telescoping_max_error=identity_error,
                original_regret=original.tolist(), summed_hindsight_residual=differences.sum(0).tolist(),
                positive_admission_steps=sum(s['admitted'] and s['positive_hindsight_residual'] for s in steps),
                positive_fallback_steps=sum(not s['admitted'] and s['positive_hindsight_residual'] for s in steps),
                all_admitted_bank_checks_passed=True, steps=steps)


def run():
    torch.set_num_threads(1)
    verify_manifest(PLAN)
    require_committed([PLAN/'plan.json',PLAN/'sha256.json'])
    plan=json.loads((PLAN/'plan.json').read_text())
    for path,expected in plan['source_sha256'].items():
        if digest(ROOT/path)!=expected or digest(PLAN/'source'/path)!=expected:
            raise ValueError('Frozen diagnostic source changed: '+path)
    _,protocol=read_protocol(STUDY)
    verify_manifest(STUDY)
    if digest(STUDY/'protocol.json')!=plan['study_protocol_sha256'] or digest(STUDY/'episodes.json')!=plan['study_episodes_sha256']:
        raise ValueError('Completed calibration identity changed')
    original=json.loads((STUDY/'episodes.json').read_text())
    lookup={identity(row):row for row in original}
    if select_cases(original)!=plan['cases']:
        raise ValueError('Diagnostic coverage changed')
    OUTPUT.mkdir(parents=True,exist_ok=False)
    fs=factories(protocol)
    warm=np.array([[-80.,12.],[-85.,13.],[-120.,11.]])
    for factory in fs.values():
        factory(warm).select(warm,np.zeros(3))
    rows,diagnostics=[],[]
    with DisturbedMergeEnv() as env:
        for number,case in enumerate(plan['cases']):
            key=case['initial_index'],case['disturbance'],case['controller']
            reference=lookup[key]
            method=protocol['spec']['methods'][case['controller']]
            row,trace,initial=run_episode(env,reference['seed'],reference['scenario'],case['disturbance'],fs[case['controller']],True)
            row,adopted,checks,failures=build_record(row,trace,initial,case['initial_index'],case['controller'],method,protocol['spec'])
            save(OUTPUT/f'row_{number}.json',row)
            save(OUTPUT/f'trace_{number}.json',trace)
            compare_boundary(row,reference)
            check_episode(trace,row,method,adopted,checks,failures)
            archived=STUDY/f"shard_{case['initial_index']%protocol['spec']['shards']}"/f"trace_{case['initial_index']}_{case['disturbance']}_{case['controller']}.json"
            if archived.exists():
                old=json.loads(archived.read_text())
                if len(old)!=len(trace):
                    raise ValueError('Archived trajectory length differs')
                for actual,saved in zip(trace,old):
                    for field in ('command','x','disturbance'):
                        np.testing.assert_array_equal(actual[field],saved[field])
            baseline_key=case['initial_index'],case['disturbance'],'baseline'
            analysis=hindsight(row,trace,lookup[baseline_key])
            save(OUTPUT/f'mechanism_{number}.json',analysis)
            diagnostics.append(dict(case_index=number,**case,all_nontiming_metrics_equal=True,
                                    archived_trajectory_compared=archived.exists(),
                                    mechanism={k:v for k,v in analysis.items() if k!='steps'}))
            rows.append(row)
            save(OUTPUT/'progress.json',dict(completed_replays=len(rows),expected_replays=len(plan['cases'])))
            print(json.dumps(dict(completed_replays=len(rows),expected_replays=len(plan['cases']))),flush=True)
    harms=[d for d in diagnostics if d['controller']!='baseline']
    complete_mechanisms=all(d['mechanism']['available'] and d['mechanism']['baseline_forecast_matches_actual'] and
                            d['mechanism']['telescoping_matches_recorded_regret'] for d in diagnostics)
    report=dict(actual_sumo_replays=len(rows),source_harmful_runs=len(harms),paired_baseline_runs=len(rows)-len(harms),
                all_selected_non_timing_metrics_match=True,all_paired_physical_traces_reconstructed=True,
                all_hindsight_identities_match=complete_mechanisms,
                harmful_runs_with_positive_admission_residual=sum(d['mechanism'].get('positive_admission_steps',0)>0 for d in harms),
                positive_hindsight_admission_steps=sum(d['mechanism'].get('positive_admission_steps',0) for d in harms),
                positive_hindsight_fallback_steps=sum(d['mechanism'].get('positive_fallback_steps',0) for d in harms),
                per_model_harmful_runs=dict(Counter(d['controller'] for d in harms)),
                additional_independent_risk_samples=0,final_confirmation_outcomes_read=False,
                outcome_dependent_selection=True,online_controller_uses_realized_future=False,
                risk_certified_by_diagnosis=False,policy_changes=0,
                plan_sha256=digest(PLAN/'plan.json'),study_episodes_sha256=digest(STUDY/'episodes.json'),cases=diagnostics)
    save(OUTPUT/'episodes.json',rows)
    save(OUTPUT/'audit.json',report)
    save(OUTPUT/'sha256.json',{p.relative_to(OUTPUT).as_posix():digest(p) for p in OUTPUT.rglob('*') if p.is_file() and p.name!='sha256.json'})
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','run'])
    args=parser.parse_args()
    prepare() if args.command=='prepare' else run()
