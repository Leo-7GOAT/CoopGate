"""Verify augmented-state predictions across every old baseline and selected real SUMO replays."""
import argparse
from collections import Counter, defaultdict
import ctypes
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from coopgate.actuator_prediction_v30 import predict_suffix
from coopgate.disturbance_v9 import disturbance_tape
from coopgate.episode_store_v21 import EpisodeStore, encode, observations
from coopgate.joint_planner_v15 import JointRolloutGate
from coopgate.matched_shift_v21 import MatchedShiftedMergeEnv, audit_matched_trace, tape_fingerprint
from coopgate.shifted_sumo_v19 import PLANTS, SCENARIOS
from diagnose_regression_v28 import compact_trace, paired_replay_check
from execute_prospective_v23 import exclusive_json, runtime_identity, sources_committed, utc
from experiment_ego_v13 import run_episode
from present_completed_v21 import load_complete
from profile_phase_runtime_v24 import committed, digest
from recover_checkpoint_v19 import atomic_bytes
import recover_interrupted_v21 as original

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/v30/actuator_prediction_audit'
OLD = ROOT/'artifacts/v21/interrupted_recovery_20261022/results/transfer'
BASE = ROOT/'artifacts/v27/physical_switch_comparison/plan.json'
SOURCES = ('src/coopgate/actuator_prediction_v30.py', 'tools/audit_actuator_prediction_v30.py',
    'tests/test_actuator_prediction_v30.py', 'tests/test_actuator_prediction_audit_v30.py',
    'tools/diagnose_regression_v28.py', 'src/coopgate/regression_diagnostic_v28.py',
    'src/coopgate/physical_comparison_analysis_v27.py')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def case(row):
    return original.identity('transfer', row)


def scope(rows):
    selected = sorted([r for r in rows if r['controller'] == 'baseline'], key=case)
    keys = [case(r) for r in selected]
    if (len(keys) != 640 or len(set(keys)) != 640
            or Counter((r['scenario'],r['plant_name']) for r in selected)
            != Counter({(s,p):32 for s in SCENARIOS for p in PLANTS})
            or any({r['initial_index'] for r in selected if (r['scenario'],r['plant_name']) == (s,p)} != set(range(32))
                   for s in SCENARIOS for p in PLANTS)):
        raise ValueError('All 640 original baseline cases and all 20 scenario/plant cells are required')
    # Fixed coverage plus every known abort; neither condition filters for success.
    replay = [r for r in selected if r['initial_index'] == 0 or r['aborted']]
    return selected, replay


def comparison(prediction, row, trace):
    p = prediction
    states = np.asarray([t['x'] for t in trace])
    predicted_states = np.asarray([t['x'] for t in p['trace']])
    same_length = len(p['trace']) == len(trace)
    state_error = float(np.abs(states-predicted_states).max()) if same_length and len(trace) else (0. if same_length else None)
    commands_exact = same_length and all(a['command'] == b['command'] for a,b in zip(p['trace'],trace))
    expected = np.column_stack((row['per_vehicle_cost'], row['per_vehicle_finish']))
    metric_error = float(np.max(np.abs(np.asarray(p['values'])-expected)))
    error_identity = ((p['error'] is None) == (not row['aborted'])
        and (p['error'] is None or p['error']['message'] == row['abort_reason']))
    if same_length:
        memory_error = max((float(np.max(np.abs(np.asarray(a['actuator_states'])-b['actuator_states'])))
                            for a,b in zip(p['trace'],trace)), default=0.)
        actual_error_difference = max((float(np.max(np.abs(np.asarray(a['disturbance'])-b['disturbance'])))
                                      for a,b in zip(p['trace'],trace)), default=0.)
    else:
        memory_error = actual_error_difference = None
    checks = dict(commands_exact=commands_exact, same_steps=same_length,
        states_match=state_error is not None and state_error <= 1e-8,
        metrics_match=metric_error <= 1e-9, completion_match=p['complete'] == row['completed'],
        abort_identity_match=error_identity, route_checks_match=p['route_violations'] == row['contract_violations'],
        actuator_states_match=memory_error is not None and memory_error <= 1e-9,
        effective_errors_match=actual_error_difference is not None and actual_error_difference <= 1e-9,
        error_bound_counts_match=p['error_contract_exceeded_substeps'] == row['error_contract_exceeded_substeps'],
        maximum_error_match=abs(p['max_actual_error']-row['max_actual_error']) <= 1e-9)
    return dict(all_match=all(checks.values()), checks=checks, state_max_abs_error=state_error,
        original_metric_max_abs_error=metric_error, actuator_max_abs_error=memory_error,
        effective_error_max_abs_error=actual_error_difference, original_aborted=row['aborted'],
        original_completed=row['completed'], original_error_bound_exceeded=row['error_contract_exceeded_substeps'] > 0,
        commands=len(trace))


def freeze():
    load_complete()
    base = read(BASE)
    rows = read(OLD/'episodes.json')
    original.transfer.validate_grid(rows, base['protocols']['transfer']['spec'])
    selected, replay = scope(rows)
    sources = dict(base['source_sha256'])
    sources.update({name:digest(ROOT/name) for name in SOURCES})
    for name, expected in sources.items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen dependency changed: '+name)
    sources_committed([ROOT/name for name in sources])
    provenance = {tuple(p['case']):p for p in read(OLD/'provenance.json')}
    OUT.mkdir(parents=True, exist_ok=False)
    meta = dict(schema='v30_all_transfer_baseline_inputs', old_episodes_sha256=digest(OLD/'episodes.json'))
    store = EpisodeStore(OUT/'inputs.sqlite', meta)
    try:
        for row in selected:
            reference = provenance[case(row)]
            path = ROOT/reference['path']
            if digest(path) != reference['sha256']:
                raise ValueError('Old reviewed trace changed')
            store.append(case(row), dict(row=row, original_trace=reference), compact_trace(read(path)))
        store.seal([case(r) for r in selected])
    finally:
        store.close()
    exclusive_json(OUT/'plan.json', dict(utc=utc(), source_sha256=sources, runtime=runtime_identity(),
        input_metadata=meta, input_database_sha256=digest(OUT/'inputs.sqlite'),
        input_seal_sha256=digest(OUT/'inputs.sqlite.seal.json'), input_cases=[list(case(r)) for r in selected],
        replay_cases=[list(case(r)) for r in replay], original_aborts=sum(r['aborted'] for r in selected),
        offline_baselines=640, real_sumo_replays=len(replay),
        selection='Every original baseline. Real replays: first preassigned initial in every scenario/plant cell plus every original baseline abort, with duplicates removed.',
        scientific_scope='Known actuator parameters and realized future external innovations are diagnostic inputs. No source command-dependent effective errors are transplanted; each predicted branch propagates its own actuator memory.',
        prediction_rule='The unchanged feedback baseline, physical checks, original horizon and metric; record all mismatches and all aborts without skipping cases or snapping forecasts to observed states.',
        new_controller=False, new_independent_initials=0, new_risk_certificate=False,
        active_calibration_outcomes_read=False))
    print(json.dumps(dict(offline_baselines=640, real_sumo_replays=len(replay), original_aborts=sum(r['aborted'] for r in selected))), flush=True)


def inputs():
    committed(OUT/'plan.json')
    plan = read(OUT/'plan.json')
    if (runtime_identity() != plan['runtime'] or digest(OUT/'inputs.sqlite') != plan['input_database_sha256']
            or digest(OUT/'inputs.sqlite.seal.json') != plan['input_seal_sha256']):
        raise ValueError('Frozen prediction runtime/input changed')
    for name, expected in plan['source_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen prediction source changed: '+name)
    sources_committed([ROOT/name for name in plan['source_sha256']])
    return plan


def metadata(stage):
    return dict(plan_sha256=digest(OUT/'plan.json'), stage=stage)


def run():
    plan = inputs()
    execution = OUT/'execution'
    execution.mkdir(exist_ok=False)
    if os.name == 'nt':
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetPriorityClass.argtypes = (ctypes.c_void_p,ctypes.c_uint32)
        kernel.SetPriorityClass.restype = ctypes.c_int
        if not kernel.SetPriorityClass(kernel.GetCurrentProcess(),subprocess.BELOW_NORMAL_PRIORITY_CLASS):
            raise ctypes.WinError(ctypes.get_last_error())
    torch.set_num_threads(1)
    exclusive_json(execution/'start.json', dict(native_pid=os.getpid(), parent_pid=os.getppid(), utc=utc(),
        plan_sha256=digest(OUT/'plan.json')))
    records = {key:(record['row'],trace) for key,record,trace in observations(
        OUT/'inputs.sqlite', plan['input_metadata'], plan['input_cases'])}
    offline = OUT/'offline'
    offline.mkdir()
    store = EpisodeStore(offline/'observations.sqlite', metadata('offline'))
    predictions = {}
    try:
        for index, raw_case in enumerate(plan['input_cases']):
            key = tuple(raw_case)
            row, trace = records[key]
            audit_matched_trace(trace,row)
            tape = disturbance_tape(row['seed'], row['disturbance'], row['external_innovation_radius'],250)
            if tape_fingerprint(tape) != row['innovation_tape_sha256']:
                raise ValueError('Original innovation identity differs')
            prediction = predict_suffix(np.asarray(row['initial']),np.zeros(3),np.zeros(3),0,tape,row['plant'],record_trace=True)
            steps = prediction.pop('trace')
            review = comparison(dict(prediction,trace=steps), row, trace)
            store.append(key, dict(prediction=prediction,comparison=review), steps)
            if raw_case in plan['replay_cases']:
                predictions[key] = (prediction,steps)
            atomic_bytes(execution/'progress.json',encode(dict(phase='offline',completed=index+1,expected=640,native_pid=os.getpid(),utc=utc())))
        store.seal(plan['input_cases'])
    finally:
        store.close()
    exclusive_json(offline/'complete.json',dict(cases=640,database_sha256=digest(offline/'observations.sqlite'),seal_sha256=digest(offline/'observations.sqlite.seal.json')))
    replay = OUT/'replays'
    replay.mkdir()
    store = EpisodeStore(replay/'observations.sqlite',metadata('replays'))
    reviews = []
    try:
        for plant in PLANTS:
            with MatchedShiftedMergeEnv(plant=plant) as env:
                for raw_case in plan['replay_cases']:
                    key = tuple(raw_case)
                    if key[4] != plant:
                        continue
                    old, old_trace = records[key]
                    row, trace, initial = run_episode(env, old['seed'],old['scenario'],old['disturbance'],lambda _:JointRolloutGate(),True)
                    row.update(initial=initial.tolist(),plant_name=plant,controller='baseline',
                        per_vehicle_cost=(np.asarray(row['per_vehicle_finish'])+.05*np.asarray(row['per_vehicle_variation'])).tolist())
                    row['team_cost'] = float(np.mean(row['per_vehicle_cost']))
                    store.append(key,dict(row=row),trace)
                    exact = paired_replay_check(row,trace,old,old_trace)
                    prediction, steps = predictions[key]
                    reviews.append(dict(case=raw_case,original_replay=exact,model_comparison=comparison(dict(prediction,trace=steps),row,trace)))
                    atomic_bytes(execution/'progress.json',encode(dict(phase='sumo_replays',completed=len(reviews),expected=plan['real_sumo_replays'],native_pid=os.getpid(),utc=utc())))
        expected = [key for plant in PLANTS for key in plan['replay_cases'] if key[4] == plant]
        store.seal(expected)
    finally:
        store.close()
    exclusive_json(replay/'complete.json',dict(actual_sumo_runs=len(reviews),reviews=reviews,
        database_sha256=digest(replay/'observations.sqlite'),seal_sha256=digest(replay/'observations.sqlite.seal.json')))
    exclusive_json(execution/'collection_complete.json',dict(offline_baselines=640,actual_sumo_runs=len(reviews),
        native_pid=os.getpid(),plan_sha256=digest(OUT/'plan.json'),utc=utc()))
    print(json.dumps(dict(offline_baselines=640,actual_sumo_runs=len(reviews))),flush=True)


def summarize():
    plan = inputs()
    start, exit, done = [read(OUT/'execution'/name) for name in ('start.json','observation.json','collection_complete.json')]
    if (exit['exit_code'] != 0 or exit['native_pid'] != start['native_pid'] or done['native_pid'] != start['native_pid']
            or any(item['plan_sha256'] != digest(OUT/'plan.json') for item in (start,exit,done))
            or done['offline_baselines'] != 640 or done['actual_sumo_runs'] != plan['real_sumo_replays']):
        raise ValueError('The whole declared scope and an actual successful native exit are required')
    groups, mismatches, offline_rows = defaultdict(list), [], []
    for stage in ('offline','replays'):
        directory = OUT/stage
        receipt = read(directory/'complete.json')
        if (digest(directory/'observations.sqlite') != receipt['database_sha256']
                or digest(directory/'observations.sqlite.seal.json') != receipt['seal_sha256']):
            raise ValueError('Completed prediction observations changed')
        expected = plan['input_cases'] if stage == 'offline' else [key for p in PLANTS for key in plan['replay_cases'] if key[4] == p]
        for key, record, trace in observations(directory/'observations.sqlite',metadata(stage),expected):
            if stage == 'offline':
                check = record['comparison']
                groups[key[4]].append(check)
                offline_rows.append(dict(case=list(key),**check))
                if not check['all_match']:
                    mismatches.append(dict(case=list(key),**check))
            else:
                audit_matched_trace(trace,record['row'])
    real = read(OUT/'replays/complete.json')
    output = OUT/'analysis'
    output.mkdir(exist_ok=False)
    exclusive_json(output/'report.json',dict(plan_sha256=digest(OUT/'plan.json'),exit_sha256=digest(OUT/'execution/observation.json'),
        offline_baselines=len(offline_rows),real_sumo_replays=real['actual_sumo_runs'],
        by_plant={p:dict(cases=len(v),matched=sum(r['all_match'] for r in v),aborted=sum(r['original_aborted'] for r in v),
            error_bound_exceeded=sum(r['original_error_bound_exceeded'] for r in v),commands=sum(r['commands'] for r in v),
            maximum_state_error=max((r['state_max_abs_error'] for r in v if r['state_max_abs_error'] is not None),default=None)) for p,v in groups.items()},
        all_cases=offline_rows,mismatches=mismatches,real_replay_checks=real['reviews'],
        exact_model_known=True,realized_future_noise_diagnostic_only=True,new_controller=False,
        new_physical_certificate=False,new_independent_initials=0,new_risk_certificate=False))
    exclusive_json(output/'sha256.json',{'report.json':digest(output/'report.json')})
    print(json.dumps(dict(offline_baselines=len(offline_rows),real_sumo_replays=real['actual_sumo_runs'],mismatches=len(mismatches))),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=('freeze','run','summarize'))
    globals()[parser.parse_args().action]()
