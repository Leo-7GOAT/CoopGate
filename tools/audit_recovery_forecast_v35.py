"""Check complete recovery forecasts against every archived V34 SUMO episode.

True coefficients and realized innovations are diagnostic inputs only. They
must never become inputs to a deployed confidence gate.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from coopgate.actuator_recovery_v34 import class_bounds, memory_step
from coopgate.disturbance_v9 import disturbance_tape
from coopgate.episode_store_v21 import EpisodeStore, encode, observations
from coopgate.fast_v8 import parameters
from coopgate.merge_v4 import MergeConfig
from coopgate.recovery_forecast_v35 import forecast, recovery_command
from execute_prospective_v23 import exclusive_json, runtime_identity, sources_committed, utc
from profile_phase_runtime_v24 import committed, digest
from recover_checkpoint_v19 import atomic_bytes
import study_actuator_recovery_v34 as source

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/v35/recovery_prediction_audit'
SOURCES = ('src/coopgate/recovery_forecast_v35.py','tests/test_recovery_forecast_v35.py',
           'tools/audit_recovery_forecast_v35.py','tests/test_recovery_prediction_audit_v35.py')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def freeze():
    prior = source.inputs()
    report = read(source.OUT/'analysis/report.json')
    receipt = read(source.OUT/'completion_verification.json')
    if (not report['all_independent_physical_and_observer_reviews_pass']
            or receipt['report_sha256'] != digest(source.OUT/'analysis/report.json')
            or receipt['summary_exit_code'] != 0 or report['actual_sumo_runs'] != 640):
        raise ValueError('Complete, independently reviewed actual recovery executions required')
    hashes = dict(prior['source_sha256'])
    hashes.update({p:digest(ROOT/p) for p in SOURCES})
    for name,expected in hashes.items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen source changed: '+name)
    sources_committed([ROOT/name for name in hashes])
    files = [source.OUT/name for name in ('plan.json','observations.sqlite','observations.sqlite.seal.json',
             'analysis/report.json','completion_verification.json')]
    OUT.mkdir(parents=True,exist_ok=False)
    exclusive_json(OUT/'plan.json',dict(utc=utc(),runtime=runtime_identity(),source_sha256=hashes,
        inputs_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in files},
        cases=prior['cases'],episodes=640,expected_commands=sum(r['review']['commands'] for r in report['cases']),
        scope='All 640 completed V34 episodes and every one of their control decisions, including all original nominal-baseline failure conditions.',
        diagnostic_information='Known realized per-vehicle plant parameters and complete realized external innovations. This is an offline identity check, not an online admissible-information experiment.',
        prediction='One uninterrupted whole-episode forecast from the initial state. No observed-state correction or future command replay is inserted into the predictor.',
        contracts='Carry the full command-history actuator interval separately from the reference actuator point. Use the same stopping terminal set and recovery baseline at every predicted decision.',
        new_sumo_runs=0,new_independent_initials=0,new_risk_certificate=False,
        active_calibration_outcomes_read=False,partial_v32_effectiveness_read=False))
    print(json.dumps(dict(episodes=640,expected_commands=sum(r['review']['commands'] for r in report['cases']),
                         plan_sha256=digest(OUT/'plan.json'))),flush=True)


def inputs():
    committed(OUT/'plan.json')
    plan = read(OUT/'plan.json')
    for name,expected in {**plan['source_sha256'],**plan['inputs_sha256']}.items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen predictor input changed: '+name)
    sources_committed([ROOT/name for name in plan['source_sha256']])
    if runtime_identity() != plan['runtime']:
        raise ValueError('Frozen runtime changed')
    source.inputs()
    return plan


def observed_crossings(row, trace):
    state = np.asarray(row['initial']).copy()
    crossing = np.where(state[:,0] >= 60.,0.,np.nan)
    gain,drag = np.asarray(row['plant']['gain']),np.asarray(row['plant']['drag'])
    for step,item in enumerate(trace):
        for sub in range(5):
            before = state[:,0].copy()
            acceleration = gain*np.asarray(item['actuator_states'][sub])-drag*state[:,1]**2+np.asarray(item['innovation'][sub])
            state[:,1] = np.clip(state[:,1]+.1*acceleration,0.,20.)
            state[:,0] += .1*state[:,1]
            crossed = np.isnan(crossing) & (before < 60.) & (state[:,0] >= 60.)
            crossing[crossed] = step*.5+sub*.1+.1*(60.-before[crossed])/(state[crossed,0]-before[crossed])
        np.testing.assert_allclose(state,item['x'],atol=1e-8,rtol=0)
        state = np.asarray(item['x']).copy()
    return crossing


def compare_prediction(predicted,row,trace):
    checks = dict(complete=predicted['complete'] == row['completed'],
        error_free=predicted['error_code'] == 0,
        trace_length=predicted['steps'] == len(trace) == len(predicted['trace']),
        route_violations=predicted['route_violations'] == row['contract_violations'],
        legacy_error_count=predicted['legacy_error_exceeded_substeps'] == row['error_contract_exceeded_substeps'])
    errors = {}
    values = np.column_stack((row['per_vehicle_cost'],row['per_vehicle_finish']))
    lo,hi = np.asarray(trace[-1]['actuator_memory_lower']),np.asarray(trace[-1]['actuator_memory_upper'])
    for _ in range(5):
        lo,hi = memory_step(lo,hi,np.asarray(trace[-1]['command']),class_bounds(row['plant_name']))
    expected = dict(values=values,final_state=trace[-1]['x'],final_actuator=trace[-1]['actuator_states'][-1],
                    final_observer_lower=lo,final_observer_upper=hi,
                    max_actual_error=row['max_actual_error'])
    for key,value in expected.items():
        errors[key] = float(np.max(np.abs(np.asarray(predicted[key])-value)))
        checks[key] = errors[key] <= 1e-8
    actual_crossing = observed_crossings(row,trace)
    forecast_crossing = np.asarray([np.nan if value is None else value for value in predicted['crossing']])
    checks['crossing'] = bool(np.allclose(forecast_crossing,actual_crossing,atol=1e-8,rtol=0,equal_nan=True))
    checks['commands_exact'] = checks['trace_length'] and all(a['command'] == b['command'] for a,b in zip(predicted['trace'],trace))
    checks['decision_times'] = checks['trace_length'] and all(abs(a['t']-b['t']) <= 1e-10 for a,b in zip(predicted['trace'],trace))
    checks['recovery_identity'] = checks['trace_length'] and all(a['recovery_applied'] == b['recovery_applied'] for a,b in zip(predicted['trace'],trace))
    for field in ('x','actuator_states','disturbance','actuator_memory_lower','actuator_memory_upper'):
        errors[field] = max((float(np.max(np.abs(np.asarray(a[field])-b[field])))
                            for a,b in zip(predicted['trace'],trace)),default=0.)
        checks[field] = checks['trace_length'] and errors[field] <= 1e-8
    return dict(all_match=all(checks.values()),checks=checks,maximum_absolute_errors=errors)


def run():
    plan = inputs()
    exclusive_json(OUT/'start.json',dict(native_pid=os.getpid(),utc=utc(),plan_sha256=digest(OUT/'plan.json')))
    prior = read(source.OUT/'plan.json')
    store = EpisodeStore(OUT/'predictions.sqlite',dict(plan_sha256=digest(OUT/'plan.json'),stage='offline_prediction'))
    results = []
    total_commands = 0
    packed = parameters(MergeConfig())
    try:
        for key,record,trace in observations(source.OUT/'observations.sqlite',source.metadata(),prior['cases']):
            row = record['row']
            bounds = class_bounds(key[4])
            state,previous = np.asarray(row['initial']),np.zeros(3)
            command_checks = []
            for step,item in enumerate(trace):
                current = recovery_command(state,previous,step,np.asarray(item['actuator_memory_lower']),
                    np.asarray(item['actuator_memory_upper']),bounds,packed)
                command_checks.append(dict(step=step,exact_command=bool(np.array_equal(current[0],item['command'])),
                    certified=bool(current[1]),same_recovery=bool(current[2]) == item['recovery_applied'],
                    same_proposal_failure=bool(current[3]) == (item['nominal_proposal_error'] is not None)))
                state,previous = np.asarray(item['x']),np.asarray(item['command'])
            tape = disturbance_tape(row['seed'],row['disturbance'],.1,250)
            predicted = forecast(np.asarray(row['initial']),np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3),
                                 0,tape,row['plant'],bounds,record_trace=True)
            compared = compare_prediction(predicted,row,trace)
            commands_match = all(all(value for name,value in r.items() if name != 'step') for r in command_checks)
            result = dict(case=list(key),command_checks=command_checks,commands=len(trace),
                          all_commands_match=commands_match,**compared)
            predicted_trace = predicted.pop('trace')
            store.append(key,dict(prediction=predicted,review=result),predicted_trace)
            results.append(result)
            total_commands += len(trace)
            atomic_bytes(OUT/'progress.json',encode(dict(utc=utc(),native_pid=os.getpid(),
                completed_episodes=len(results),expected_episodes=plan['episodes'],commands=total_commands)))
        store.seal(plan['cases'])
    finally:
        store.close()
    if len(results) != plan['episodes'] or total_commands != plan['expected_commands']:
        raise ValueError('Incomplete frozen forecast scope')
    passed = all(r['all_match'] and r['all_commands_match'] for r in results)
    report = dict(plan_sha256=digest(OUT/'plan.json'),episodes=len(results),commands=total_commands,
        all_match=passed,cases=results,mismatches=[r['case'] for r in results if not r['all_match'] or not r['all_commands_match']],
        new_sumo_runs=0,new_independent_initials=0,new_risk_certificate=False,
        scope=plan['scope'],diagnostic_information=plan['diagnostic_information'])
    exclusive_json(OUT/'report.json',report)
    exclusive_json(OUT/'sha256.json',{name:digest(OUT/name) for name in ('report.json','predictions.sqlite','predictions.sqlite.seal.json')})
    print(json.dumps(dict(episodes=len(results),commands=total_commands,all_match=passed,mismatches=report['mismatches'])),flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','run'))
    globals()[parser.parse_args().action]()
