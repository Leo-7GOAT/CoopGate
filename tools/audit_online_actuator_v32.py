"""Full predictor equivalence and a fixed 120-run online SUMO integration.

Only development data are used. The two new selectors share candidate generation
and fixed innovation paths; neither receives a randomized true plant or future
innovation. Existing completed baseline/phase05 results are retained comparators.
"""
import argparse
from collections import Counter
import ctypes
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from coopgate.actuator_intervention_v31 import intervention_state
from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.disturbance_v9 import disturbance_tape
from coopgate.episode_store_v21 import EpisodeStore, encode, observations
from coopgate.matched_shift_v21 import MatchedShiftedMergeEnv, audit_matched_trace
from coopgate.online_actuator_gate_v32 import ReferenceActuatorGate, forecast, reference_parameters
from coopgate.physical_comparison_analysis_v27 import harmful
from coopgate.shifted_sumo_v19 import PLANTS, SCENARIOS
from execute_prospective_v23 import exclusive_json, runtime_identity, sources_committed, utc
from experiment_ego_v13 import run_episode
from profile_phase_runtime_v24 import committed, digest
from recover_checkpoint_v19 import atomic_bytes

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/v32/online_actuator_preflight'
INTERVENTIONS = ROOT/'artifacts/v31/actuator_interventions_roundoff'
BASELINES = ROOT/'artifacts/v30/actuator_prediction_audit'
COMPARISON = ROOT/'artifacts/v27/physical_switch_comparison/plan.json'
SOURCES = ('src/coopgate/online_actuator_gate_v32.py','tools/audit_online_actuator_v32.py',
           'tests/test_online_actuator_gate_v32.py','tests/test_online_actuator_protocol_v32.py')
KINDS = ('nominal','reference')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def integration_scope(input_cases):
    selected = [list(k) for plant in PLANTS for k in input_cases if k[4] == plant and k[2] == 0]
    counts = Counter((k[1],k[4],k[5]) for k in selected)
    expected = Counter({(scene,plant,'phase05_'+model):1 for scene in SCENARIOS for plant in PLANTS
                        for model in ('1802','1812','1822')})
    if len(selected) != 60 or len({tuple(k) for k in selected}) != 60 or counts != expected:
        raise ValueError('One preassigned initial in all 60 scene/plant/model cells required')
    return [key+[kind] for key in selected for kind in KINDS]


def freeze():
    prior = read(INTERVENTIONS/'plan.json')
    report = read(INTERVENTIONS/'analysis/report.json')
    receipt = read(INTERVENTIONS/'completion_verification.json')
    if (not report['all_forecasts_match'] or report['actual_sumo_runs'] != 698
            or receipt['report_sha256'] != digest(INTERVENTIONS/'analysis/report.json')):
        raise ValueError('The complete independently checked intervention study is required')
    sources = dict(prior['source_sha256'])
    sources.update({name:digest(ROOT/name) for name in SOURCES})
    for name,expected in sources.items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen dependency changed: '+name)
    sources_committed([ROOT/name for name in sources])
    OUT.mkdir(parents=True,exist_ok=False)
    needed = ('plan.json','inputs.sqlite','inputs.sqlite.seal.json','analysis/report.json','completion_verification.json',
              'offline/observations.sqlite','offline/observations.sqlite.seal.json')
    exclusive_json(OUT/'plan.json',dict(utc=utc(),runtime=runtime_identity(),source_sha256=sources,
        completed_interventions={name:digest(INTERVENTIONS/name) for name in needed},
        completed_baseline_plan_sha256=digest(BASELINES/'plan.json'),
        models=read(COMPARISON)['models'],kernel_cases=prior['admission_cases'],kernel_branches=698,
        integration_schedule=integration_scope(prior['input_cases']),actual_sumo_runs=120,
        reference_parameters={p:reference_parameters(p) for p in PLANTS},
        prediction_models=list(KINDS),innovation_radius=.1,phase_tolerance=.05,
        scope='All prior 349 admission points for predictor equivalence; one fixed initial from each of all 60 scene/plant/model cells for real integration, both new controllers. No removal of strong lag or failures.',
        online_inputs='Current kinematics, actual previous commands, global decision time, and preconfigured class-level reference coefficients only. No true randomized per-episode plant, observed actuator memory, episode seed, or future innovation reaches the selector.',
        comparison='The nominal/reference pair shares the trained actors, candidate generation, 33 predeclared .1-radius innovation paths, all-vehicle cost/time/phase conditions, original metrics, physical checks and fallback. Prediction model and resulting ranking differ.',
        prior_comparators='Completed original baseline and phase05 on exactly the same old conditions. Their different disturbance banks/modeling and search work remain explicit; no equal-compute claim.',
        independence='Old development initials, not new efficacy samples. Integration checks do not confer a population risk certificate.',
        failure_rule='Preserve every abort, timeout, zero adoption and negative gain. A failed predicted baseline cannot supply performance-admission evidence; baseline fallback has no new physical guarantee.',
        new_independent_initials=0,new_risk_certificate=False,active_calibration_outcomes_read=False))
    print(json.dumps(dict(kernel_branches=698,actual_sumo_runs=120,plan_sha256=digest(OUT/'plan.json'))),flush=True)


def inputs():
    committed(OUT/'plan.json')
    plan = read(OUT/'plan.json')
    if runtime_identity() != plan['runtime'] or digest(BASELINES/'plan.json') != plan['completed_baseline_plan_sha256']:
        raise ValueError('Frozen runtime or baseline input changed')
    for name,expected in plan['source_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen online source changed: '+name)
    for name,expected in plan['completed_interventions'].items():
        if digest(INTERVENTIONS/name) != expected:
            raise ValueError('Completed reference evidence changed')
    for model in plan['models'].values():
        if digest(ROOT/model['path']) != model['sha256']:
            raise ValueError('Trained actor changed')
    sources_committed([ROOT/name for name in plan['source_sha256']])
    return plan


def archived():
    plan = read(INTERVENTIONS/'plan.json')
    return {key:(record['row'],trace) for key,record,trace in observations(
        INTERVENTIONS/'inputs.sqlite',plan['input_metadata'],plan['input_cases'])}


def compare_kernel(actual,expected,trace):
    checks = {key:actual[key] == expected[key] for key in
              ('complete','steps','route_violations','error_contract_exceeded_substeps')}
    errors = {}
    for key in ('values','final_state','final_actuator','max_actual_error'):
        errors[key] = float(np.max(np.abs(np.asarray(actual[key])-expected[key])))
        checks[key] = errors[key] <= 1e-9
    if expected['error'] is None:
        checks['error_identity'] = actual['error_code'] == 0 and actual['error_step'] is None
    else:
        message = expected['error']['message']
        code = (1 if message == 'Joint planner outside backup set' else
                3 if message == 'Predicted command lacks the original physical certificate' else 2)
        checks['error_identity'] = actual['error_code'] == code and actual['error_step'] == expected['error']['step']
    a = np.array([np.nan if x is None else x for x in actual['crossing']])
    b = np.array([np.nan if x is None else x for x in expected['crossing']])
    checks['crossings'] = bool(np.allclose(a,b,atol=1e-9,rtol=0,equal_nan=True))
    checks['trace_length'] = len(actual['trace']) == len(trace)
    checks['commands_exact'] = checks['trace_length'] and all(a['command']==b['command'] for a,b in zip(actual['trace'],trace))
    for field in ('x','actuator_states','disturbance'):
        error = max((float(np.max(np.abs(np.asarray(a[field])-b[field]))) for a,b in zip(actual['trace'],trace)),default=0.)
        errors[field] = error
        checks[field] = checks['trace_length'] and error <= 1e-9
    return dict(all_match=all(checks.values()),checks=checks,maximum_absolute_errors=errors)


def kernel():
    plan = inputs()
    directory = OUT/'kernel'
    directory.mkdir(exist_ok=False)
    records = archived()
    meta = dict(plan_sha256=digest(INTERVENTIONS/'plan.json'),stage='offline')
    results = []
    for key,record,traces in observations(INTERVENTIONS/'offline/observations.sqlite',meta,plan['kernel_cases']):
        row,trace = records[key[:-1]]
        state,previous,memory = intervention_state(row,trace,key[-1])
        tape = disturbance_tape(row['seed'],row['disturbance'],row['external_innovation_radius'],250)
        for role in ('baseline_suffix','candidate_then_baseline'):
            command = np.asarray(trace[key[-1]]['command']) if role == 'candidate_then_baseline' else None
            actual = forecast(state,previous,memory,key[-1],tape,row['plant'],command,record_trace=True)
            result = compare_kernel(actual,record['branches'][role],traces[role])
            results.append(dict(case=list(key)+[role],**result))
    if len(results) != 698:
        raise ValueError('All original intervention branches required')
    report = dict(branches=698,all_match=all(r['all_match'] for r in results),
        mismatches=[r for r in results if not r['all_match']],cases=results,new_sumo_runs=0,
        plan_sha256=digest(OUT/'plan.json'),scope='Independent model/actual-replay equivalence, not online future information.')
    exclusive_json(directory/'report.json',report)
    exclusive_json(directory/'sha256.json',dict(report=digest(directory/'report.json')))
    print(json.dumps(dict(branches=698,all_match=report['all_match'],mismatches=len(report['mismatches']))),flush=True)


def execution_priority():
    if os.name == 'nt':
        library = ctypes.WinDLL('kernel32',use_last_error=True)
        library.GetCurrentProcess.restype = ctypes.c_void_p
        library.SetPriorityClass.argtypes = (ctypes.c_void_p,ctypes.c_uint32)
        library.SetPriorityClass.restype = ctypes.c_int
        if not library.SetPriorityClass(library.GetCurrentProcess(),subprocess.BELOW_NORMAL_PRIORITY_CLASS):
            raise ctypes.WinError(ctypes.get_last_error())


def review_online(row,trace,kind,profile):
    audit_matched_trace(trace,row)
    parameters = reference_parameters(profile if kind == 'reference' else 'nominal')
    tau = np.asarray(parameters['tau'])
    rho = np.zeros(3)
    rho[tau > 0] = np.exp(-.1/tau[tau > 0])
    memory = np.zeros(3)
    state = np.asarray(row['initial'])
    count = 0
    for step,item in enumerate(trace):
        if (item['reference_parameters'] != parameters or item['prediction_model'] != kind
                or item['reference_profile'] != profile or item['future_realization_used']
                or item['true_randomized_plant_used'] or item['new_risk_certificate']
                or item['forecast_bank_size'] != 33 or item['external_bank_radius'] != .1):
            raise ValueError('Online model/input contract changed')
        np.testing.assert_array_equal(item['reconstructed_actuator'],memory)
        active = state[:,0] < 60.
        actual_adoption = bool(np.any(np.abs(np.asarray(item['command'])[active]-np.asarray(item['baseline_command'])[active]) > 1e-9))
        if actual_adoption != item['nominal_joint_action_admitted']:
            raise ValueError('Adoption must mean an executed active-vehicle command differs from baseline')
        if actual_adoption:
            values,phases = np.asarray(item['selected_bank_differences']),np.asarray(item['selected_phase_differences'])
            if (values.shape != (33,3,2) or phases.shape != (33,3) or not np.isfinite(values).all()
                    or not np.isfinite(phases).all() or np.any(values > 1e-10) or np.any(phases > .05+1e-10)
                    or values[0,:,0].mean() >= -1e-9):
                raise ValueError('Full componentwise fixed-bank evidence is required for every admission')
        count += actual_adoption
        for _ in range(5): memory = rho*memory+(1-rho)*np.asarray(item['command'])
        state = np.asarray(item['x'])
    return dict(physical_trace_reconstructed=True,reference_memory_exact=True,actual_admitted_steps=int(count),
        all_admissions_have_full_finite_bank=True,new_risk_certificate=False)


def run():
    plan = inputs()
    if (digest(OUT/'kernel/report.json') != read(OUT/'kernel/sha256.json')['report']
            or not read(OUT/'kernel/report.json')['all_match']):
        raise ValueError('Complete predictor verification must pass before real integration')
    execution = OUT/'execution'
    execution.mkdir(exist_ok=False)
    execution_priority()
    torch.set_num_threads(1)
    exclusive_json(execution/'start.json',dict(native_pid=os.getpid(),parent_pid=os.getppid(),utc=utc(),plan_sha256=digest(OUT/'plan.json')))
    models = {name:load_coupled(ROOT/data['path'])[0] for name,data in plan['models'].items()}
    records = archived()
    # Warm compilation and all fixed profiles on one explicit synthetic state,
    # with no simulator execution or evaluation initial involved.
    started = time.perf_counter()
    for profile in PLANTS:
        for kind in KINDS:
            ReferenceActuatorGate(models['1802'],profile,kind).select(np.array([[-80.,12.],[-85.,13.],[-120.,11.]]),np.zeros(3))
    exclusive_json(execution/'warmup.json',dict(seconds=time.perf_counter()-started,profiles=4,variants=2,actual_sumo_runs=0))
    store = EpisodeStore(OUT/'observations.sqlite',dict(plan_sha256=digest(OUT/'plan.json'),stage='online_integration'))
    completed = 0
    try:
        for profile in PLANTS:
            with MatchedShiftedMergeEnv(plant=profile) as env:
                for key in (k for k in plan['integration_schedule'] if k[4] == profile):
                    source,_ = records[tuple(key[:-1])]
                    kind,model = key[-1],models[key[5].rsplit('_',1)[1]]
                    factory = lambda _,model=model,kind=kind:ReferenceActuatorGate(model,profile,kind)
                    row,trace,initial = run_episode(env,source['seed'],source['scenario'],source['disturbance'],factory,True)
                    costs = np.asarray(row['per_vehicle_finish'])+.05*np.asarray(row['per_vehicle_variation'])
                    row.update(initial=initial.tolist(),plant_name=profile,controller=kind+'_'+key[5].rsplit('_',1)[1],
                        per_vehicle_cost=costs.tolist(),team_cost=float(costs.mean()))
                    review = review_online(row,trace,kind,profile)
                    store.append(key,dict(row=row,review=review),trace)
                    completed += 1
                    atomic_bytes(execution/'progress.json',encode(dict(actual_sumo_runs=completed,expected_sumo_runs=120,native_pid=os.getpid(),utc=utc())))
        store.seal(plan['integration_schedule'])
    finally:
        store.close()
    exclusive_json(execution/'collection_complete.json',dict(native_pid=os.getpid(),utc=utc(),actual_sumo_runs=completed,
        plan_sha256=digest(OUT/'plan.json'),database_sha256=digest(OUT/'observations.sqlite'),seal_sha256=digest(OUT/'observations.sqlite.seal.json')))
    print(json.dumps(dict(actual_sumo_runs=completed)),flush=True)


def summarize():
    plan = inputs()
    start,done,exit = [read(OUT/'execution'/name) for name in ('start.json','collection_complete.json','observation.json')]
    if (exit['exit_code'] != 0 or done['actual_sumo_runs'] != 120 or exit['native_pid'] != start['native_pid']
            or done['native_pid'] != start['native_pid']
            or any(r['plan_sha256'] != digest(OUT/'plan.json') for r in (start,done,exit))
            or digest(OUT/'observations.sqlite') != done['database_sha256']
            or digest(OUT/'observations.sqlite.seal.json') != done['seal_sha256']):
        raise ValueError('The whole integration scope and actual successful native exit are required')
    old = archived()
    base_plan = read(BASELINES/'plan.json')
    baselines = {key:row['row'] for key,row,_ in observations(BASELINES/'inputs.sqlite',base_plan['input_metadata'],base_plan['input_cases'])}
    cases = []
    for key,record,trace in observations(OUT/'observations.sqlite',dict(plan_sha256=digest(OUT/'plan.json'),stage='online_integration'),plan['integration_schedule']):
        row,kind = record['row'],key[-1]
        repeated = review_online(row,trace,kind,key[4])
        if repeated != record['review']:
            raise ValueError('Independent online trace review differs')
        reference = baselines[(*key[:-2],'baseline')]
        phase,_ = old[key[:-1]]
        cases.append(dict(case=list(key),row=row,review=repeated,baseline_team_cost=reference['team_cost'],
            phase05_team_cost=phase['team_cost'],baseline_failed=not reference['completed'],
            cost_gain_percent=100*(reference['team_cost']-row['team_cost'])/reference['team_cost'],
            phase05_cost_gain_percent=100*(reference['team_cost']-phase['team_cost'])/reference['team_cost'],
            harmful_relative_to_baseline=harmful(row,reference),adopted=repeated['actual_admitted_steps'] > 0))
    cells = []
    for plant in PLANTS:
        for kind in KINDS:
            rows = [r for r in cases if r['case'][4] == plant and r['case'][-1] == kind]
            base = np.mean([r['baseline_team_cost'] for r in rows])
            callbacks = [v for r in rows for v in r['row']['controller_ms']]
            cells.append(dict(plant=plant,model_kind=kind,runs=len(rows),completed=sum(r['row']['completed'] for r in rows),
                adopted=sum(r['adopted'] for r in rows),harmful_and_adopted=sum(r['adopted'] and r['harmful_relative_to_baseline'] for r in rows),
                cost_gain_percent=100*(base-np.mean([r['row']['team_cost'] for r in rows]))/base,
                callback_ms_p95=float(np.quantile(callbacks,.95)),callback_ms_max=float(np.max(callbacks))))
    analysis = OUT/'analysis'
    analysis.mkdir(exist_ok=False)
    report = dict(actual_sumo_runs=120,all_physical_and_online_input_reviews_pass=True,cases=cases,cells=cells,
        source_initials=5,new_independent_initials=0,new_risk_certificate=False,
        scope='Fixed development integration only; 5 old initials shared by all plants/models/variants. These numbers are not a new efficacy confirmation or a generalization claim.',plan_sha256=digest(OUT/'plan.json'))
    exclusive_json(analysis/'report.json',report)
    exclusive_json(analysis/'sha256.json',dict(report=digest(analysis/'report.json')))
    print(json.dumps(dict(actual_sumo_runs=120,cells=cells)),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','kernel','run','summarize'))
    args = parser.parse_args()
    globals()[args.action]()
