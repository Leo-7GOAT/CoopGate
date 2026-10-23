"""Validate every archived phase05 intervention with two full SUMO continuations.

All five scenes, four plants and three fixed models are retained. This is a
post-hoc counterfactual mechanism study, not a fresh efficacy test or a new gate.
"""
import argparse
from collections import Counter
import ctypes
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from coopgate.actuator_intervention_v31 import (
    ROLES, RecordedPrefixBranch, compare_branch, intervention_state, pair_effect)
from coopgate.actuator_prediction_v30 import predict_suffix
from coopgate.disturbance_v9 import disturbance_tape
from coopgate.episode_store_v21 import EpisodeStore, encode, observations
from coopgate.matched_shift_v21 import MatchedShiftedMergeEnv, audit_matched_trace, tape_fingerprint
from coopgate.shifted_sumo_v19 import PLANTS, SCENARIOS
from diagnose_regression_v28 import compact_trace
from execute_prospective_v23 import exclusive_json, runtime_identity, sources_committed, utc
from experiment_ego_v13 import run_episode
from present_completed_v21 import load_complete
from profile_phase_runtime_v24 import committed, digest
from recover_checkpoint_v19 import atomic_bytes
import recover_interrupted_v21 as original

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/v31/actuator_interventions'
OLD = ROOT/'artifacts/v21/interrupted_recovery_20261022/results/transfer'
BASE = ROOT/'artifacts/v30/actuator_prediction_audit/plan.json'
COMPARISON = ROOT/'artifacts/v27/physical_switch_comparison/plan.json'
SOURCES = ('src/coopgate/actuator_intervention_v31.py', 'tools/audit_interventions_v31.py',
           'tests/test_actuator_interventions_v31.py', 'tests/test_intervention_scope_v31.py')
MODELS = ('phase05_1802', 'phase05_1812', 'phase05_1822')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def case(row):
    return original.identity('transfer', row)


def scope(rows):
    selected = sorted((r for r in rows if r['controller'] in MODELS), key=case)
    expected = Counter({(s,p,m):32 for s in SCENARIOS for p in PLANTS for m in MODELS})
    counts = Counter((r['scenario'], r['plant_name'], r['controller']) for r in selected)
    if (len(selected) != 1920 or len({case(r) for r in selected}) != 1920 or counts != expected
            or any({r['initial_index'] for r in selected if
                    (r['scenario'],r['plant_name'],r['controller']) == cell} != set(range(32)) for cell in expected)):
        raise ValueError('All 1920 original phase05 episodes in all 60 cells are required')
    return selected


def freeze():
    load_complete()
    base, comparison = read(BASE), read(COMPARISON)
    rows = read(OLD/'episodes.json')
    original.transfer.validate_grid(rows, comparison['protocols']['transfer']['spec'])
    selected = scope(rows)
    sources = dict(base['source_sha256'])
    sources.update({name:digest(ROOT/name) for name in SOURCES})
    for name, expected in sources.items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen source changed: '+name)
    sources_committed([ROOT/name for name in sources])
    provenance = {tuple(p['case']):p for p in read(OLD/'provenance.json')}
    OUT.mkdir(parents=True, exist_ok=False)
    meta = dict(schema='v31_all_phase05_intervention_inputs', source_episodes_sha256=digest(OLD/'episodes.json'))
    store = EpisodeStore(OUT/'inputs.sqlite', meta)
    admissions = []
    try:
        for row in selected:
            reference = provenance[case(row)]
            path = ROOT/reference['path']
            if digest(path) != reference['sha256']:
                raise ValueError('Original trace bytes changed')
            trace = compact_trace(read(path))
            steps = [i for i,t in enumerate(trace) if t['nominal_joint_action_admitted']]
            if len(steps) != row['learned_action_steps']:
                raise ValueError('An archived learning admission was lost')
            for step in steps:
                intervention_state(row,trace,step)
                admissions.append(list(case(row))+[step])
            store.append(case(row),dict(row=row,original_trace=reference),trace)
        store.seal([case(r) for r in selected])
    finally:
        store.close()
    if len(admissions) != 349 or sum(r['learned_action_steps'] > 0 for r in selected) != 333:
        raise ValueError('All 349 admissions from all 333 adopting episodes are required')
    schedule = [key for plant in PLANTS for key in admissions if key[4] == plant]
    exclusive_json(OUT/'plan.json', dict(utc=utc(),source_sha256=sources,runtime=runtime_identity(),
        input_metadata=meta,input_database_sha256=digest(OUT/'inputs.sqlite'),
        input_seal_sha256=digest(OUT/'inputs.sqlite.seal.json'),input_cases=[list(case(r)) for r in selected],
        admission_cases=schedule,replay_roles=list(ROLES),input_episodes=1920,adopted_episodes=333,
        admitted_steps=349,actual_sumo_runs=698,
        parent_plans={str(p.relative_to(ROOT)):digest(p) for p in (BASE,COMPARISON)},
        scope='Every admitted phase05 action in the completed five-scene/four-plant/three-model transfer grid. No success or failure filtering.',
        execution='Both roles physically replay the entire archived prefix, including prior learning actions. One role then takes the archived candidate once; the other immediately uses baseline. Both continue with the identical feedback baseline on the original clock.',
        oracle='Known true actuator parameters and realized future external innovations are used only in offline forecasts. The SUMO controllers receive no future innovation or realized command error. Each branch evolves its own actuator memory.',
        comparator='A nominal-actuator forecast at the identical visited state and identical external path diagnoses model mismatch; it is not a new online selector.',
        outcomes='Retain original per-vehicle decision-grid arrival and cost. Report complete comparisons and all incomplete branches separately; capped failures never authorize performance improvement.',
        tolerances=dict(state=1e-8,metrics=1e-9,memory_and_errors=1e-9,crossing=1e-8),
        new_independent_initials=0,new_controller=False,new_risk_certificate=False,
        active_calibration_outcomes_read=False))
    print(json.dumps(dict(input_episodes=1920,admissions=349,actual_sumo_runs=698,plan_sha256=digest(OUT/'plan.json'))),flush=True)


def inputs():
    committed(OUT/'plan.json')
    plan = read(OUT/'plan.json')
    if (runtime_identity() != plan['runtime'] or digest(OUT/'inputs.sqlite') != plan['input_database_sha256']
            or digest(OUT/'inputs.sqlite.seal.json') != plan['input_seal_sha256']):
        raise ValueError('Frozen runtime or inputs changed')
    for name,expected in {**plan['source_sha256'],**plan['parent_plans']}.items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen dependency changed: '+name)
    sources_committed([ROOT/name for name in plan['source_sha256']])
    return plan


def metadata(stage):
    return dict(plan_sha256=digest(OUT/'plan.json'),stage=stage)


def predictions(row, trace, step, plant, keep_trace):
    state, previous, memory = intervention_state(row,trace,step)
    tape = disturbance_tape(row['seed'],row['disturbance'],row['external_innovation_radius'],250)
    if tape_fingerprint(tape) != row['innovation_tape_sha256']:
        raise ValueError('Original innovation path identity differs')
    return {role:predict_suffix(state,previous,memory,step,tape,plant,
                first_command=np.asarray(trace[step]['command']) if role == 'candidate_then_baseline' else None,
                record_trace=keep_trace) for role in ROLES}


def run():
    plan = inputs()
    execution = OUT/'execution'
    execution.mkdir(exist_ok=False)
    if os.name == 'nt':
        kernel = ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetPriorityClass.argtypes = (ctypes.c_void_p,ctypes.c_uint32)
        kernel.SetPriorityClass.restype = ctypes.c_int
        if not kernel.SetPriorityClass(kernel.GetCurrentProcess(),subprocess.BELOW_NORMAL_PRIORITY_CLASS):
            raise ctypes.WinError(ctypes.get_last_error())
    torch.set_num_threads(1)
    exclusive_json(execution/'start.json',dict(native_pid=os.getpid(),parent_pid=os.getppid(),utc=utc(),
        plan_sha256=digest(OUT/'plan.json')))
    records = {key:(record['row'],trace) for key,record,trace in observations(
        OUT/'inputs.sqlite',plan['input_metadata'],plan['input_cases'])}
    for stage in ('offline','replays'):
        (OUT/stage).mkdir()
    offline = EpisodeStore(OUT/'offline/observations.sqlite',metadata('offline'))
    replay = EpisodeStore(OUT/'replays/observations.sqlite',metadata('replays'))
    checked, count = set(), 0
    try:
        for plant in PLANTS:
            with MatchedShiftedMergeEnv(plant=plant) as env:
                for key in (k for k in plan['admission_cases'] if k[4] == plant):
                    source_key, step = tuple(key[:-1]), key[-1]
                    source, source_trace = records[source_key]
                    if source_key not in checked:
                        audit_matched_trace(source_trace,source)
                        checked.add(source_key)
                    actual = predictions(source,source_trace,step,source['plant'],True)
                    nominal = predictions(source,source_trace,step,dict(gain=[1.]*3,drag=[0.]*3,tau=[0.]*3),False)
                    pair, nominal_pair = pair_effect(actual), pair_effect(nominal)
                    predicted_traces = {role:actual[role].pop('trace') for role in ROLES}
                    offline.append(key,dict(branches=actual,pair=pair,nominal_pair=nominal_pair,
                        nominal_branches=nominal,original_bank_maximum=np.asarray(source_trace[step]['selected_bank_differences']).max(axis=0).tolist(),
                        original_episode_aborted=source['aborted'],original_episode_complete=source['completed']),predicted_traces)
                    for role in ROLES:
                        factory = lambda _, role=role:RecordedPrefixBranch(source,source_trace,step,role)
                        row, trace, initial = run_episode(env,source['seed'],source['scenario'],source['disturbance'],factory,True)
                        costs = np.asarray(row['per_vehicle_finish'])+.05*np.asarray(row['per_vehicle_variation'])
                        row.update(initial=initial.tolist(),plant_name=plant,controller=role,
                            per_vehicle_cost=costs.tolist(),team_cost=float(costs.mean()))
                        audit_matched_trace(trace,row)
                        review = compare_branch(dict(actual[role],trace=predicted_traces[role]),row,trace,source,source_trace,step)
                        replay.append(key+[role],dict(row=row,comparison=review),trace)
                        count += 1
                    atomic_bytes(execution/'progress.json',encode(dict(native_pid=os.getpid(),utc=utc(),
                        completed_admissions=count//2,expected_admissions=349,actual_sumo_runs=count,expected_sumo_runs=698)))
        offline.seal(plan['admission_cases'])
        replay.seal([key+[role] for key in plan['admission_cases'] for role in ROLES])
    finally:
        offline.close()
        replay.close()
    for stage in ('offline','replays'):
        folder = OUT/stage
        exclusive_json(folder/'complete.json',dict(database_sha256=digest(folder/'observations.sqlite'),
            seal_sha256=digest(folder/'observations.sqlite.seal.json'),cases=349 if stage == 'offline' else count))
    exclusive_json(execution/'collection_complete.json',dict(native_pid=os.getpid(),utc=utc(),
        plan_sha256=digest(OUT/'plan.json'),admissions=349,actual_sumo_runs=count,source_episodes_checked=len(checked)))
    print(json.dumps(dict(admissions=349,actual_sumo_runs=count,source_episodes_checked=len(checked))),flush=True)


def summarize():
    plan = inputs()
    start, done, exit = [read(OUT/'execution'/name) for name in ('start.json','collection_complete.json','observation.json')]
    if (exit['exit_code'] != 0 or done['admissions'] != 349 or done['actual_sumo_runs'] != 698
            or done['source_episodes_checked'] != 333 or exit['native_pid'] != start['native_pid']
            or done['native_pid'] != start['native_pid']
            or any(r['plan_sha256'] != digest(OUT/'plan.json') for r in (start,done,exit))):
        raise ValueError('The complete frozen scope and actual successful exit are required')
    schedules = dict(offline=plan['admission_cases'],replays=[k+[r] for k in plan['admission_cases'] for r in ROLES])
    data = {}
    for stage,schedule in schedules.items():
        directory = OUT/stage
        receipt = read(directory/'complete.json')
        if (receipt['cases'] != len(schedule) or digest(directory/'observations.sqlite') != receipt['database_sha256']
                or digest(directory/'observations.sqlite.seal.json') != receipt['seal_sha256']):
            raise ValueError('Sealed complete observations changed')
        data[stage] = {key:(row,trace) for key,row,trace in observations(directory/'observations.sqlite',metadata(stage),schedule)}
    original_inputs = {key:(row['row'],trace) for key,row,trace in observations(
        OUT/'inputs.sqlite',plan['input_metadata'],plan['input_cases'])}
    groups = {(s,p,m):dict(admissions=0,complete_pairs=0,local_regressions=0,baseline_incomplete=0,
        candidate_incomplete=0,nominal_regressions=0,nominal_incomplete=0,classification_changes=0)
        for s in SCENARIOS for p in PLANTS for m in MODELS}
    mismatches, cases, commands = [], [], 0
    for raw_key in plan['admission_cases']:
        key = tuple(raw_key)
        record, predicted = data['offline'][key]
        source, source_trace = original_inputs[key[:-1]]
        repeated = pair_effect(record['branches'])
        if repeated != record['pair'] or pair_effect(record['nominal_branches']) != record['nominal_pair']:
            raise ValueError('Pair accounting differs')
        role_rows = {}
        for role in ROLES:
            result, trace = data['replays'][(*key,role)]
            audit_matched_trace(trace,result['row'])
            review = compare_branch(dict(record['branches'][role],trace=predicted[role]),
                result['row'],trace,source,source_trace,key[-1])
            if review != result['comparison']:
                raise ValueError('Recomputed forecast comparison differs')
            if not review['all_match']:
                mismatches.append(dict(case=list(key),role=role,comparison=review))
            commands += len(trace)
            role_rows[role] = result['row']
        actual_delta = None
        if record['pair']['both_complete']:
            actual_delta = np.column_stack((role_rows[ROLES[1]]['per_vehicle_cost'],role_rows[ROLES[1]]['per_vehicle_finish']))-np.column_stack((role_rows[ROLES[0]]['per_vehicle_cost'],role_rows[ROLES[0]]['per_vehicle_finish']))
            if not np.allclose(actual_delta,record['pair']['difference'],atol=1e-9,rtol=0):
                mismatches.append(dict(case=list(key),role='paired_difference',
                    actual=actual_delta.tolist(),predicted=record['pair']['difference']))
        a,b = record['pair'],record['nominal_pair']
        changed = a['admissible_performance_evidence'] != b['admissible_performance_evidence']
        group = groups[(key[1],key[4],key[5])]
        for field,increment in dict(admissions=1,complete_pairs=a['both_complete'],
            local_regressions=a['local_regression'] is True,baseline_incomplete=not a['baseline_complete'],
            candidate_incomplete=not a['candidate_complete'],nominal_regressions=b['local_regression'] is True,
            nominal_incomplete=not b['both_complete'],classification_changes=changed).items():
            group[field] += int(increment)
        cases.append(dict(case=raw_key,actual_model_pair=a,nominal_model_pair=b,
            model_changes_performance_admissibility=changed,
            actual_pair_difference=None if actual_delta is None else actual_delta.tolist(),
            recorded_episode_aborted=record['original_episode_aborted']))
    by_plant = {p:{field:sum(g[field] for (s,plant,m),g in groups.items() if plant == p)
                  for field in next(iter(groups.values()))} for p in PLANTS}
    report = dict(admissions=349,actual_sumo_runs=698,source_episodes=1920,adopting_source_episodes=333,
        replayed_commands=commands,mismatched_branches=mismatches,all_forecasts_match=not mismatches,
        cells=[dict(scenario=s,plant=p,model=m,**g) for (s,p,m),g in groups.items()],
        by_plant=by_plant,cases=cases,new_independent_initials=0,new_controller=False,new_risk_certificate=False,
        interpretation='Local post-hoc interventions share old initials, models and prefixes. Known future innovation/true parameters diagnose model errors only; no deployment claim or population conditional-risk estimate follows.',
        inputs_plan_sha256=digest(OUT/'plan.json'),input_complete_receipts={stage:digest(OUT/stage/'complete.json') for stage in schedules})
    analysis = OUT/'analysis'
    analysis.mkdir(exist_ok=False)
    exclusive_json(analysis/'report.json',report)
    exclusive_json(analysis/'sha256.json',dict(report=digest(analysis/'report.json')))
    print(json.dumps(dict(all_forecasts_match=not mismatches,mismatched_branches=len(mismatches),
        replayed_commands=commands,by_plant=by_plant)),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','run','summarize'))
    args = parser.parse_args()
    globals()[args.action]()
