"""Complete nominal-admission diagnostics and all observed in-bound regressions.

Old outcomes selected this mechanism study. It is not fresh efficacy data, a new
controller, or risk certification. Original metrics and live studies are frozen.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.episode_store_v21 import EpisodeStore, encode, observations
from coopgate.matched_shift_v21 import MatchedShiftedMergeEnv, audit_matched_trace
from coopgate.phase_repair_v21 import PhaseRepairGate
from coopgate.physical_comparison_analysis_v27 import harmful
from coopgate.regression_diagnostic_v28 import (
    SingleInterventionBaseline, decompose_outcome, observed_crossings, realized_noise_branches)
from coopgate.joint_planner_v15 import JointRolloutGate
from execute_prospective_v23 import exclusive_json, runtime_identity, sources_committed, utc
from experiment_ego_v13 import run_episode
from present_completed_v21 import load_complete
from profile_phase_runtime_v24 import committed, digest
from recover_checkpoint_v19 import atomic_bytes
import recover_interrupted_v21 as original

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/v28/regression_mechanisms'
OLD = ROOT/'artifacts/v21/interrupted_recovery_20261022/results/transfer'
COMPARISON = ROOT/'artifacts/v27/physical_switch_comparison/plan.json'
SOURCES = ('src/coopgate/regression_diagnostic_v28.py', 'tools/diagnose_regression_v28.py',
           'tests/test_regression_diagnostic_v28.py', 'tests/test_regression_diagnostic_protocol_v28.py')
ROLES = ('baseline', 'original_phase05', 'single_action_then_baseline')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def case(row):
    return original.identity('transfer', row)


def select_scope(rows):
    lookup = {case(r):r for r in rows}
    selected = sorted((r for r in rows if r['plant_name'] == 'nominal'
        and (r['controller'] == 'baseline' or r['controller'].startswith('phase05_'))), key=case)
    keys = [case(r) for r in selected]
    expected = {(i, scene, local, seed, 'nominal', name)
        for i, scene, local, seed in original.transfer.cases(read(COMPARISON)['protocols']['transfer']['spec'])
        for name in ('baseline', 'phase05_1802', 'phase05_1812', 'phase05_1822')}
    if len(keys) != 640 or len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('All nominal-plant baseline/phase05 cases are required')
    regressions = [r for r in rows if r['controller'].startswith('phase05_')
        and r['plant_name'] != 'lag_strong' and harmful(r, lookup[(*case(r)[:-1], 'baseline')])]
    if any(r['plant_name'] != 'nominal' for r in regressions):
        raise ValueError('Additional in-bound failure needs an explicit actuator-aware diagnostic')
    if len(regressions) != 4:
        raise ValueError('Previously observed four regressions must all be retained')
    return selected, sorted(regressions, key=case)


def compact_trace(trace):
    required = ('t', 'x', 'command', 'disturbance', 'innovation', 'actuator_states', 'realized_error_in_contract')
    result = []
    for item in trace:
        record = {k:item[k] for k in required}
        admitted = bool(item.get('nominal_joint_action_admitted', False))
        record['nominal_joint_action_admitted'] = admitted
        if admitted:
            record.update({k:item[k] for k in ('baseline_command', 'selected_bank_differences',
                                              'selected_phase_differences', 'repair_weight')})
        result.append(record)
    return result


def freeze():
    load_complete()
    base = read(COMPARISON)
    rows = read(OLD/'episodes.json')
    original.transfer.validate_grid(rows, base['protocols']['transfer']['spec'])
    selected, regressions = select_scope(rows)
    sources = dict(base['source_sha256'])
    # The descriptive harm helper is a newer dependency outside the old simulator.
    sources['src/coopgate/physical_comparison_analysis_v27.py'] = digest(ROOT/'src/coopgate/physical_comparison_analysis_v27.py')
    for name, expected in sources.items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen source changed: '+name)
    sources.update({name:digest(ROOT/name) for name in SOURCES})
    sources_committed([ROOT/name for name in sources])
    OUT.mkdir(parents=True, exist_ok=False)
    provenance = {tuple(p['case']):p for p in read(OLD/'provenance.json')}
    meta = dict(schema='v28_nominal_diagnostic_inputs', old_episodes_sha256=digest(OLD/'episodes.json'))
    store = EpisodeStore(OUT/'inputs.sqlite', meta)
    admissions = []
    source_hashes = {}
    try:
        for row in selected:
            reference = provenance[case(row)]
            path = ROOT/reference['path']
            if digest(path) != reference['sha256']:
                raise ValueError('Original full observation changed')
            trace = compact_trace(read(path))
            steps = [i for i, t in enumerate(trace) if t['nominal_joint_action_admitted']]
            if len(steps) != row['learned_action_steps']:
                raise ValueError('Original adoption count differs')
            admissions.extend([list(case(row))+[step] for step in steps])
            source_hashes[reference['path']] = reference['sha256']
            store.append(case(row), dict(row=row, source_trace=reference), trace)
        store.seal([case(r) for r in selected])
    finally:
        store.close()
    if len(admissions) != 87 or sum(r['learned_action_steps'] > 0 for r in selected) != 81:
        raise ValueError('All 87 existing nominal admissions required')
    if any(r['learned_action_steps'] != 1 for r in regressions):
        raise ValueError('Single-intervention replay requires every selected regression to have one admission')
    plan = dict(purpose=__doc__, utc=utc(), source_sha256=sources, runtime=runtime_identity(),
        comparison_plan_sha256=digest(COMPARISON), models=base['models'], input_metadata=meta,
        inputs_database_sha256=digest(OUT/'inputs.sqlite'), inputs_seal_sha256=digest(OUT/'inputs.sqlite.seal.json'),
        source_full_trace_sha256=source_hashes, input_cases=[list(case(r)) for r in selected],
        diagnostic_cases=[list(case(r)) for r in selected if r['controller'] != 'baseline'],
        admission_cases=admissions, regression_cases=[list(case(r)) for r in regressions],
        replay_roles=list(ROLES), expected_inputs=640, nominal_phase05_episodes=480,
        adopted_episodes=81, admitted_steps=87, actual_sumo_replays=12,
        selected_regressions=4, distinct_selected_initials=2,
        scope='All 480 nominal-plant phase05 episodes, their 160 baseline episodes, and every admitted decision. All four observed in-bound regressions replayed with baseline, original gate, and one archived action followed by feedback baseline.',
        selection='Post-hoc complete failure-mechanism study. Three model failures share the same low-speed-yield initial and command; not independent failures.',
        oracle='Known realized external-noise future used only offline, never exposed to the control policy. Additive counterfactuals restricted to the nominal actuator; mismatched effective error is command-dependent.',
        metrics='Retain original decision-grid finish and cost. Substep-linear interpolated crossing and rounding decomposition are diagnostic additions only.',
        failures='Preserve full prefixes and errors; no skipped selected failure, silent resume, replacement seed, or retuning.',
        new_independent_initials=0, new_risk_certificate=False, controller_or_threshold_changed=False,
        active_calibration_outcomes_read=False)
    exclusive_json(OUT/'plan.json', plan)
    print(json.dumps(dict(plan_sha256=digest(OUT/'plan.json'), archived_inputs=640,
                         offline_admissions=87, actual_sumo_replays=12)), flush=True)


def inputs():
    committed(OUT/'plan.json')
    plan = read(OUT/'plan.json')
    if (runtime_identity() != plan['runtime'] or digest(COMPARISON) != plan['comparison_plan_sha256']
            or digest(OUT/'inputs.sqlite') != plan['inputs_database_sha256']
            or digest(OUT/'inputs.sqlite.seal.json') != plan['inputs_seal_sha256']):
        raise ValueError('Frozen diagnostic runtime/input differs')
    for name, expected in plan['source_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen source differs: '+name)
    for model in plan['models'].values():
        if digest(ROOT/model['path']) != model['sha256']:
            raise ValueError('Frozen trained actor differs')
    sources_committed([ROOT/name for name in plan['source_sha256']])
    return plan


def metadata(stage):
    return dict(plan_sha256=digest(OUT/'plan.json'), stage=stage)


def paired_replay_check(row, trace, expected, reference_trace):
    audit_matched_trace(trace, row)
    for key in ('seed', 'scenario', 'initial', 'plant', 'disturbance', 'radius', 'innovation_tape_sha256',
                'external_innovation_radius', 'physical_error_radius', 'per_vehicle_finish',
                'per_vehicle_variation', 'per_vehicle_cost', 'completed', 'aborted', 'collision', 'contract_violations'):
        np.testing.assert_equal(row[key], expected[key])
    if len(trace) != len(reference_trace):
        raise ValueError('Replayed trajectory has different length')
    for current, archived in zip(trace, reference_trace, strict=True):
        np.testing.assert_array_equal(current['command'], archived['command'])
        np.testing.assert_allclose(current['x'], archived['x'], atol=1e-8, rtol=0)
    return dict(physical_trace_checked=True, original_metrics_preserved=True,
                executed_commands_exact=True, reconstructed_steps=len(trace))


def oracle_replay_check(diagnostic, decomposition):
    if not diagnostic['both_branches_complete']:
        raise ValueError('Selected regression oracle branch did not complete')
    difference = np.asarray(diagnostic['realized_difference'])
    if difference.shape != (3, 2):
        raise ValueError('Complete per-vehicle counterfactual costs and times required')
    np.testing.assert_allclose(difference[:, 0], decomposition['original_cost_delta'], atol=1e-9, rtol=0)
    np.testing.assert_allclose(difference[:, 1], decomposition['logged_finish_delta'], atol=1e-9, rtol=0)
    np.testing.assert_allclose(diagnostic['realized_phase_delta'], decomposition['interpolated_crossing_delta'], atol=1e-7, rtol=0)
    return dict(original_cost_and_time_matched=True, observed_crossing_delta_matched=True)


def run():
    plan = inputs()
    execution = OUT/'execution'
    execution.mkdir(exist_ok=False)
    if os.name == 'nt':
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        kernel.SetPriorityClass.restype = ctypes.c_int
        if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), subprocess.BELOW_NORMAL_PRIORITY_CLASS):
            raise ctypes.WinError(ctypes.get_last_error())
    exclusive_json(execution/'start.json', dict(native_pid=os.getpid(), parent_pid=os.getppid(), utc=utc(),
        plan_sha256=digest(OUT/'plan.json'), priority='below_normal' if os.name == 'nt' else 'default'))
    torch.set_num_threads(1)
    records = {key:(record['row'], trace) for key, record, trace in observations(
        OUT/'inputs.sqlite', plan['input_metadata'], plan['input_cases'])}
    baselines = {}
    for key, (row, trace) in records.items():
        audit_matched_trace(trace, row)
        if row['controller'] == 'baseline':
            baselines[key] = observed_crossings(row['initial'], trace)
    offline = OUT/'offline'
    offline.mkdir()
    store = EpisodeStore(offline/'observations.sqlite', metadata('offline'))
    admitted = 0
    selected_oracles = {}
    try:
        for index, raw_case in enumerate(plan['diagnostic_cases']):
            key = tuple(raw_case)
            row, trace = records[key]
            base_key = (*key[:-1], 'baseline')
            baseline, _ = records[base_key]
            crossing = observed_crossings(row['initial'], trace)
            decomposition = decompose_outcome(row, crossing, baseline, baselines[base_key])
            results = [realized_noise_branches(row, trace, step) for step, item in enumerate(trace)
                       if item['nominal_joint_action_admitted']]
            record = dict(source_case=raw_case, source_control_steps=len(trace),
                original_harmful=harmful(row, baseline), original_adopted=row['learned_action_steps'] > 0,
                original_costs=row['per_vehicle_cost'], baseline_costs=baseline['per_vehicle_cost'],
                decomposition=decomposition, local_admissions=len(results))
            store.append(key, record, results)
            if raw_case in plan['regression_cases']:
                selected_oracles[key] = (results, decomposition)
            admitted += len(results)
            atomic_bytes(execution/'progress.json', encode(dict(phase='offline', completed_episodes=index+1,
                expected_episodes=480, completed_admissions=admitted, expected_admissions=87, utc=utc(), native_pid=os.getpid())))
        store.seal(plan['diagnostic_cases'])
    finally:
        store.close()
    if admitted != plan['admitted_steps']:
        raise ValueError('An existing admission was lost')
    exclusive_json(offline/'complete.json', dict(episodes=480, admissions=admitted,
        database_sha256=digest(offline/'observations.sqlite'), seal_sha256=digest(offline/'observations.sqlite.seal.json')))
    models = {name:load_coupled(ROOT/value['path'])[0] for name, value in plan['models'].items()}
    replay = OUT/'replays'
    replay.mkdir()
    store = EpisodeStore(replay/'observations.sqlite', metadata('sumo_replays'))
    schedule = [key+[role] for key in plan['regression_cases'] for role in plan['replay_roles']]
    reviews, oracle_checks, executed = [], [], 0
    try:
        with MatchedShiftedMergeEnv(plant='nominal') as env:
            for raw_case in plan['regression_cases']:
                key = tuple(raw_case)
                source, source_trace = records[key]
                base, base_trace = records[(*key[:-1], 'baseline')]
                steps = [i for i, item in enumerate(source_trace) if item['nominal_joint_action_admitted']]
                if len(steps) != 1:
                    raise ValueError('Every selected regression must retain its single original admission')
                step = steps[0]
                # Before the sole learning action, the entire observed prefix
                # must be the same full-baseline prefix, not just a similar state.
                for left, right in zip(source_trace[:step], base_trace[:step], strict=True):
                    np.testing.assert_equal(left['command'], right['command'])
                    np.testing.assert_equal(left['x'], right['x'])
                state = source['initial'] if step == 0 else source_trace[step-1]['x']
                previous = np.zeros(3) if step == 0 else source_trace[step-1]['command']
                factories = dict(baseline=lambda _:JointRolloutGate(),
                    original_phase05=lambda _, m=models[source['controller'].rsplit('_', 1)[1]]:PhaseRepairGate(m, .05),
                    single_action_then_baseline=lambda _:SingleInterventionBaseline(step, state, previous, source_trace[step]['command']))
                for role in plan['replay_roles']:
                    row, trace, initial = run_episode(env, source['seed'], source['scenario'], source['disturbance'], factories[role], True)
                    row.update(initial=initial.tolist(), plant_name='nominal', initial_index=source['initial_index'],
                        global_initial_index=source['global_initial_index'], controller=source['controller']+'_'+role,
                        control_steps=len(trace), learned_action_steps=sum(bool(t.get('nominal_joint_action_admitted')) for t in trace))
                    costs = np.asarray(row['per_vehicle_finish'])+.05*np.asarray(row['per_vehicle_variation'])
                    row.update(per_vehicle_cost=costs.tolist(), team_cost=float(costs.mean()))
                    store.append(raw_case+[role], dict(row=row, source_case=raw_case, role=role), trace)
                    expected, expected_trace = (base, base_trace) if role == 'baseline' else (source, source_trace)
                    check = paired_replay_check(row, trace, expected, expected_trace)
                    if row['learned_action_steps'] != (0 if role == 'baseline' else 1):
                        raise ValueError('Diagnostic execution changed the number of learning interventions')
                    reviews.append(dict(source_case=raw_case, role=role, **check))
                    executed += 1
                    atomic_bytes(execution/'progress.json', encode(dict(phase='sumo_replay', completed_runs=executed,
                        expected_runs=12, native_pid=os.getpid(), utc=utc())))
                    print(json.dumps(dict(completed_sumo_replays=executed, expected_sumo_replays=12)), flush=True)
                local, decomposition = selected_oracles[key]
                if len(local) != 1:
                    raise ValueError('Single-admission causal reconstruction required')
                oracle_checks.append(dict(source_case=raw_case, **oracle_replay_check(local[0], decomposition)))
        store.seal(schedule)
    finally:
        store.close()
    exclusive_json(replay/'complete.json', dict(actual_sumo_runs=executed, reviews=reviews, oracle_checks=oracle_checks,
        database_sha256=digest(replay/'observations.sqlite'), seal_sha256=digest(replay/'observations.sqlite.seal.json')))
    if executed != 12:
        raise ValueError('All selected counterfactual replays required')
    exclusive_json(execution/'collection_complete.json', dict(native_pid=os.getpid(), utc=utc(),
        plan_sha256=digest(OUT/'plan.json'), offline_episodes=480, offline_admissions=87, actual_sumo_runs=12))


def completed(plan):
    receipt, start, done = (read(OUT/'execution'/name) for name in ('observation.json', 'start.json', 'collection_complete.json'))
    sha = digest(OUT/'plan.json')
    if (receipt.get('exit_code') != 0 or receipt.get('native_pid') != start['native_pid']
            or done['native_pid'] != start['native_pid'] or any(r['plan_sha256'] != sha for r in (receipt, start, done))
            or done['offline_episodes'] != 480 or done['offline_admissions'] != 87 or done['actual_sumo_runs'] != 12):
        raise ValueError('Complete diagnostic and actual successful native exit required')
    records = []
    for folder, stage, schedule in [('offline', 'offline', plan['diagnostic_cases']),
        ('replays', 'sumo_replays', [key+[role] for key in plan['regression_cases'] for role in plan['replay_roles']])]:
        directory = OUT/folder
        check = read(directory/'complete.json')
        if (digest(directory/'observations.sqlite') != check['database_sha256']
                or digest(directory/'observations.sqlite.seal.json') != check['seal_sha256']):
            raise ValueError('Completed diagnostic block changed')
        data = list(observations(directory/'observations.sqlite', metadata(stage), schedule))
        if folder == 'replays':
            if len(check['reviews']) != 12 or check['actual_sumo_runs'] != 12:
                raise ValueError('All real replay reviews required')
            if ([c['source_case'] for c in check['oracle_checks']] != plan['regression_cases']
                    or not all(c['original_cost_and_time_matched'] and c['observed_crossing_delta_matched'] for c in check['oracle_checks'])):
                raise ValueError('All selected oracle/SUMO reconciliations required')
            for (key, record, trace), review in zip(data, check['reviews'], strict=True):
                audit_matched_trace(trace, record['row'])
                if (review['source_case'] != list(key[:-1]) or review['role'] != key[-1]
                        or not all(review[k] for k in ('physical_trace_checked', 'original_metrics_preserved', 'executed_commands_exact'))
                        or review['reconstructed_steps'] != len(trace)):
                    raise ValueError('Replay verification scope differs')
        records.append(data)
    return records


def summarize():
    plan = inputs()
    offline, replays = completed(plan)
    local = [dict(case=list(key), **item) for key, _, trace in offline for item in trace]
    if len(offline) != 480 or len(local) != 87:
        raise ValueError('Complete nominal admission scope required')
    selected = [dict(case=list(key), **record, local_diagnostics=trace)
                for key, record, trace in offline if list(key) in plan['regression_cases']]
    report = dict(plan_sha256=digest(OUT/'plan.json'), utc=utc(),
        actual_exit_sha256=digest(OUT/'execution/observation.json'), nominal_episodes=480,
        adopted_episodes=sum(record['original_adopted'] for _, record, _ in offline), admitted_steps=87,
        local_complete=sum(item['both_branches_complete'] for item in local),
        local_regressions=sum(item['realized_local_regression'] is True for item in local),
        local_incomplete=sum(not item['both_branches_complete'] for item in local),
        all_local_diagnostics=local, selected_regressions=selected,
        actual_sumo_replays=len(replays), replayed_commands=sum(len(trace) for _, _, trace in replays),
        distinct_selected_initials=2, original_metric_definition_changed=False,
        new_independent_initials=0, new_risk_certificate=False, new_controller=False,
        active_calibration_outcomes_read=False,
        scope='Post-hoc diagnostics on complete old nominal-plant admissions and all observed in-bound regressions. Future-noise branch forecasts are explanatory only; repeated models and replays add no independent evidence.')
    output = OUT/'analysis'
    output.mkdir(exist_ok=False)
    exclusive_json(output/'report.json', report)
    exclusive_json(output/'sha256.json', {p.name:digest(p) for p in output.iterdir() if p.is_file()})
    print(json.dumps({k:report[k] for k in ('nominal_episodes', 'admitted_steps', 'local_complete', 'local_regressions',
                                          'local_incomplete', 'actual_sumo_replays', 'replayed_commands')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze', 'run', 'summarize'))
    args = parser.parse_args()
    {'freeze':freeze, 'run':run, 'summarize':summarize}[args.action]()
