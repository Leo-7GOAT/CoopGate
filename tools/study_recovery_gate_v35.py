"""Frozen paired SUMO integration of MARL admission and actuator-aware recovery."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.episode_store_v21 import EpisodeStore, encode, observations
from coopgate.matched_shift_v21 import MatchedShiftedMergeEnv
from coopgate.physical_comparison_analysis_v27 import equal_fields
from coopgate.recovery_gate_audit_v35 import cell_summary, relative_outcome, review_joint_trace
from coopgate.recovery_joint_gate_v35 import RecoveryJointGate, RULES
import audit_buffered_scope_v25 as infrastructure
import audit_recovery_forecast_v35 as predictor
import study_actuator_recovery_v34 as recovery
from execute_prospective_v23 import check_acknowledgment, exclusive_json, runtime_identity, sources_committed, utc
from experiment_ego_v13 import run_episode
from profile_phase_runtime_v24 import committed, digest
from recover_checkpoint_v19 import atomic_bytes

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/v35/recovery_gate_integration'
SCRIPT = 'tools/study_recovery_gate_v35.py'
MODEL_PLAN = ROOT/'artifacts/v32/online_actuator_preflight/plan.json'
MODELS = ('1802','1812','1822')
SOURCES = ('src/coopgate/recovery_joint_gate_v35.py','src/coopgate/recovery_gate_audit_v35.py',
           SCRIPT,'tests/test_recovery_joint_gate_v35.py','tests/test_recovery_gate_audit_v35.py',
           'tests/test_recovery_gate_study_v35.py')
PHYSICAL_FIELDS = ('t','x','command','innovation','actuator_states','disturbance')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def integration_groups(cases, failures):
    # Validate the complete source grid before selecting coverage, including
    # the ten documented failures. No new outcome is inspected for selection.
    if (any(len(k) != 6 or k[-1] != 'actuator_recovery' for k in cases)
            or any(len(k) != 6 or k[-1] != 'baseline' for k in failures)):
        raise ValueError('Original full recovery grid and baseline failure identities required')
    normalized = [list(k[:5])+['baseline'] for k in cases]
    recovery.complete_cases(normalized)
    conditions = {tuple(k[:5]) for k in cases}
    failed = {tuple(k[:5]) for k in failures}
    if len(failures) != 10 or len(failed) != 10 or not failed <= conditions:
        raise ValueError('All ten distinct original failed conditions are required')
    coverage = {min((k for k in conditions if k[1] == s and k[4] == p),key=lambda k:k[2])
                for s in recovery.SCENES for p in recovery.PROFILES}
    selected = sorted(coverage | failed, key=lambda k:(k[2],k[1],k[4],k[0]))
    if len(coverage) != 20 or len(selected) != 30:
        raise ValueError('Expected 20 coverage conditions plus ten old failure conditions')
    methods = [('baseline','none')]+[(rule,model) for model in MODELS for rule in RULES[1:]]
    result = []
    for index, condition in enumerate(selected):
        offset = index % len(methods)
        ordered = methods[offset:]+methods[:offset]
        if index % 2:
            ordered = ordered[::-1]
        result.append(dict(block_index=index,condition=list(condition),old_baseline_failed=condition in failed,
            cases=[list(condition)+list(method) for method in ordered]))
    return result


def archived_recovery():
    plan = read(recovery.OUT/'plan.json')
    return {tuple(k[:5]):(record['row'],trace) for k,record,trace in observations(
        recovery.OUT/'observations.sqlite',recovery.metadata(),plan['cases'])}


def freeze():
    prior = predictor.inputs()
    report = read(predictor.OUT/'report.json')
    proof = read(predictor.OUT/'completion_verification.json')
    if (not proof['all_match'] or proof['episodes'] != 640 or proof['commands'] != 15032
            or proof['actual_exit_code'] != 0 or proof['report_sha256'] != digest(predictor.OUT/'report.json')
            or not report['all_match']):
        raise ValueError('Complete recovery predictor audit required before integration')
    recovery_plan = read(recovery.OUT/'plan.json')
    groups = integration_groups(recovery_plan['cases'],recovery_plan['baseline_failed_cases'])
    sources = dict(prior['source_sha256'])
    sources.update({name:digest(ROOT/name) for name in SOURCES})
    for name, expected in sources.items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen dependency changed: '+name)
    sources_committed([ROOT/name for name in sources])
    models = read(MODEL_PLAN)['models']
    if tuple(models) != MODELS:
        raise ValueError('All three existing trained actor sets required')
    for item in models.values():
        if digest(ROOT/item['path']) != item['sha256']:
            raise ValueError('Frozen learned model changed')
    files = [predictor.OUT/n for n in ('plan.json','report.json','completion_verification.json')]
    files += [MODEL_PLAN,recovery.OUT/'plan.json',recovery.OUT/'observations.sqlite',
              recovery.OUT/'observations.sqlite.seal.json']
    OUT.mkdir(parents=True,exist_ok=False)
    exclusive_json(OUT/'plan.json',dict(utc=utc(),runtime=runtime_identity(),source_sha256=sources,
        inputs_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in files},models=models,
        groups=groups,blocks=len(groups),actual_sumo_runs=sum(len(g['cases']) for g in groups),
        methods=list(RULES),old_unique_initials=len({g['condition'][0] for g in groups}),
        new_independent_initials=0,new_risk_certificate=False,
        selection='First original initial in each of the 20 scene/plant cells, plus every one of the ten archived nominal-baseline failures. All 30 conditions and all three fixed model sets; no filtering by new results.',
        method='All methods use the identical V34 recovery baseline, online command-history interval, known actuator-class bounds and stopping certificate. Performance variants share all candidate compositions, projection, reference parameters and ranking; only one versus all 33 fixed prediction paths differs.',
        prior_adapters='MPS/MAMPS-inspired shared-oracle adapters only. No reproduction of their original optimizer, deterministic benchmark or guarantees. Their candidate/compute mechanisms differ from the performance gates.',
        learned_policy='Three actual independently parameterized interacting actors per fixed model set, trained by model-based multi-agent approximate policy iteration and data aggregation. No new MAPPO training or equilibrium claim.',
        online_information='Observed kinematics, executed previous command, decision count, declared actuator-class bounds and fixed trained actor outputs. No realized randomized plant parameters, true actuator memory, seed or future innovations.',
        prediction='One clean path or all 33 fixed paths, radius .1, with class-midpoint reference dynamics. Candidate-first and baseline-only both continue the same full actuator-recovery baseline. Incomplete predictions cannot authorize learning.',
        performance_rule='Every vehicle cost and registered arrival difference <=1e-10, strict nominal mean cost gain >1e-9, interpolated arrival difference <=.05+1e-10. A finite-path margin is not a probability or statistical certificate.',
        adoption='At least one active vehicle actually executes a command differing from the common baseline by >1e-9. Cleared vehicle commands are pinned to baseline. Same event for every method.',
        harm='Controller failure or any vehicle cost/arrival above paired baseline by >1e-9. Baseline failure makes relative harm unassessable. Report conditional descriptive harm among adopted episodes; zero adoption denominator is null.',
        execution='One native worker with acknowledged identity and an actual parent-held exit handle. Same source initial, true plant and external innovation tape for all 13 methods. Fixed rotating/reversing method order per condition. All rules, models and plants warmed on one declared state before timing.',
        review='Persist each actual record before any check. Independently reconstruct every physical substep and baseline command during collection. After complete actual successful exit, replay every full gate decision/evidence and terminal error; also match new baseline physical trajectories to all 30 archived recovery counterparts.',
        outcomes='Preserve every abort, timeout, collision, original capped cost, model, cell and terminal call. Withhold all-scope gain if any baseline/controller pair is incomplete; show complete-pair descriptive gain separately.',
        timing='Actual complete select callbacks including exceptions. Report all calls, p95/max, sum and >500ms; no equal-compute or hard-real-time assertion.',
        statistical_scope='Integration on 15 reused development initials with 30 condition entries; no new independent evidence of generalization, no calibration selection and no confidence intervals presented as risk certificates.',
        ongoing_calibration_changed=False,ongoing_v32_changed=False,partial_formal_outcomes_read=False))
    print(json.dumps(dict(actual_sumo_runs=390,conditions=30,plan_sha256=digest(OUT/'plan.json'))),flush=True)


def inputs():
    committed(OUT/'plan.json')
    plan = read(OUT/'plan.json')
    for name, expected in {**plan['source_sha256'],**plan['inputs_sha256']}.items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen integration source/input changed: '+name)
    sources_committed([ROOT/name for name in plan['source_sha256']])
    if runtime_identity() != plan['runtime']:
        raise ValueError('Frozen runtime changed')
    source = read(recovery.OUT/'plan.json')
    if plan['groups'] != integration_groups(source['cases'],source['baseline_failed_cases']):
        raise ValueError('Incomplete or changed frozen integration schedule')
    for item in plan['models'].values():
        if digest(ROOT/item['path']) != item['sha256']:
            raise ValueError('Frozen learned policy changed')
    return plan


def metadata(group):
    return dict(plan_sha256=digest(OUT/'plan.json'),block_index=group['block_index'],stage='recovery_gate_integration')


def match_archived_baseline(row,trace,source):
    prior, old_trace = source
    equal_fields(row,prior,recovery.PAIR_FIELDS)
    if len(trace) != len(old_trace):
        raise ValueError('New common baseline must retain its complete archived trajectory')
    for actual, expected in zip(trace,old_trace,strict=True):
        equal_fields(actual,expected,PHYSICAL_FIELDS)
    equal_fields(row,prior,('completed','aborted','collision','per_vehicle_cost','per_vehicle_finish','team_cost'))


def worker():
    execution = OUT/'execution'
    startup, ack = execution/'worker.native.json', execution/'worker.acknowledged.json'
    atomic_bytes(startup,encode(dict(native_pid=os.getpid(),parent_pid=os.getppid(),utc=utc())))
    deadline = time.monotonic()+180.
    while not ack.exists():
        if time.monotonic() > deadline:
            raise TimeoutError('No native parent acknowledgment; no simulation started')
        time.sleep(.2)
    proof = read(ack)
    if (proof['native_pid'] != os.getpid() or proof['parent_pid'] != os.getppid()
            or proof['native_start_sha256'] != digest(startup)):
        raise ValueError('Native acknowledgment mismatch')
    plan = inputs()
    recovery.execution_priority()
    torch.set_num_threads(1)
    models = {s:load_coupled(ROOT/m['path'])[0] for s,m in plan['models'].items()}
    old = archived_recovery()
    warm = np.array([[-140.,12.],[-220.,10.],[-190.,12.]])
    for profile in recovery.PROFILES:
        RecoveryJointGate(None,profile,'baseline').select(warm,np.zeros(3))
        for model in models.values():
            for rule in RULES[1:]:
                RecoveryJointGate(model,profile,rule).select(warm,np.zeros(3))
    (OUT/'blocks').mkdir(exist_ok=False)
    completed = 0
    with ExitStack() as stack:
        envs = {p:stack.enter_context(MatchedShiftedMergeEnv(plant=p)) for p in recovery.PROFILES}
        for group in plan['groups']:
            directory = OUT/'blocks'/f"{group['block_index']:05d}"
            directory.mkdir()
            store = EpisodeStore(directory/'observations.sqlite',metadata(group))
            reviews = []
            try:
                for key in group['cases']:
                    source = old[tuple(key[:5])]
                    rule, model_id = key[-2:]
                    model = None if model_id == 'none' else models[model_id]
                    wrapped = []
                    def factory(_):
                        gate = infrastructure.TimedGate(RecoveryJointGate(model,key[4],rule))
                        wrapped.append(gate)
                        return gate
                    started = time.perf_counter()
                    row,trace,initial = run_episode(envs[key[4]],source[0]['seed'],source[0]['scenario'],
                        source[0]['disturbance'],factory,True)
                    cost = np.asarray(row['per_vehicle_finish'])+.05*np.asarray(row['per_vehicle_variation'])
                    row.update(initial=initial.tolist(),plant_name=key[4],rule=rule,model=model_id,
                        controller=rule if rule == 'baseline' else rule+'_'+model_id,
                        certificate_contract='bounded_actuator_memory_stopping',per_vehicle_cost=cost.tolist(),
                        team_cost=float(cost.mean()),control_steps=len(trace),
                        learned_action_steps=sum(bool(t['actual_learning_used']) for t in trace))
                    record = dict(row=row,calls=wrapped[0].calls,episode_wall_seconds=time.perf_counter()-started)
                    store.append(key,record,trace)
                    equal_fields(row,source[0],recovery.PAIR_FIELDS)
                    infrastructure.check_calls(row,trace,record['calls'])
                    review = review_joint_trace(row,trace,model)
                    if rule == 'baseline':
                        match_archived_baseline(row,trace,source)
                    reviews.append(dict(case=key,**review))
                    completed += 1
                    atomic_bytes(execution/'progress.json',encode(dict(utc=utc(),native_pid=os.getpid(),
                        completed_runs=completed,expected_runs=plan['actual_sumo_runs'],
                        completed_blocks=group['block_index'],expected_blocks=plan['blocks'])))
                store.seal(group['cases'])
            finally:
                store.close()
            exclusive_json(directory/'complete.json',dict(**metadata(group),reviews=reviews,
                actual_sumo_runs=len(reviews),database_sha256=digest(directory/'observations.sqlite'),
                seal_sha256=digest(directory/'observations.sqlite.seal.json')))
            atomic_bytes(execution/'progress.json',encode(dict(utc=utc(),native_pid=os.getpid(),
                completed_runs=completed,expected_runs=plan['actual_sumo_runs'],
                completed_blocks=group['block_index']+1,expected_blocks=plan['blocks'])))
    exclusive_json(execution/'collection_complete.json',dict(plan_sha256=digest(OUT/'plan.json'),
        native_pid=os.getpid(),actual_sumo_runs=completed,blocks=plan['blocks'],utc=utc()))
    print(json.dumps(dict(actual_sumo_runs=completed)),flush=True)


def watch():
    inputs()
    infrastructure.OUT, infrastructure.SCRIPT = OUT, SCRIPT
    # The shared watcher validates its global inputs before spawning the worker.
    infrastructure.inputs = inputs
    infrastructure.watch()


def summarize():
    plan = inputs()
    execution = OUT/'execution'
    receipt,launch,done = [read(execution/n) for n in ('exit.json','launch.json','collection_complete.json')]
    infrastructure.validate_completion(plan,digest(OUT/'plan.json'),receipt,launch,done)
    check_acknowledgment(execution,'worker',launch['launcher_pid'],launch['native_pid'])
    torch.set_num_threads(1)
    models = {s:load_coupled(ROOT/m['path'])[0] for s,m in plan['models'].items()}
    old, cases, total = archived_recovery(), [], 0
    for group in plan['groups']:
        directory = OUT/'blocks'/f"{group['block_index']:05d}"
        proof = read(directory/'complete.json')
        if (proof['database_sha256'] != digest(directory/'observations.sqlite')
                or proof['seal_sha256'] != digest(directory/'observations.sqlite.seal.json')
                or proof['actual_sumo_runs'] != len(group['cases'])):
            raise ValueError('Complete unchanged paired block required')
        records = {}
        for index,(key,record,trace) in enumerate(observations(directory/'observations.sqlite',metadata(group),group['cases'])):
            row = record['row']
            model = None if key[-1] == 'none' else models[key[-1]]
            equal_fields(row,old[tuple(key[:5])][0],recovery.PAIR_FIELDS)
            infrastructure.check_calls(row,trace,record['calls'])
            review = dict(case=list(key),**review_joint_trace(row,trace,model,replay=True))
            if review != proof['reviews'][index]:
                raise ValueError('Complete gate replay and collection physical audit differ')
            if key[-2] == 'baseline':
                match_archived_baseline(row,trace,old[tuple(key[:5])])
            records[key[-2:]] = dict(case=list(key),row=row,calls=record['calls'],review=review)
            total += 1
        baseline = records[('baseline','none')]['row']
        for record in records.values():
            equal_fields(record['row'],baseline,recovery.PAIR_FIELDS)
            cases.append(dict(**record,baseline=baseline,**relative_outcome(record['row'],baseline)))
    if total != plan['actual_sumo_runs']:
        raise ValueError('Incomplete integration observations')
    cells = []
    for rule in RULES:
        for model in (('none',) if rule == 'baseline' else (None,*MODELS)):
            for scene in (None,*recovery.SCENES):
                for plant in (None,*recovery.PROFILES):
                    selected = [p for p in cases if p['row']['rule'] == rule
                        and (model is None or p['row']['model'] == model)
                        and (scene is None or p['row']['scenario'] == scene)
                        and (plant is None or p['row']['plant_name'] == plant)]
                    if selected:
                        cells.append(dict(rule=rule,model=model,scene=scene,plant=plant,**cell_summary(selected)))
    target = OUT/'analysis'
    target.mkdir(exist_ok=False)
    report = dict(plan_sha256=digest(OUT/'plan.json'),actual_sumo_runs=total,
        all_physical_and_full_gate_replays_match=True,old_baseline_failure_conditions=10,
        new_independent_initials=0,new_risk_certificate=False,cases=cases,cells=cells,
        scope=plan['statistical_scope'],prior_adapter_scope=plan['prior_adapters'])
    exclusive_json(target/'report.json',report)
    exclusive_json(target/'sha256.json',dict(report=digest(target/'report.json')))
    print(json.dumps(dict(actual_sumo_runs=total,cells=len(cells),all_replays_match=True)),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('freeze','worker','watch','summarize'))
    globals()[parser.parse_args().command]()
