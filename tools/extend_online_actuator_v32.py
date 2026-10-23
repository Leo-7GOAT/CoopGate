"""Frozen full-grid online actuator comparison; preserve every failure.

This repeats all 160 old initials under four plants, three fixed models and
both preflight controllers. It is development evidence, not fresh calibration.
"""
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
from coopgate.episode_store_v21 import EpisodeStore, encode, observations
from coopgate.matched_shift_v21 import MatchedShiftedMergeEnv
from coopgate.online_actuator_gate_v32 import ReferenceActuatorGate
from coopgate.physical_comparison_analysis_v27 import equal_fields, expected_cases, failed, harmful
from audit_buffered_scope_v25 import TimedGate, check_calls, validate_completion
import audit_online_actuator_v32 as preflight
from execute_prospective_v23 import (acknowledge_child, check_acknowledgment,
    exclusive_json, runtime_identity, sources_committed, utc)
from experiment_ego_v13 import run_episode
from profile_phase_runtime_v24 import committed, digest
from recover_checkpoint_v19 import atomic_bytes

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/v32/online_actuator_complete_grid'
SCRIPT = 'tools/extend_online_actuator_v32.py'
TEST = 'tests/test_online_actuator_extension_v32.py'
MODELS = ('1802', '1812', '1822')
KINDS = ('nominal', 'reference')
PAIR_FIELDS = ('seed', 'scenario', 'initial', 'disturbance', 'radius', 'plant',
    'external_innovation_radius', 'physical_error_radius', 'innovation_tape_sha256')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def full_groups(input_cases, spec):
    actual = [tuple(k) for k in input_cases]
    expected = {k for k in expected_cases('transfer', spec, ['baseline']+['phase05_'+m for m in MODELS])
                if k[-1]!='baseline'}
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Complete original scene/plant/model/initial grid required')
    conditions = sorted({k[:-1] for k in actual},
        key=lambda k:(k[2], spec['scenarios'].index(k[1]), spec['plants'].index(k[4])))
    groups = []
    for index, condition in enumerate(conditions):
        cases = []
        for mi, model in enumerate(MODELS):
            order = KINDS if (index*len(MODELS)+mi) % 2 == 0 else KINDS[::-1]
            cases.extend([*condition, 'phase05_'+model, kind] for kind in order)
        groups.append(dict(block_index=index, condition=list(condition), cases=cases))
    return groups


def relative_outcome(row, baseline, admitted):
    """An incomplete baseline supplies no relative-performance evidence."""
    base_ok, row_ok = not failed(baseline), not failed(row)
    assessable = base_ok and row_ok
    return dict(adopted=bool(admitted), controller_failed=not row_ok,
        baseline_failed=not base_ok, relative_performance_assessable=assessable,
        relative_harm=harmful(row, baseline) if base_ok else None,
        paired_cost_gain_percent=(100*(baseline['team_cost']-row['team_cost'])/
            baseline['team_cost']) if assessable else None)


def summarize_cell(cases):
    complete = [r for r in cases if r['relative_performance_assessable']]
    comparable_adopted = [r for r in cases if r['adopted'] and not r['baseline_failed']]
    times = [c['wall_ms'] for r in cases for c in r['calls']]
    before = sum(r['baseline_team_cost'] for r in complete)
    gain = 100*(before-sum(r['row']['team_cost'] for r in complete))/before if complete else None
    return dict(runs=len(cases), completed=sum(not r['controller_failed'] for r in cases),
        baseline_complete=sum(not r['baseline_failed'] for r in cases),
        adopted=sum(r['adopted'] for r in cases), comparable_adopted=len(comparable_adopted),
        harmful_and_adopted=sum(bool(r['relative_harm']) for r in comparable_adopted),
        adopted_with_incomplete_baseline=sum(r['adopted'] and r['baseline_failed'] for r in cases),
        paired_complete=len(complete), all_scope_cost_gain_percent=gain if len(complete)==len(cases) else None,
        complete_pairs_descriptive_gain_percent=gain,
        out_of_physical_contract=sum(r['row']['error_contract_exceeded_substeps']>0 for r in cases),
        callback_count=len(times), callback_sum_ms=sum(times), callback_over_500ms=sum(t>500 for t in times),
        callback_p95_ms=float(np.quantile(times,.95)) if times else None,
        callback_max_ms=max(times) if times else None)


def freeze():
    prior = preflight.inputs()
    report = read(preflight.OUT/'analysis/report.json')
    receipt = read(preflight.OUT/'completion_verification.json')
    if (not report['all_physical_and_online_input_reviews_pass'] or report['actual_sumo_runs'] != 120
            or receipt['report_sha256'] != digest(preflight.OUT/'analysis/report.json')
            or not receipt['actual_summary_exit_observed']):
        raise ValueError('Complete independently reviewed online preflight required')
    spec = read(preflight.COMPARISON)['protocols']['transfer']['spec']
    groups = full_groups(read(preflight.INTERVENTIONS/'plan.json')['input_cases'], spec)
    sources = dict(prior['source_sha256'])
    sources.update({p:digest(ROOT/p) for p in (SCRIPT, TEST, 'tools/audit_buffered_scope_v25.py')})
    for name, expected in sources.items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen dependency changed: '+name)
    sources_committed([ROOT/name for name in sources])
    paths = [preflight.OUT/name for name in ('plan.json','analysis/report.json',
        'analysis/sha256.json','completion_verification.json')]
    paths += [preflight.COMPARISON, preflight.BASELINES/'inputs.sqlite',
        preflight.BASELINES/'inputs.sqlite.seal.json', preflight.BASELINES/'plan.json']
    OUT.mkdir(exist_ok=False)
    exclusive_json(OUT/'plan.json',dict(utc=utc(),source_sha256=sources,
        inputs_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in paths},
        runtime=runtime_identity(), models=prior['models'], spec=spec, groups=groups,
        blocks=len(groups), actual_sumo_runs=sum(len(g['cases']) for g in groups),
        old_independent_initials=160, new_independent_initials=0, workers=1,
        controllers='Unchanged V32 nominal/reference pair; both repeat the full grid, including the 120 preflight conditions. Repeats are not independent samples.',
        comparison='Same actors, candidate generation, disturbance bank, original cost/time, physical checks, fallback and matched exogenous innovations. Only the prediction actuator model and resulting decisions differ. Search work and execution times need not be equal.',
        order='Block order is original within-scenario index, scene then plant; nominal/reference execution order alternates by fixed pair index, never by outcomes.',
        comparators='Original completed baseline and phase05 traces on all matching conditions, with exact external-input checks. Historical phase05 has a different innovation bank and implementation.',
        failure_rule='Every failure and capped original metric is retained. No all-scope relative cost gain when a pair is incomplete; complete-pair gains are separately labeled descriptive. Incomplete baseline gives no relative-performance evidence.',
        review='All physical traces, reconstructed online memories, finite-bank admission records and terminal-call timings are reviewed during execution and after full successful exit.',
        scope='All five scenes, four plants, three models and 32 initials per scene; shared old development data. No new risk certificate, real-time guarantee or generalization claim.',
        new_risk_certificate=False, active_calibration_outcomes_read=False))
    print(json.dumps(dict(blocks=len(groups),actual_sumo_runs=3840,plan_sha256=digest(OUT/'plan.json'))),flush=True)


def inputs():
    committed(OUT/'plan.json')
    plan = read(OUT/'plan.json')
    for name, expected in {**plan['source_sha256'],**plan['inputs_sha256']}.items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen full-grid source/input changed: '+name)
    sources_committed([ROOT/name for name in plan['source_sha256']])
    if runtime_identity() != plan['runtime']:
        raise ValueError('Frozen runtime changed')
    preflight.inputs()
    return plan


def metadata(group):
    return dict(plan_sha256=digest(OUT/'plan.json'),block_index=group['block_index'],stage='full_online_grid')


def baselines():
    plan = read(preflight.BASELINES/'plan.json')
    return {key:record['row'] for key,record,_ in observations(preflight.BASELINES/'inputs.sqlite',
        plan['input_metadata'],plan['input_cases'])}


def verify(record, trace, key, baseline, old):
    row = record['row']
    equal_fields(row,baseline,PAIR_FIELDS)
    equal_fields(row,old,PAIR_FIELDS)
    check_calls(row,trace,record['calls'])
    review = preflight.review_online(row,trace,key[-1],key[4])
    if review != record['review']:
        raise ValueError('Independent online review differs')
    return relative_outcome(row,baseline,review['actual_admitted_steps']>0)


def worker():
    execution = OUT/'execution'
    startup = execution/'worker.native.json'
    atomic_bytes(startup,encode(dict(native_pid=os.getpid(),parent_pid=os.getppid(),utc=utc())))
    deadline = time.monotonic()+180
    ack = execution/'worker.acknowledged.json'
    while not ack.exists():
        if time.monotonic()>deadline:
            raise TimeoutError('No parent acknowledgment; no simulation started')
        time.sleep(.2)
    proof = read(ack)
    if (proof['native_pid']!=os.getpid() or proof['parent_pid']!=os.getppid()
            or proof['native_start_sha256']!=digest(startup)):
        raise ValueError('Native process acknowledgment mismatch')
    plan = inputs()
    preflight.execution_priority()
    torch.set_num_threads(1)
    models = {s:load_coupled(ROOT/m['path'])[0] for s,m in plan['models'].items()}
    old, base = preflight.archived(), baselines()
    for profile in plan['spec']['plants']:
        for kind in KINDS:
            ReferenceActuatorGate(models['1802'],profile,kind).select(
                np.array([[-80.,12.],[-85.,13.],[-120.,11.]]),np.zeros(3))
    root = OUT/'blocks'
    root.mkdir(exist_ok=False)
    completed = 0
    with ExitStack() as stack:
        envs = {p:stack.enter_context(MatchedShiftedMergeEnv(plant=p)) for p in plan['spec']['plants']}
        for group in plan['groups']:
            directory = root/f"{group['block_index']:05d}"
            directory.mkdir()
            store = EpisodeStore(directory/'observations.sqlite',metadata(group))
            try:
                for key in group['cases']:
                    source = old[tuple(key[:-1])][0]
                    model = models[key[5].rsplit('_',1)[1]]
                    wrapped = []
                    def factory(_):
                        gate = TimedGate(ReferenceActuatorGate(model,key[4],key[-1]))
                        wrapped.append(gate)
                        return gate
                    started = time.perf_counter()
                    row,trace,initial = run_episode(envs[key[4]],source['seed'],source['scenario'],source['disturbance'],factory,True)
                    costs = np.asarray(row['per_vehicle_finish'])+.05*np.asarray(row['per_vehicle_variation'])
                    row.update(initial=initial.tolist(),plant_name=key[4],controller=key[-1]+'_'+key[5].rsplit('_',1)[1],
                        per_vehicle_cost=costs.tolist(),team_cost=float(costs.mean()))
                    record = dict(row=row,calls=wrapped[0].calls,episode_wall_seconds=time.perf_counter()-started,
                        review=preflight.review_online(row,trace,key[-1],key[4]))
                    # Commit actual execution even if the subsequent input/timing review fails.
                    store.append(key,record,trace)
                    verify(record,trace,key,base[(*key[:-2],'baseline')],source)
                    completed += 1
                store.seal(group['cases'])
            finally:
                store.close()
            exclusive_json(directory/'complete.json',dict(**metadata(group),actual_sumo_runs=len(group['cases']),
                database_sha256=digest(directory/'observations.sqlite'),seal_sha256=digest(directory/'observations.sqlite.seal.json')))
            atomic_bytes(execution/'progress.json',encode(dict(utc=utc(),native_pid=os.getpid(),
                completed_blocks=group['block_index']+1,expected_blocks=plan['blocks'],
                completed_runs=completed,expected_runs=plan['actual_sumo_runs'])))
            print(json.dumps(dict(block=group['block_index'],completed_runs=completed,total_runs=plan['actual_sumo_runs'])),flush=True)
    if completed!=plan['actual_sumo_runs']:
        raise ValueError('Full original grid not completed')
    exclusive_json(execution/'collection_complete.json',dict(plan_sha256=digest(OUT/'plan.json'),
        native_pid=os.getpid(),actual_sumo_runs=completed,blocks=plan['blocks'],utc=utc()))


def watch():
    inputs()
    execution = OUT/'execution'
    execution.mkdir(exist_ok=False)
    with (execution/'worker.stdout.log').open('x') as stdout,(execution/'worker.stderr.log').open('x') as stderr:
        command = [sys.executable,'-X','utf8',SCRIPT,'worker']
        process = subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        exclusive_json(execution/'spawn.json',dict(watcher_pid=os.getpid(),launcher_pid=process.pid,
            command=command,plan_sha256=digest(OUT/'plan.json'),utc=utc()))
        try:
            native = acknowledge_child(process,execution,'worker')
            exclusive_json(execution/'launch.json',dict(**native,watcher_pid=os.getpid(),plan_sha256=digest(OUT/'plan.json'),utc=utc()))
        except Exception as error:
            exclusive_json(execution/'observation_error.json',dict(type=type(error).__name__,message=str(error),launcher_pid=process.pid,utc=utc()))
            code = process.wait()
            exclusive_json(execution/'exit.json',dict(launcher_pid=process.pid,exit_code=code,plan_sha256=digest(OUT/'plan.json'),utc=utc()))
            raise
        code = process.wait()
        exclusive_json(execution/'exit.json',dict(**native,exit_code=code,plan_sha256=digest(OUT/'plan.json'),utc=utc()))
    if code:
        raise SystemExit(code)


def summarize():
    plan = inputs()
    execution = OUT/'execution'
    receipt,launch,done = [read(execution/name) for name in ('exit.json','launch.json','collection_complete.json')]
    validate_completion(plan,digest(OUT/'plan.json'),receipt,launch,done)
    check_acknowledgment(execution,'worker',launch['launcher_pid'],launch['native_pid'])
    old,base = preflight.archived(),baselines()
    cases = []
    for group in plan['groups']:
        directory = OUT/'blocks'/f"{group['block_index']:05d}"
        complete = read(directory/'complete.json')
        if (complete['database_sha256']!=digest(directory/'observations.sqlite')
                or complete['seal_sha256']!=digest(directory/'observations.sqlite.seal.json')
                or complete['actual_sumo_runs']!=len(group['cases'])):
            raise ValueError('Complete sealed block required')
        for key,record,trace in observations(directory/'observations.sqlite',metadata(group),group['cases']):
            baseline,previous = base[(*key[:-2],'baseline')],old[key[:-1]][0]
            outcome = verify(record,trace,key,baseline,previous)
            cases.append(dict(case=list(key),**record,**outcome,baseline_team_cost=baseline['team_cost'],
                phase05_outcome=relative_outcome(previous,baseline,previous['learned_action_steps']>0)))
    if len(cases)!=plan['actual_sumo_runs']:
        raise ValueError('Every frozen condition required')
    cells = []
    for scene in plan['spec']['scenarios']:
        for plant in plan['spec']['plants']:
            for model in MODELS:
                for kind in KINDS:
                    rows = [r for r in cases if r['case'][1]==scene and r['case'][4]==plant
                        and r['case'][5]=='phase05_'+model and r['case'][-1]==kind]
                    cells.append(dict(scene=scene,plant=plant,model=model,kind=kind,**summarize_cell(rows)))
    plant_cells = [dict(plant=plant,kind=kind,**summarize_cell([r for r in cases
        if r['case'][4]==plant and r['case'][-1]==kind])) for plant in plan['spec']['plants'] for kind in KINDS]
    report = dict(plan_sha256=digest(OUT/'plan.json'),actual_sumo_runs=len(cases),cells=cells,
        plant_cells=plant_cells,cases=cases,all_physical_and_input_reviews_pass=True,new_independent_initials=0,
        new_risk_certificate=False,scope=plan['scope'],failure_rule=plan['failure_rule'])
    analysis = OUT/'analysis'
    analysis.mkdir(exist_ok=False)
    exclusive_json(analysis/'report.json',report)
    exclusive_json(analysis/'sha256.json',dict(report=digest(analysis/'report.json')))
    print(json.dumps(dict(actual_sumo_runs=len(cases),cells=len(cells),plant_cells=plant_cells)),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','watch','worker','summarize'))
    globals()[parser.parse_args().action]()
