"""Frozen full-grid development of actuator-aware physical stopping recovery."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import time

import numpy as np

from coopgate.actuator_recovery_v34 import ActuatorRecoveryBaseline, class_bounds
from coopgate.disturbance_v9 import disturbance_tape
from coopgate.episode_store_v21 import EpisodeStore, encode, observations
from coopgate.matched_shift_v21 import MatchedShiftedMergeEnv, matched_plant_parameters, tape_fingerprint
from coopgate.merge_v4 import MergeConfig, transition_safe
from coopgate.physical_comparison_analysis_v27 import equal_fields
import audit_buffered_scope_v25 as infrastructure
from audit_online_actuator_v32 import BASELINES, execution_priority
from execute_prospective_v23 import check_acknowledgment, exclusive_json, runtime_identity, sources_committed, utc
from experiment_ego_v13 import run_episode
from profile_phase_runtime_v24 import committed, digest
from recover_checkpoint_v19 import atomic_bytes

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/v34/actuator_recovery_scope'
SCRIPT = 'tools/study_actuator_recovery_v34.py'
PROFILES = ('nominal', 'gain_drag', 'lag_mild', 'lag_strong')
SCENES = ('nominal', 'dense', 'slow_yield', 'asymmetric_speed', 'tight_headway')
SOURCES = ('src/coopgate/actuator_recovery_v34.py', SCRIPT,
           'tests/test_actuator_recovery_v34.py', 'tests/test_recovery_scope_v34.py')
PAIR_FIELDS = ('seed','scenario','initial','disturbance','radius','plant',
               'external_innovation_radius','physical_error_radius','innovation_tape_sha256')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def baseline_records():
    plan = read(BASELINES/'plan.json')
    return {k:(record['row'], trace) for k,record,trace in observations(
        BASELINES/'inputs.sqlite', plan['input_metadata'], plan['input_cases'])}


def complete_cases(keys):
    keys = sorted(tuple(k) for k in keys)
    if len(keys) != 640 or len(set(keys)) != 640 or any(k[-1] != 'baseline' for k in keys):
        raise ValueError('All 640 original baseline conditions are required')
    for scene in SCENES:
        for profile in PROFILES:
            if sum(k[1] == scene and k[4] == profile for k in keys) != 32:
                raise ValueError('Missing original scene/plant cell')
    return [list(k[:-1])+['actuator_recovery'] for k in keys]


def freeze():
    old = read(ROOT/'artifacts/v33/buffered_actuator_audit/plan.json')
    sources = dict(old['source_sha256'])
    sources.update({name:digest(ROOT/name) for name in SOURCES})
    for name,expected in sources.items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen dependency changed: '+name)
    sources_committed([ROOT/p for p in sources])
    records = baseline_records()
    cases = complete_cases(records)
    files = [BASELINES/n for n in ('plan.json','inputs.sqlite','inputs.sqlite.seal.json',
                                   'analysis/report.json','completion_verification.json')]
    OUT.mkdir(parents=True,exist_ok=False)
    exclusive_json(OUT/'plan.json',dict(utc=utc(),source_sha256=sources,
        inputs_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in files},runtime=runtime_identity(),
        models={},cases=cases,blocks=len(cases),actual_sumo_runs=len(cases),
        parameter_bounds={p:class_bounds(p).tolist() for p in PROFILES},
        baseline_failed_cases=[list(k) for k,(r,_) in records.items() if not r['completed']],
        online_inputs='Observed position/speed, actual previous command, decision count and declared plant-class bounds. Initial actuator memory is zero. No true coefficients, simulator memory, seed or future noise reaches the controller.',
        method='Unchanged nominal strong-baseline proposal; accept only if an interval actuator-memory stopping certificate succeeds, otherwise use the certified jerk-limited stopping sequence.',
        purpose='Test the physical scope of a separate development recovery controller, including every original failure. No new MARL or performance-risk guarantee is inferred.',
        comparison='Full same-initial/plant/innovation comparison with the archived nominal-model baseline. The new controller uses declared class bounds; this is not an equal-information ablation or superiority claim over other predictive safety filters.',
        failure_rule='Keep initial non-certification, subsequent aborts, timeouts and negative cost outcomes. Stopping is not task completion. If either paired policy is incomplete, relative performance gain is not an all-scope efficacy result.',
        statistical_scope='160 old initial conditions, shared by four plants. No new independent efficacy sample, calibration, selection or population-risk certificate.',
        actual_prior_risk_calibration_changed=False,ongoing_v32_study_changed=False,
        new_independent_initials=0,new_performance_risk_certificate=False))
    print(json.dumps(dict(actual_sumo_runs=len(cases),plan_sha256=digest(OUT/'plan.json'))),flush=True)


def inputs():
    committed(OUT/'plan.json')
    plan = read(OUT/'plan.json')
    for name,expected in {**plan['source_sha256'],**plan['inputs_sha256']}.items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen recovery source/input changed: '+name)
    sources_committed([ROOT/p for p in plan['source_sha256']])
    if runtime_identity() != plan['runtime']:
        raise ValueError('Frozen runtime changed')
    if complete_cases(baseline_records()) != plan['cases']:
        raise ValueError('Incomplete original scope')
    return plan


def metadata():
    return dict(plan_sha256=digest(OUT/'plan.json'),stage='actuator_recovery_development')


def review_trace(row, trace):
    """Independent plant recurrence plus replay of all online input/evidence.

    True coefficients and innovations are used only here, after execution.
    The original additive-error certificate is deliberately not asserted for
    this different controller. Its old .5 diagnostics remain in the raw rows.
    """
    c = MergeConfig()
    if len(trace) > round(c.horizon/c.decision_dt):
        raise ValueError('Trace exceeds the declared horizon')
    plant = matched_plant_parameters(row['seed'],row['plant_name'],.1)
    if row['plant'] != plant or row['physical_error_radius'] != .5 or row['external_innovation_radius'] != .1:
        raise ValueError('Changed physical comparison or reference bookkeeping')
    if row['certificate_contract'] != 'bounded_actuator_memory_stopping':
        raise ValueError('Incorrect new physical-contract label')
    tape = disturbance_tape(row['seed'],row['disturbance'],.1,250)
    if row['innovation_tape_sha256'] != tape_fingerprint(tape):
        raise ValueError('Changed external innovations')
    x = np.asarray(row['initial'],dtype=float).copy()
    previous, memory = np.zeros(3), np.zeros(3)
    gain, drag, tau = [np.asarray(plant[k]) for k in ('gain','drag','tau')]
    retention = np.zeros(3)
    retention[tau > 0.] = np.exp(-.1/tau[tau > 0.])
    bounds = class_bounds(row['plant_name'])
    for values,low,high in ((gain,0,1),(drag,2,3),(retention,4,5)):
        if np.any(values < bounds[:,low]) or np.any(values > bounds[:,high]):
            raise ValueError('Realized plant outside the declared class')
    gate = ActuatorRecoveryBaseline(row['plant_name'])
    active, finish, variation = np.ones(3,dtype=bool), np.full(3,25.), np.zeros(3)
    cursor = exceeded = violations = recovery_steps = 0
    maximum = max_slew = 0.
    for step,item in enumerate(trace):
        command, info = gate.select(x,previous)
        np.testing.assert_array_equal(command,item['command'])
        for key,value in info.items():
            if key != 'gate_ms' and item.get(key) != value:
                raise ValueError('Online recovery evidence differs: '+key)
        if (np.any(memory < np.asarray(info['actuator_memory_lower'])-1e-10)
                or np.any(memory > np.asarray(info['actuator_memory_upper'])+1e-10)):
            raise ValueError('True actuator memory escaped the observed-command interval')
        np.testing.assert_allclose(item['t'],(step+1)*.5,rtol=0,atol=1e-10)
        variation += active*np.abs(command-previous)
        max_slew = max(max_slew,float(np.max(np.abs(command-previous))/.5))
        recovery_steps += int(info['recovery_applied'])
        for key in ('innovation','actuator_states','disturbance'):
            values = np.asarray(item[key])
            if values.shape != (5,3) or not np.isfinite(values).all():
                raise ValueError('Incomplete physical substep evidence')
        step_error = 0.
        for sub in range(5):
            innovation = tape[cursor]
            cursor += 1
            np.testing.assert_array_equal(innovation,item['innovation'][sub])
            # Independent dynamics: do not call apply_actuator or the interval
            # propagation implementation to reconstruct the realized plant.
            memory = retention*memory+(1.-retention)*command
            acceleration = gain*memory-drag*x[:,1]**2+innovation
            error = acceleration-command
            np.testing.assert_allclose(memory,item['actuator_states'][sub],rtol=0,atol=1e-9)
            np.testing.assert_allclose(error,item['disturbance'][sub],rtol=0,atol=1e-9)
            error_size = float(np.max(np.abs(error)))
            maximum,step_error = max(maximum,error_size),max(step_error,error_size)
            exceeded += int(error_size > .5+1e-9)
            before = x[:,0].copy()
            x[:,1] = np.clip(x[:,1]+acceleration*.1,0.,20.)
            x[:,0] += x[:,1]*.1
            violations += not bool(transition_safe(before[None],x[:,0][None],c)[0])
        np.testing.assert_allclose(x,item['x'],rtol=0,atol=1e-8)
        if item['realized_error_in_contract'] != (step_error <= .5+1e-9):
            raise ValueError('Old error-contract diagnostic differs')
        x = np.asarray(item['x']).copy()
        newly = active & (x[:,0] >= c.goal)
        finish[newly],active[newly] = item['t'],False
        previous = command
    if row['aborted']:
        try:
            gate.select(x,previous)
        except (RuntimeError,ValueError) as exc:
            if str(exc) != row['abort_reason']:
                raise ValueError('Terminal error identity differs') from exc
        else:
            raise ValueError('Recorded abort did not reproduce')
        if trace:
            raise ValueError('Recovery became unavailable after a certified command')
    for actual,key in ((finish,'per_vehicle_finish'),(variation,'per_vehicle_variation'),
                       (maximum,'max_actual_error'),(max_slew,'max_command_slew'),
                       (len(trace)*.5,'elapsed')):
        np.testing.assert_allclose(actual,row[key],rtol=0,atol=1e-8)
    costs = finish+.05*variation
    np.testing.assert_allclose(costs,row['per_vehicle_cost'],rtol=0,atol=1e-9)
    np.testing.assert_allclose(costs.mean(),row['team_cost'],rtol=0,atol=1e-9)
    completed = bool(not active.any() and not row['collision'] and not row['aborted'])
    if (completed != row['completed'] or exceeded != row['error_contract_exceeded_substeps']
            or violations != row['contract_violations'] or bool(row['abort_reason']) != row['aborted']
            or row['nominal_physical_assumption_observed'] != (exceeded == 0)):
        raise ValueError('Full physical outcome accounting differs')
    if not completed and not row['aborted'] and len(trace) != round(c.horizon/c.decision_dt):
        raise ValueError('An incomplete prefix is not a completed timeout execution')
    if violations or row['collision']:
        raise ValueError('A certified execution violated the declared physical contract')
    return dict(commands=len(trace),substeps=cursor,recovery_steps=recovery_steps,
                completed=completed,initial_recovery_unavailable=bool(row['aborted'] and not trace),
                legacy_error_exceeded_substeps=exceeded,physical_violations=violations,
                all_memory_intervals_contain_realized_state=True)


def worker():
    execution = OUT/'execution'
    startup = execution/'worker.native.json'
    atomic_bytes(startup,encode(dict(native_pid=os.getpid(),parent_pid=os.getppid(),utc=utc())))
    ack = execution/'worker.acknowledged.json'
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
    execution_priority()
    old = baseline_records()
    for profile in PROFILES:
        ActuatorRecoveryBaseline(profile).select(np.array([[-140.,12.],[-220.,10.],[-190.,12.]]),np.zeros(3))
    store = EpisodeStore(OUT/'observations.sqlite',metadata())
    reviews = []
    try:
        with ExitStack() as stack:
            envs = {p:stack.enter_context(MatchedShiftedMergeEnv(plant=p)) for p in PROFILES}
            for index,key in enumerate(plan['cases']):
                source = old[(*key[:-1],'baseline')][0]
                wrapped = []
                def factory(_):
                    gate = infrastructure.TimedGate(ActuatorRecoveryBaseline(key[4]))
                    wrapped.append(gate)
                    return gate
                start = time.perf_counter()
                row,trace,initial = run_episode(envs[key[4]],source['seed'],source['scenario'],source['disturbance'],factory,True)
                costs = np.asarray(row['per_vehicle_finish'])+.05*np.asarray(row['per_vehicle_variation'])
                row.update(initial=initial.tolist(),plant_name=key[4],controller='actuator_recovery',
                    certificate_contract='bounded_actuator_memory_stopping',
                    per_vehicle_cost=costs.tolist(),team_cost=float(costs.mean()))
                record = dict(row=row,calls=wrapped[0].calls,episode_wall_seconds=time.perf_counter()-start)
                # Persist the actual outcome before checking; a verifier failure
                # must not erase the evidence which caused it.
                store.append(key,record,trace)
                equal_fields(row,source,PAIR_FIELDS)
                infrastructure.check_calls(row,trace,record['calls'])
                reviews.append(dict(case=key,**review_trace(row,trace)))
                atomic_bytes(execution/'progress.json',encode(dict(utc=utc(),native_pid=os.getpid(),
                    completed_runs=index+1,expected_runs=plan['actual_sumo_runs'])))
        store.seal(plan['cases'])
    finally:
        store.close()
    exclusive_json(execution/'collection_review.json',reviews)
    exclusive_json(execution/'collection_complete.json',dict(plan_sha256=digest(OUT/'plan.json'),
        native_pid=os.getpid(),actual_sumo_runs=len(reviews),blocks=len(reviews),utc=utc(),
        database_sha256=digest(OUT/'observations.sqlite'),seal_sha256=digest(OUT/'observations.sqlite.seal.json'),
        collection_review_sha256=digest(execution/'collection_review.json')))
    print(json.dumps(dict(actual_sumo_runs=len(reviews))),flush=True)


def watch():
    inputs()
    infrastructure.OUT,infrastructure.SCRIPT = OUT,SCRIPT
    infrastructure.watch()


def summarize():
    plan = inputs()
    execution = OUT/'execution'
    receipt,launch,done = [read(execution/n) for n in ('exit.json','launch.json','collection_complete.json')]
    infrastructure.validate_completion(plan,digest(OUT/'plan.json'),receipt,launch,done)
    check_acknowledgment(execution,'worker',launch['launcher_pid'],launch['native_pid'])
    for path,key in ((OUT/'observations.sqlite','database_sha256'),
                     (OUT/'observations.sqlite.seal.json','seal_sha256'),
                     (execution/'collection_review.json','collection_review_sha256')):
        if digest(path) != done[key]:
            raise ValueError('Complete recovery data changed')
    old, prior = baseline_records(),read(execution/'collection_review.json')
    cases,all_calls = [],[]
    for index,(key,record,trace) in enumerate(observations(OUT/'observations.sqlite',metadata(),plan['cases'])):
        row = record['row']
        reference = old[(*key[:-1],'baseline')][0]
        equal_fields(row,reference,PAIR_FIELDS)
        infrastructure.check_calls(row,trace,record['calls'])
        review = dict(case=list(key),**review_trace(row,trace))
        if review != prior[index]:
            raise ValueError('Independent full review differs')
        all_calls.extend(record['calls'])
        cases.append(dict(case=list(key),review=review,row=row,baseline=reference,
            both_complete=bool(row['completed'] and reference['completed'])))
    cells = []
    for profile in PROFILES:
        for scene in (None,*SCENES):
            group = [r for r in cases if r['case'][4] == profile and (scene is None or r['case'][1] == scene)]
            paired = [r for r in group if r['both_complete']]
            mean = lambda role, rows:float(np.mean([r[role]['team_cost'] for r in rows])) if rows else None
            all_complete = len(paired) == len(group)
            gain = (100*(1.-mean('row',paired)/mean('baseline',paired))) if paired else None
            cells.append(dict(plant=profile,scene=scene,runs=len(group),
                baseline_completed=sum(r['baseline']['completed'] for r in group),
                recovery_completed=sum(r['row']['completed'] for r in group),
                baseline_aborted=sum(r['baseline']['aborted'] for r in group),
                recovery_aborted=sum(r['row']['aborted'] for r in group),
                recovery_timeouts=sum(not r['row']['completed'] and not r['row']['aborted'] for r in group),
                baseline_capped_mean_cost=mean('baseline',group),recovery_capped_mean_cost=mean('row',group),
                both_complete_pairs=len(paired),all_scope_cost_gain_percent=gain if all_complete else None,
                complete_pair_descriptive_gain_percent=gain,
                recovery_episodes=sum(r['review']['recovery_steps'] > 0 for r in group),
                initial_recovery_unavailable=sum(r['review']['initial_recovery_unavailable'] for r in group)))
    values = [c['wall_ms'] for c in all_calls]
    report = dict(plan_sha256=digest(OUT/'plan.json'),actual_sumo_runs=len(cases),
        all_independent_physical_and_observer_reviews_pass=True,cases=cases,cells=cells,
        callbacks=dict(calls=len(values),terminal_calls=sum(c['error'] is not None for c in all_calls),
            p50_p95_p99_max_ms=np.percentile(values,[50,95,99,100]).tolist(),over_500ms=sum(v > 500. for v in values)),
        new_independent_initials=0,new_performance_risk_certificate=False,
        interpretation=plan['comparison']+' '+plan['statistical_scope'])
    target = OUT/'analysis'
    target.mkdir(exist_ok=False)
    exclusive_json(target/'report.json',report)
    exclusive_json(target/'sha256.json',dict(report=digest(target/'report.json')))
    print(json.dumps(dict(actual_sumo_runs=len(cases),cells=[r for r in cells if r['scene'] is None],
                         callbacks=report['callbacks'])),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','watch','worker','summarize'))
    globals()[parser.parse_args().action]()
