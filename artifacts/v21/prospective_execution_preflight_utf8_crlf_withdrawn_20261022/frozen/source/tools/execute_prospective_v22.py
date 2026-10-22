"""Complete, fixed-budget SUMO calibration and confirmation execution.

The preflight uses one OLD initial and synthetic calibration, never new risk
evidence. Fresh execution requires its successful native process receipts and
a separately committed protocol. Interrupted stores are preserved; a separate
explicit recovery protocol is required, never an outcome-dependent restart.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import sumolib
import torch

from coopgate.disturbance_v9 import DisturbedMergeEnv
from coopgate.episode_store_v21 import EpisodeStore, encode, inventory, observations
from coopgate.prospective_data_v21 import paired_observations, evaluate_adapters
from coopgate.prospective_validation_v21 import (
    ADOPTION_MIN, CALIBRATION_N, CONFIRMATION_N, GAIN_PERCENT_MIN, MODELS, MODES,
    calibrate_observations, profile_names,
)
from observable_profile_prediction_v22 import (
    forecast, prediction_from_rows, choose_profiles, verify_clean_identity,
)
from prepare_prospective_validation_v21 import read_rules, RULES, check_reserved_ranges
from probe_disturbed_gate_v19 import digest
from prospective_runtime_v21 import IndependentReview, execute, method_definitions, verify_record
from recover_checkpoint_v19 import atomic_bytes
from recover_interrupted_v21 import compare_replay, manifest, read
from select_completed_v21_scoped import require_project_stopped
from start_final_v19 import require_committed, verify_manifest
from study_gate_v19 import regime_for
from study_matched_transfer_v21 import factories


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = 'tools/execute_prospective_v22.py'
TEST = 'tests/test_execute_prospective_v22.py'
CHOICE = ROOT/'artifacts/v21/validation_attempts/attempt_1/selection.json'
OUTPUTS = {
    'preflight': ROOT/'artifacts/v21/prospective_execution_preflight_utf8_20261022',
    'prospective': ROOT/'artifacts/v21/prospective_execution_attempt_1_utf8',
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def native_snapshot():
    """Use explicit UTF-8 at both ends; never hide undecodable live processes."""
    command = ("$OutputEncoding = [System.Text.UTF8Encoding]::new($false)\n"
               "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)\n"
               "Get-CimInstance Win32_Process | Where-Object { "
               "$_.Name -match '^(python|sumo)' } | Select-Object "
               "ProcessId,ParentProcessId,Name,CreationDate,CommandLine | ConvertTo-Json -Depth 3")
    result = subprocess.run(['powershell','-NoProfile','-Command',command],
                            text=False,capture_output=True,check=True)
    raw = result.stdout.decode('utf-8-sig')
    data = json.loads(raw) if raw.strip() else []
    data = data if isinstance(data,list) else [data]
    if any(not isinstance(p,dict) or type(p.get('ProcessId')) is not int
           or type(p.get('ParentProcessId')) is not int or not isinstance(p.get('Name'),str) for p in data):
        raise ValueError('Incomplete native process identity; refuse to infer no workers')
    return data


def exclusive_json(path, value):
    path = Path(path)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def value_digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def source_paths():
    # Include transitive local imports; existing frozen scientific files are
    # never edited to accommodate this executor.
    return sorted({*ROOT.joinpath('src/coopgate').glob('*.py'),
                   *ROOT.joinpath('tools').glob('*.py'), ROOT/TEST, ROOT/'pyproject.toml'})


def sources_committed(paths):
    tracked = subprocess.run(['git', 'ls-files', '-z'], cwd=ROOT, check=True,
                             capture_output=True).stdout.decode('utf-8').split('\0')
    wanted = {p.relative_to(ROOT).as_posix() for p in paths}
    changed = subprocess.run(['git', 'diff', '--name-only', '-z', 'HEAD'], cwd=ROOT,
                             check=True, capture_output=True).stdout.decode('utf-8').split('\0')
    if not wanted <= set(tracked) or wanted & set(changed):
        raise ValueError('All execution sources must be committed and unchanged')


def runtime_identity():
    binaries = {}
    for name in ('sumo', 'netconvert'):
        located = sumolib.checkBinary(name)
        path = Path(shutil.which(located) or located).resolve()
        binaries[name] = dict(path=str(path), sha256=digest(path))
    return dict(python=sys.version, executable=str(Path(sys.executable).resolve()),
        packages={p:importlib.metadata.version(p) for p in ('numpy', 'scipy', 'torch', 'numba', 'traci', 'sumolib')},
        binaries=binaries)


def stages(family):
    result = {}
    for name, count, seed, shards, deployment in (
            ('preflight', 1, 63100001, 1, True),
            ('calibration', CALIBRATION_N, 63500001, 3, False),
            ('confirmation', CONFIRMATION_N, 63600001, 3, True)):
        result[name] = dict(seed_base=seed, n_initials=count, shards=shards,
            sampling='iid_mixture', modes=list(MODES), methods=method_definitions(family),
            all_traces_saved=True, audit_initial_indices=[], data_role=name,
            main_runs=count*40, observed_shadow_runs=count*10,
            deployment_runs=count*40 if deployment else 0, deployment=deployment,
            total_sumo_runs=count*(90 if deployment else 50))
    return result


def synthetic_calibration(family):
    return dict(primary_family=family, common_profiles=list(profile_names(family)),
        synthetic_fixture_only=True, risk_certified=False,
        conformal=dict(infinite_q=False, q=.1, calibration_initials=4096),
        ltt_selection={str(s):'baseline' if j == 1 else f'{family}_{s}' for j,s in enumerate(MODELS)},
        mean_spi_selection={str(s):f'{"nominal" if j == 2 else "raw"}_{s}' for j,s in enumerate(MODELS)})


def assigned(spec, shard):
    if type(shard) is not int or not 0 <= shard < spec['shards']:
        raise ValueError('Unknown worker shard')
    return list(range(shard, spec['n_initials'], spec['shards']))


def schedule(spec, index):
    if type(index) is not int or not 0 <= index < spec['n_initials']:
        raise ValueError('Unknown independent initial')
    names = list(spec['methods'])
    roles = [(index, 'clean', n, 'observed_shadow') for n in names]
    for role in ('collection', 'deployment') if spec['deployment'] else ('collection',):
        for mi, mode in enumerate(spec['modes']):
            order = np.random.default_rng(spec['seed_base']+index+mi+1919).permutation(names)
            roles.extend((index, mode, str(n), role) for n in order)
    return roles


def deployment_owners(choices, family):
    expected = ('raw', 'nominal', family, 'conformal', 'ltt', 'mean_spi')
    names = ('baseline', *profile_names(family))
    if set(choices) != set(expected):
        raise ValueError('All six comparison adapters are required')
    owners = {n:[] for n in names}
    for adapter in expected:
        if set(choices[adapter]) != set(map(str, MODELS)):
            raise ValueError('Every fixed model must be retained')
        for seed in map(str, MODELS):
            name = choices[adapter][seed]
            if name not in names or (name != 'baseline' and not name.endswith('_'+seed)):
                raise ValueError('Deployment cannot cross models or add profiles')
            if adapter in ('raw', 'nominal', family) and name != f'{adapter}_{seed}':
                raise ValueError('Fixed comparison adapter changed')
            owners[name].append([adapter, seed])
    # The three fixed adapters collectively require all nine candidates. A
    # tenth baseline trajectory is always executed, even when nobody selects it.
    if any(not owners[n] for n in names if n != 'baseline'):
        raise ValueError('Whole-family deployment coverage is incomplete')
    return owners


def dispatch_factory(profile, choices, family, observed, fs):
    owners = deployment_owners(choices, family)
    if profile not in owners:
        raise ValueError('Unregistered deployment profile')
    initial = np.asarray(observed).copy()
    def construct(current):
        np.testing.assert_allclose(current, initial, atol=1e-8, rtol=0)
        for adapter, seed in owners[profile]:
            if choices[adapter][seed] != profile:
                raise ValueError('Frozen whole-controller choice changed')
        return fs[profile](current)
    return construct


def final_acceptance(calibration, evaluation, family, n=CONFIRMATION_N):
    members = evaluation['results'][family]['per_training_seed']
    if set(members) != set(map(str, MODELS)):
        raise ValueError('Incomplete primary model family')
    decisions = {}
    for seed, row in members.items():
        actual = row['actual_random_mode']
        decisions[seed] = bool(row['runs'] == n and actual['adopted']/n >= ADOPTION_MIN
            and row['mean_team_gain_percent'] >= GAIN_PERCENT_MIN
            and row['mean_team_gain_ci95'][0] > 0)
    certified = calibration['primary_certificate']['all_fixed_models_both_targets_certified']
    return dict(primary_risk_certified=bool(certified), per_model_registered_usefulness=decisions,
        complete_primary_success=bool(certified and all(decisions.values())),
        all_actual_runs_completed=all(r['completed']==n for r in members.values()),
        comparator_can_replace_failed_primary=False, statewise_safety_guarantee=False,
        real_time_deployment_claim=False)


def require_native_completion(directory, plan_sha, stage, shards):
    execution = Path(directory)/'execution'
    watch, receipt = read(execution/'watcher_launch.json'), read(execution/'supervisor_exit.json')
    if (watch['plan_sha256'] != plan_sha or receipt['plan_sha256'] != plan_sha
            or watch['stage'] != stage or receipt['stage'] != stage
            or receipt['supervisor_pid'] != watch['supervisor_pid'] or receipt['exit_code'] != 0):
        raise ValueError('Actual successful supervisor termination is required')
    for phase in ('collect', 'review'):
        launch, exits = read(execution/f'{phase}_launch.json'), read(execution/f'{phase}_exits.json')
        if (launch['plan_sha256'] != plan_sha or launch['stage'] != stage
                or launch['supervisor_pid'] != watch['supervisor_pid']
                or len(launch['worker_pids']) != shards or len(set(launch['worker_pids'])) != shards
                or [r['pid'] for r in exits['workers']] != launch['worker_pids']
                or any(r['exit_code'] != 0 for r in exits['workers'])):
            raise ValueError('Every original worker/reviewer exit must be successful')
    complete = read(execution/'completed.json')
    if complete['plan_sha256'] != plan_sha or complete['stage'] != stage:
        raise ValueError('Completion identity changed')
    return complete


def freeze(scope):
    output = OUTPUTS[scope]
    if output.exists():
        raise FileExistsError('Preserve prior protocol and all executions')
    require_project_stopped(native_snapshot())
    rules = read_rules()
    require_committed([CHOICE])
    choice = read(CHOICE)
    family = choice['selected_family']
    if (choice['rules_sha256'] != digest(RULES/'plan.json') or family is None
            or choice['common_profiles'] != list(profile_names(family)) or choice['attempt'] != 1
            or not choice['candidates'][family]['eligible']):
        raise ValueError('Different complete-family selection or risk budget')
    files = source_paths()
    sources_committed(files)
    sources = {p.relative_to(ROOT).as_posix():digest(p) for p in files}
    prerequisites = {}
    if scope == 'prospective':
        old_output, old = load('preflight')
        done = complete_stage(old_output, old, 'preflight')
        verify_manifest(old_output/'preflight/analysis')
        accepted = read(old_output/'preflight/analysis/acceptance.json')
        if (not accepted['integration_only'] or accepted['all_runtime_offline_profile_choices_exact'] != 18
                or accepted['new_independent_initials'] != 0 or accepted['risk_certified']):
            raise ValueError('Preflight must verify every actual and offline adapter choice')
        require_committed([old_output/'preflight/execution/supervisor_exit.json',
                           old_output/'preflight/execution/completed.json',
                           old_output/'preflight/analysis/acceptance.json', old_output/'preflight/analysis/sha256.json'])
        if old['source_sha256'] != sources or old['family'] != family:
            raise ValueError('Full pipeline preflight must use identical sources and family')
        prerequisites = dict(preflight_plan_sha256=digest(old_output/'frozen/plan.json'),
            preflight_completion_sha256=digest(old_output/'preflight/execution/completed.json'),
            preflight_supervisor_exit_sha256=digest(old_output/'preflight/execution/supervisor_exit.json'),
            preflight_analysis_manifest_sha256=digest(old_output/'preflight/analysis/sha256.json'),
            preflight_review_counts=done['review_counts'])
        check_reserved_ranges()
    plan = dict(scope=scope, utc=utc(), family=family, attempt=choice['attempt'],
        allocated_error=choice['allocated_error'], models=rules['models'], stages=stages(family),
        selection_sha256=digest(CHOICE), rules_sha256=digest(RULES/'plan.json'),
        source_sha256=sources, runtime=runtime_identity(), prerequisites=prerequisites,
        fixture=synthetic_calibration(family), new_calibration_authorized=scope=='prospective',
        primary_main_runs=204800, observed_shadow_runs=51200, deployment_runs=40960,
        prospective_total_sumo_runs=296960, independent_calibration_initials=4096,
        independent_confirmation_initials=1024, preflight_new_independent_initials=0,
        deployment_rule='Choose all 18 adapter/model profiles from observed-only predictions and fixed calibration before collecting real trajectories; execute all ten whole profiles in four modes, share identical profile executions across adapter aliases, compare every command/state and non-timing result with separate common-grid runs',
        timing_rule='Retain controller/setup timings, full observed forecast wall time, selection wall time and complete initial wall time; predictor cost belongs to conformal, not to adapters that need no forecast; no realtime claim',
        failure_rule='Retain incomplete/aborted/colliding physical episodes and finite capped metrics. Infrastructure or initialization errors stop the stage; preserve all prefixes; no outcome-based exclusions or automatic rerun',
        interruption_rule='New explicit frozen recovery protocol required; preserve committed prefixes and unknown original exits; replay only deterministic audit boundaries and missing suffixes, never reset risk attempt or add initials',
        analysis_rule='No partial performance analysis. Calibration only after all collection/review/native successful exits; final requires committed calibration. Run final even if the primary risk claim fails; it cannot rescue that claim',
        immutable_policy=True, unchanged_statistical_rules=True)
    frozen = output/'frozen'
    frozen.mkdir(parents=True)
    exclusive_json(frozen/'plan.json', plan)
    for name in sources:
        target = frozen/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
    exclusive_json(frozen/'sha256.json', manifest(frozen))
    print(json.dumps(dict(scope=scope, plan_sha256=digest(frozen/'plan.json'),
        authorized_runs=90 if scope=='preflight' else 296960, new_independent_initials=0 if scope=='preflight' else 5120)), flush=True)


def load(scope):
    output = OUTPUTS[scope]
    frozen = output/'frozen'
    verify_manifest(frozen)
    require_committed([frozen/'plan.json', frozen/'sha256.json'])
    plan = read(frozen/'plan.json')
    if (plan['scope'] != scope or plan['stages'] != stages(plan['family'])
            or digest(CHOICE) != plan['selection_sha256'] or digest(RULES/'plan.json') != plan['rules_sha256']):
        raise ValueError('Frozen execution design, selection or statistical rules changed')
    read_rules()
    for name, expected in plan['source_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen execution source changed: '+name)
    for item in plan['models'].values():
        if digest(ROOT/item['path']) != item['sha256']:
            raise ValueError('Frozen learned model changed')
    if runtime_identity() != plan['runtime']:
        raise ValueError('Frozen Python/package/SUMO runtime changed')
    if scope == 'prospective':
        previous = OUTPUTS['preflight']
        links = dict(preflight_plan_sha256=previous/'frozen/plan.json',
            preflight_completion_sha256=previous/'preflight/execution/completed.json',
            preflight_supervisor_exit_sha256=previous/'preflight/execution/supervisor_exit.json',
            preflight_analysis_manifest_sha256=previous/'preflight/analysis/sha256.json')
        if any(digest(path) != plan['prerequisites'][key] for key,path in links.items()):
            raise ValueError('Required complete preflight evidence changed')
        verify_manifest(previous/'preflight/analysis')
        complete_stage(previous, read(previous/'frozen/plan.json'), 'preflight')
    return output, plan


def stage_spec(plan, stage):
    if ((plan['scope']=='preflight' and stage!='preflight')
            or (plan['scope']=='prospective' and stage not in ('calibration', 'confirmation'))):
        raise ValueError('This frozen protocol does not authorize that data split')
    return plan['stages'][stage]


def calibration_for(output, plan, stage):
    if stage == 'preflight':
        return plan['fixture'], value_digest(plan['fixture'])
    if stage == 'calibration':
        return None, None
    complete_stage(output, plan, 'calibration')
    directory = output/'calibration/analysis'
    verify_manifest(directory)
    path = directory/'calibration.json'
    require_committed([path, directory/'sha256.json'])
    result = read(path)
    if (result['execution_plan_sha256'] != digest(output/'frozen/plan.json')
            or result['primary_family'] != plan['family'] or result['primary_attempt'] != plan['attempt']
            or result['initial_count'] != CALIBRATION_N):
        raise ValueError('Different or incomplete fixed calibration')
    return result, digest(path)


def block_metadata(output, stage, index, cal_sha):
    return dict(role='prospective_common_grid', plan_sha256=digest(output/'frozen/plan.json'),
                stage=stage, initial_index=index, calibration_sha256=cal_sha)


def warm_factories(fs):
    state = np.array([[-80.,12.],[-85.,13.],[-120.,11.]])
    for factory in fs.values():
        factory(state).select(state, np.zeros(3))


def collect_initial(output, plan, stage, index, env, fs, calibration, cal_sha):
    spec = stage_spec(plan, stage)
    protocol = dict(spec=spec, models=plan['models'])
    directory = output/stage/'blocks'/f'{index:05d}'
    directory.mkdir(parents=True, exist_ok=False)
    expected = schedule(spec, index)
    store = EpisodeStore(directory/'observations.sqlite', block_metadata(output, stage, index, cal_sha))
    started = time.perf_counter()
    try:
        # Reset only to acquire the current observable state. The predictor has
        # no access to the real seed, scenario, future mode or disturbance tape.
        env.reset(spec['seed_base']+index, regime_for(spec, index))
        observed = env.x.copy()
        def retain(name, row, trace):
            store.append((index, 'clean', name, 'observed_shadow'), row, trace)
        prediction = forecast(observed, protocol, plan['family'], on_record=retain)
        selection_started = time.perf_counter()
        choices = choose_profiles(prediction, calibration, plan['family']) if calibration is not None else None
        owners = deployment_owners(choices, plan['family']) if choices is not None else None
        selection_wall = (time.perf_counter()-selection_started)*1000
        if inventory(directory/'observations.sqlite')['cases'] != [list(k) for k in expected[:10]]:
            raise ValueError('Selection must precede every real common-grid outcome')
        exclusive_json(directory/'decision.json', dict(prediction=prediction, choices=choices, owners=owners,
            calibration_sha256=cal_sha, selection_wall_ms=selection_wall, actual_main_records_before_selection=0,
            observed_shadow_records_before_selection=10, utc=utc()))
        for case in expected[10:]:
            _, mode, name, role = case
            factory = (dispatch_factory(name, choices, plan['family'], observed, fs)
                       if role=='deployment' else fs[name])
            row, trace = execute(env, (index, mode, name), spec, factory)
            store.append(case, row, trace)
            # Persistence precedes assertions: erroneous/incomplete runs remain.
            verify_record(row, trace, spec)
        store.seal(expected)
        exclusive_json(directory/'complete.json', dict(initial_index=index, actual_sumo_runs=len(expected),
            full_initial_wall_ms=(time.perf_counter()-started)*1000,
            plan_sha256=digest(output/'frozen/plan.json'), calibration_sha256=cal_sha))
        exclusive_json(directory/'sha256.json', manifest(directory))
    finally:
        store.close()


def load_block(output, plan, stage, index, cal_sha):
    directory = output/stage/'blocks'/f'{index:05d}'
    verify_manifest(directory)
    spec = stage_spec(plan, stage)
    records = list(observations(directory/'observations.sqlite', block_metadata(output, stage, index, cal_sha), schedule(spec,index)))
    for (_,mode,name,role), row, _ in records:
        if row['controller'] != name or row['disturbance'] != mode:
            raise ValueError('Stored role does not match its controller or mode')
        if role != 'observed_shadow' and (row['initial_index'] != index or row['seed'] != spec['seed_base']+index
                                         or row['scenario'] != regime_for(spec,index)):
            raise ValueError('Stored execution does not match its independent initial')
    receipt = read(directory/'complete.json')
    if (receipt['initial_index'] != index or receipt['actual_sumo_runs'] != len(records)
            or receipt['plan_sha256'] != digest(output/'frozen/plan.json') or receipt['calibration_sha256'] != cal_sha):
        raise ValueError('Complete initial identity changed')
    return directory, records


def verify_shadow_trace(shadow, actual):
    if len(shadow) != len(actual):
        raise ValueError('Observed forecast has different trajectory length')
    for first, second in zip(shadow, actual):
        np.testing.assert_array_equal(first['command'], second['command'])
        np.testing.assert_allclose(first['x'], second['x'], atol=1e-8, rtol=0)
        np.testing.assert_array_equal(first['disturbance'], second['disturbance'])


def review_initial(output, plan, stage, index, calibration, cal_sha, reviewer):
    directory, records = load_block(output, plan, stage, index, cal_sha)
    by_key = {key:(row, trace) for key,row,trace in records}
    decision = read(directory/'decision.json')
    shadow_rows = {name:row for (_,_,name,role),row,_ in records if role=='observed_shadow'}
    prediction = prediction_from_rows(decision['prediction']['observed_initial'], plan['family'], shadow_rows)
    for key, value in prediction.items():
        np.testing.assert_equal(value, decision['prediction'][key])
    if (decision['calibration_sha256'] != cal_sha or decision['actual_main_records_before_selection'] != 0
            or decision['observed_shadow_records_before_selection'] != 10):
        raise ValueError('Selection did not precede real execution')
    clean = {name:row for (_,mode,name,role),row,_ in records if role=='collection' and mode=='clean'}
    verify_clean_identity(prediction, clean)
    for name in shadow_rows:
        verify_shadow_trace(by_key[index,'clean',name,'observed_shadow'][1], by_key[index,'clean',name,'collection'][1])
    if calibration is not None:
        choices = choose_profiles(prediction, calibration, plan['family'])
        np.testing.assert_equal(choices, decision['choices'])
        np.testing.assert_equal(deployment_owners(choices, plan['family']), decision['owners'])
    elif decision['choices'] is not None or decision['owners'] is not None:
        raise ValueError('Calibration collection must not select from its own outcomes')
    reviewer.cache.clear()  # Cache is only a numerical memo, never a sample.
    for (i, mode, name, role), row, trace in records:
        if role == 'collection':
            reviewer.check(row, trace)
        elif role == 'deployment':
            verify_record(row, trace, plan['stages'][stage])
            reference, reference_trace = by_key[i,mode,name,'collection']
            compare_replay(reference, row, reference_trace, trace)
    return dict(main=40, observed_shadow=10, deployment=40 if calibration is not None else 0,
                deployed_adapter_mode_checks=72 if calibration is not None else 0)


def worker(scope, stage, phase, shard):
    torch.set_num_threads(1)
    output, plan = load(scope)
    spec = stage_spec(plan, stage)
    ids = assigned(spec, shard)
    calibration, cal_sha = calibration_for(output, plan, stage)
    counts = Counter()
    if phase == 'collect':
        fs = factories(dict(spec=spec, models=plan['models']))
        warm_factories(fs)
        with DisturbedMergeEnv() as env:
            for number, index in enumerate(ids, 1):
                collect_initial(output, plan, stage, index, env, fs, calibration, cal_sha)
                print(json.dumps(dict(stage=stage, phase=phase, shard=shard, completed_index=index,
                                      completed_initials=number, assigned_initials=len(ids))), flush=True)
        extra = {}
    else:
        reviewer = IndependentReview(dict(spec=spec, models=plan['models']))
        for number, index in enumerate(ids, 1):
            counts.update(review_initial(output, plan, stage, index, calibration, cal_sha, reviewer))
            print(json.dumps(dict(stage=stage, phase=phase, shard=shard, reviewed_initials=number,
                                  assigned_initials=len(ids))), flush=True)
        extra = dict(role_counts=dict(counts), review_counts=reviewer.counts, retained_error_cases=reviewer.error_cases)
    exclusive_json(output/stage/'execution'/f'{phase}_{shard}.complete.json', dict(pid=os.getpid(),
        initial_indices=ids, stage=stage, phase=phase, calibration_sha256=cal_sha,
        plan_sha256=digest(output/'frozen/plan.json'), **extra))


def supervise(scope, stage):
    output, plan = load(scope)
    spec = stage_spec(plan, stage)
    calibration_for(output, plan, stage)
    execution = output/stage/'execution'
    ph = digest(output/'frozen/plan.json')
    review_counts, roles = Counter(), Counter()
    receipts = {}
    for phase in ('collect', 'review'):
        processes, handles = [], []
        try:
            for shard in range(spec['shards']):
                stdout = (execution/f'{phase}_{shard}.stdout.log').open('x', encoding='utf-8')
                stderr = (execution/f'{phase}_{shard}.stderr.log').open('x', encoding='utf-8')
                handles.extend((stdout, stderr))
                processes.append(subprocess.Popen([sys.executable, '-X', 'utf8', SCRIPT, 'worker',
                    scope, stage, phase, str(shard)], cwd=ROOT, stdout=stdout, stderr=stderr,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)))
            exclusive_json(execution/f'{phase}_launch.json', dict(utc=utc(), supervisor_pid=os.getpid(),
                worker_pids=[p.pid for p in processes], plan_sha256=ph, stage=stage))
            while True:
                codes = [p.poll() for p in processes]
                atomic_bytes(execution/'state.json', encode(dict(utc=utc(), phase=phase, exit_codes=codes,
                    worker_pids=[p.pid for p in processes], stage=stage, plan_sha256=ph)))
                if all(code is not None for code in codes):
                    break
                time.sleep(10)
            exclusive_json(execution/f'{phase}_exits.json', dict(utc=utc(),
                workers=[dict(pid=p.pid,exit_code=p.returncode) for p in processes]))
            if any(p.returncode != 0 for p in processes):
                raise RuntimeError('A worker failed; preserve all records, no automatic rerun')
            for shard, process in enumerate(processes):
                path = execution/f'{phase}_{shard}.complete.json'
                receipt = read(path)
                if (receipt['pid'] != process.pid or receipt['initial_indices'] != assigned(spec,shard)
                        or receipt['plan_sha256'] != ph or receipt['stage'] != stage or receipt['phase'] != phase):
                    raise ValueError('Worker completion identity differs')
                receipts[path.name] = digest(path)
                if phase == 'review':
                    review_counts.update(receipt['review_counts'])
                    roles.update(receipt['role_counts'])
        finally:
            for handle in handles:
                handle.close()
    if (roles['main'] != spec['main_runs'] or roles['observed_shadow'] != spec['observed_shadow_runs']
            or roles['deployment'] != spec['deployment_runs'] or review_counts['physical_traces'] != spec['main_runs']):
        raise ValueError('Complete experiment budget or review scope differs')
    block_hashes = {f'{i:05d}':digest(output/stage/'blocks'/f'{i:05d}'/'sha256.json') for i in range(spec['n_initials'])}
    exclusive_json(execution/'completed.json', dict(utc=utc(), stage=stage, plan_sha256=ph,
        role_counts=dict(roles), review_counts=dict(review_counts), receipt_sha256=receipts,
        block_manifest_sha256=block_hashes, actual_sumo_runs=spec['total_sumo_runs'],
        new_independent_initials=0 if stage=='preflight' else spec['n_initials']))


def watch(scope, stage):
    output, plan = load(scope)
    stage_spec(plan, stage)
    require_project_stopped(native_snapshot())
    calibration_for(output, plan, stage)
    execution = output/stage/'execution'
    execution.mkdir(parents=True, exist_ok=False)
    with (execution/'supervisor.stdout.log').open('x',encoding='utf-8') as out, (execution/'supervisor.stderr.log').open('x',encoding='utf-8') as err:
        process = subprocess.Popen([sys.executable,'-X','utf8',SCRIPT,'supervise',scope,stage],
            cwd=ROOT, stdout=out, stderr=err, creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        identity = dict(stage=stage, plan_sha256=digest(output/'frozen/plan.json'), supervisor_pid=process.pid)
        exclusive_json(execution/'watcher_launch.json', dict(utc=utc(), watcher_pid=os.getpid(), **identity))
        code = process.wait()
        exclusive_json(execution/'supervisor_exit.json', dict(utc=utc(), exit_code=code, **identity))
    if code != 0:
        raise RuntimeError('Native supervisor exited unsuccessfully; preserve the interrupted attempt')


def complete_stage(output, plan, stage):
    spec = stage_spec(plan, stage)
    directory = output/stage
    result = require_native_completion(directory, digest(output/'frozen/plan.json'), stage, spec['shards'])
    if (result['actual_sumo_runs'] != spec['total_sumo_runs']
            or result['review_counts']['physical_traces'] != spec['main_runs']
            or result['role_counts'] != dict(main=spec['main_runs'], observed_shadow=spec['observed_shadow_runs'],
                deployment=spec['deployment_runs'], deployed_adapter_mode_checks=72*spec['n_initials'] if spec['deployment'] else 0)
            or set(result['block_manifest_sha256']) != {f'{i:05d}' for i in range(spec['n_initials'])}):
        raise ValueError('The completed review does not cover the full registered schedule')
    for name, expected in result['receipt_sha256'].items():
        if digest(directory/'execution'/name) != expected:
            raise ValueError('Worker/review evidence changed')
    for name, expected in result['block_manifest_sha256'].items():
        if digest(directory/'blocks'/name/'sha256.json') != expected:
            raise ValueError('A reviewed block manifest changed')
    return result


def analyze(scope, stage):
    output, plan = load(scope)
    spec = stage_spec(plan, stage)
    require_project_stopped(native_snapshot())
    complete = complete_stage(output, plan, stage)
    calibration, cal_sha = calibration_for(output, plan, stage)
    timing, decisions = [], []
    def rows():
        for index in range(spec['n_initials']):
            directory, records = load_block(output, plan, stage, index, cal_sha)
            decision = read(directory/'decision.json')
            timing.append(dict(initial_index=index,
                full_prediction_wall_ms=decision['prediction']['full_prediction_wall_ms'],
                selection_wall_ms=decision['selection_wall_ms'],
                full_initial_wall_ms=read(directory/'complete.json')['full_initial_wall_ms']))
            decisions.append(decision['choices'])
            for key,row,_ in records:
                if key[-1]=='collection':
                    yield row
    data = paired_observations(rows(), profiles=profile_names(plan['family']), independent_initials=spec['n_initials'],
                               seed_base=spec['seed_base'], methods=spec['methods'])
    destination = output/stage/'analysis'
    destination.mkdir(exist_ok=False)
    if stage=='calibration':
        result = calibrate_observations(data['names'], data['adopted'], data['harmful'], data['differences'],
            data['failed'], family=plan['family'], seed_base=spec['seed_base'], attempt=plan['attempt'])
        result.update(execution_plan_sha256=digest(output/'frozen/plan.json'),
            complete_execution_sha256=digest(output/stage/'execution/completed.json'),
            native_exit_sha256=digest(output/stage/'execution/supervisor_exit.json'))
        exclusive_json(destination/'calibration.json', result)
        passed = result['primary_certificate']['all_fixed_models_both_targets_certified']
    else:
        result = evaluate_adapters(data, calibration, family=plan['family'], seeds=MODELS,
            seed_base=spec['seed_base'], mode_stream=22117, bootstrap_seed=22123,
            bootstrap_samples=5000 if stage=='confirmation' else 10)
        for selection in result['selections']:
            if decisions[selection['initial_index']][selection['adapter']][selection['training_seed']] != selection['profile']:
                raise ValueError('Observed-only deployed choices differ from offline evaluation')
        result.update(actual_deployment_replays_still_required=False, actual_deployment_runs=spec['deployment_runs'],
            calibration_sha256=cal_sha, execution_plan_sha256=digest(output/'frozen/plan.json'),
            all_observed_runtime_offline_choices_exact=True, independent_review_counts=complete['review_counts'])
        acceptance = (final_acceptance(calibration, result, plan['family']) if stage=='confirmation'
                      else dict(integration_only=True, new_independent_initials=0, risk_certified=False,
                                all_runtime_offline_profile_choices_exact=len(result['selections'])))
        # Preflight effectiveness estimates are deliberately not published as
        # research results. Only wiring and exact deployment identity matter.
        if stage=='confirmation':
            exclusive_json(destination/'comparison.json', result)
        exclusive_json(destination/'acceptance.json', acceptance)
        passed = acceptance.get('complete_primary_success', False)
    exclusive_json(destination/'runtime_accounting.json', dict(per_initial=timing,
        observed_shadow_runs=spec['observed_shadow_runs'], deployment_runs=spec['deployment_runs'],
        independent_initials_added_by_extra_roles=0, real_time_claim=False))
    exclusive_json(destination/'sha256.json', manifest(destination))
    print(json.dumps(dict(stage=stage, actual_sumo_runs=spec['total_sumo_runs'], primary_complete_success=passed,
                          calibration_only=stage=='calibration', preflight_only=stage=='preflight')), flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    freeze_parser = sub.add_parser('freeze')
    freeze_parser.add_argument('scope', choices=OUTPUTS)
    for command in ('watch', 'supervise', 'analyze', 'worker'):
        item = sub.add_parser(command)
        item.add_argument('scope', choices=OUTPUTS)
        item.add_argument('stage', choices=('preflight','calibration','confirmation'))
        if command=='worker':
            item.add_argument('phase', choices=('collect','review'))
            item.add_argument('shard', type=int)
    args = parser.parse_args()
    if args.command=='freeze':
        freeze(args.scope)
    elif args.command=='worker':
        worker(args.scope,args.stage,args.phase,args.shard)
    else:
        globals()[args.command](args.scope,args.stage)
