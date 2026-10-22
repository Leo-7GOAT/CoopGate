"""Verify observed-state deployment choices on the completed historical integration.

Synthetic calibration configurations exercise wiring only; no risk certificate
or effectiveness statistics are produced. Ten real shadow replays use one old
initial and the fixed repair family, independent of pending V21 study results.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import torch

from coopgate.episode_store_v21 import EpisodeStore, observations
from coopgate.prospective_data_v21 import paired_observations, evaluate_adapters
from coopgate.prospective_validation_v21 import FAMILIES, MODELS, profile_names
from observable_profile_prediction_v22 import forecast, prediction_from_rows, choose_profiles, verify_clean_identity
from probe_disturbed_gate_v19 import digest, save
import recover_interrupted_v21 as recovery
import recover_runtime_preflight_v21 as integration
from prospective_runtime_v21 import method_definitions
from start_final_v19 import require_committed, verify_manifest

ROOT = integration.ROOT
OUTPUT = ROOT / 'artifacts/v21/observable_profile_audit_schemafix_20261022'
FROZEN = OUTPUT / 'frozen'
EXECUTION = OUTPUT / 'execution'
RESULTS = OUTPUT / 'results'
SOURCES = ('tools/observable_profile_prediction_v22.py', 'tools/audit_observable_profile_prediction_v22.py',
           'tests/test_observable_profile_prediction_v22.py')


def completed_integration():
    plan, old = integration.inputs()
    receipt = recovery.read(integration.EXECUTION / 'supervisor_exit.json')
    launch = recovery.read(integration.EXECUTION / 'watcher_launch.json')
    finished = recovery.read(integration.EXECUTION / 'completed.json')
    if (receipt['exit_code'] != 0 or receipt['supervisor_pid'] != launch['supervisor_pid']
            or receipt['plan_sha256'] != digest(integration.FROZEN / 'plan.json')
            or launch['plan_sha256'] != receipt['plan_sha256']
            or finished['audit_sha256'] != digest(integration.OUTPUT / 'audit.json')):
        raise ValueError('The actual integration must have completed successfully')
    audit = recovery.read(integration.OUTPUT / 'audit.json')
    if (audit['unique_original_planned_roles'] != 144 or audit['collection_reference_pairs'] != 64
            or audit['observed_shadow_runs'] != 16 or not audit['all_reference_pairs_exact']
            or not audit['all_observed_shadow_commands_and_outcomes_match']
            or digest(integration.OUTPUT / 'complete.sqlite') != audit['complete_database_sha256']):
        raise ValueError('Incomplete or changed actual integration evidence')
    return plan, old


def freeze():
    if OUTPUT.exists():
        raise FileExistsError('Preserve prior observable audit')
    completed_integration()
    require_committed([ROOT / name for name in SOURCES])
    FROZEN.mkdir(parents=True)
    plan = dict(utc=recovery.utc(), actual_sumo_runs=10, new_independent_initials=0,
        new_calibration_data=False, new_risk_certified=False, actual_forecast_family='repair',
        historical_initial=63100001, comparison_families=list(FAMILIES),
        previous_failed_audit_exit_sha256=digest(ROOT / 'artifacts/v21/observable_profile_audit_20261022/execution/supervisor_exit.json'),
        correction='Honor fixed engine 19367 and observed-state label; derive cost from raw shadow arrival and variation',
        fixtures='q=0, q=0.1, infinite q; synthetic calibrated choices test wiring only',
        source_sha256={name:digest(ROOT / name) for name in SOURCES},
        integration_audit_sha256=digest(integration.OUTPUT / 'audit.json'),
        integration_database_sha256=digest(integration.OUTPUT / 'complete.sqlite'))
    save(FROZEN / 'plan.json', plan)
    for name in SOURCES:
        path = FROZEN / 'source' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, path)
    save(FROZEN / 'sha256.json', integration.original.manifest(FROZEN))
    print(json.dumps(dict(actual_sumo_runs=10, new_independent_initials=0)), flush=True)


def inputs():
    verify_manifest(FROZEN)
    plan = recovery.read(FROZEN / 'plan.json')
    require_committed([FROZEN / 'plan.json', FROZEN / 'sha256.json', *[ROOT/name for name in SOURCES]])
    _, old = completed_integration()
    for name, expected in plan['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen observed-only predictor changed')
    if (digest(integration.OUTPUT / 'audit.json') != plan['integration_audit_sha256']
            or digest(integration.OUTPUT / 'complete.sqlite') != plan['integration_database_sha256']):
        raise ValueError('Completed integration changed')
    return plan, old


def synthetic_calibration(family, q):
    names = profile_names(family)
    return dict(primary_family=family, common_profiles=list(names),
        conformal=dict(infinite_q=not np.isfinite(q), q=float(q) if np.isfinite(q) else None, calibration_initials=4096),
        ltt_selection={str(s):'baseline' if j == 1 else family+'_'+str(s) for j,s in enumerate(MODELS)},
        mean_spi_selection={str(s):('nominal' if j == 2 else 'raw')+'_'+str(s) for j,s in enumerate(MODELS)})


def run():
    torch.set_num_threads(1)
    plan, old = inputs()
    RESULTS.mkdir(exist_ok=False)
    metadata = dict(role='complete_recovered_integration', plan_sha256=digest(integration.FROZEN / 'plan.json'))
    expected = [key for shard in range(3) for key in integration.role_schedule(integration.original.cases(old, shard))]
    archived = {key:(row, trace) for key,row,trace in observations(integration.OUTPUT / 'complete.sqlite', metadata, expected)}
    observed = np.asarray(archived[(0, 'clean', 'baseline', 'collection')][0]['initial'])
    selection_checks = 0
    for family in plan['comparison_families']:
        names = ('baseline', *profile_names(family))
        shadows = {n:archived[(0, 'clean', n, 'observed_shadow')][0] for n in names}
        clean = {n:archived[(0, 'clean', n, 'collection')][0] for n in names}
        prediction = prediction_from_rows(observed, family, shadows)
        verify_clean_identity(prediction, clean)
        rows = [archived[(0, mode, n, 'collection')][0] for mode in old['protocol']['spec']['modes'] for n in names]
        data = paired_observations(rows, profiles=profile_names(family), independent_initials=1,
            seed_base=plan['historical_initial'], methods=method_definitions(family))
        for q in (0., .1, np.inf):
            calibration = synthetic_calibration(family, q)
            choices = choose_profiles(prediction, calibration, family)
            offline = evaluate_adapters(data, calibration, family=family, seeds=MODELS,
                seed_base=plan['historical_initial'], mode_stream=22117, bootstrap_seed=22123, bootstrap_samples=10)
            for selection in offline['selections']:
                if choices[selection['adapter']][selection['training_seed']] != selection['profile']:
                    raise ValueError('Observed-only runtime and offline comparison disagree')
                selection_checks += 1
    family = plan['actual_forecast_family']
    protocol = dict(models=old['protocol']['models'], spec={**old['protocol']['spec'], 'methods':method_definitions(family)})
    metadata = dict(role='observed_only_predictor_replay', plan_sha256=digest(FROZEN / 'plan.json'))
    store = EpisodeStore(RESULTS / 'predictions.sqlite', metadata)
    keys = []
    def retain(name, row, trace):
        key = (family, name, 'observed_shadow')
        store.append(key, row, trace)
        keys.append(key)
        old_row, old_trace = archived[(0, 'clean', name, 'observed_shadow')]
        recovery.compare_replay(old_row, row, old_trace, trace)
        print(json.dumps(dict(actual_sumo_runs=len(keys), expected=10)), flush=True)
    try:
        prediction = forecast(observed, protocol, family, on_record=retain)
        expected = [(family, n, 'observed_shadow') for n in ('baseline', *profile_names(family))]
        store.seal(expected)
        if sum(1 for _ in observations(RESULTS / 'predictions.sqlite', metadata, expected)) != 10:
            raise ValueError('Incomplete actual predictor execution')
    finally:
        store.close()
    clean = {n:archived[(0, 'clean', n, 'collection')][0] for n in ('baseline', *profile_names(family))}
    verify_clean_identity(prediction, clean)
    inputs()
    save(RESULTS / 'prediction.json', prediction)
    save(RESULTS / 'audit.json', dict(actual_sumo_runs=10, new_independent_initials=0, risk_certified=False,
        synthetic_calibration_configurations_only=True, actual_calibration_or_final_data_generated=False,
        exact_actual_prediction_replays=10, complete_families_checked=3,
        all_runtime_offline_profile_choices_exact=selection_checks, full_forecast_wall_ms=prediction['full_prediction_wall_ms'],
        plan_sha256=digest(FROZEN / 'plan.json'), integration_audit_sha256=plan['integration_audit_sha256']))
    save(RESULTS / 'sha256.json', integration.original.manifest(RESULTS))
    save(EXECUTION / 'completed.json', dict(utc=recovery.utc(), audit_sha256=digest(RESULTS / 'audit.json')))


def watch():
    inputs()
    EXECUTION.mkdir(exist_ok=False)
    command = [sys.executable, '-X', 'utf8', 'tools/audit_observable_profile_prediction_v22.py', 'run']
    with (EXECUTION / 'supervisor.stdout.log').open('x') as out, (EXECUTION / 'supervisor.stderr.log').open('x') as err:
        process = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        save(EXECUTION / 'watcher_launch.json', dict(utc=recovery.utc(), watcher_pid=os.getpid(),
            supervisor_pid=process.pid, command=command, plan_sha256=digest(FROZEN / 'plan.json')))
        code = process.wait()
        save(EXECUTION / 'supervisor_exit.json', dict(utc=recovery.utc(), supervisor_pid=process.pid,
            exit_code=code, plan_sha256=digest(FROZEN / 'plan.json')))
    if code:
        raise SystemExit(code)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['freeze', 'watch', 'run'])
    globals()[parser.parse_args().command]()
