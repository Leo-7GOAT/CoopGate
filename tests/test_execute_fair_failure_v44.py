from copy import deepcopy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import execute_fair_failure_v44 as execute
import run_fair_runtime_v41 as runner
from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.disturbance_v9 import disturbance_tape
from coopgate.episode_store_v21 import observations
from coopgate.fair_comparison_v39 import CaseIdentity, METHODS, fair_schedule
from coopgate.fair_failure_analysis_v44 import summarize_matched_records
from coopgate.matched_shift_v21 import matched_plant_parameters


def test_private_execution_keeps_frozen_bytecode_and_globals_unchanged():
    for name in ('watch', 'worker', 'summarize'):
        original = getattr(execute.original, name)
        copied = execute.execution_function(name)
        assert copied.__code__ is original.__code__
        assert copied.__globals__ is not original.__globals__
        assert copied.__globals__['OUT'] == execute.OUT
        assert copied.__globals__['SCRIPT'] == execute.SCRIPT
        assert copied.__globals__['bound_inputs'] is execute.bound_inputs
        assert copied.__globals__['runtime'] is original.__globals__['runtime']
        assert copied.__globals__['summarize_matched_records'] is execute.final_analysis
        assert original.__globals__['OUT'] == execute.original.OUT
        assert original.__globals__['summarize_matched_records'] is not execute.final_analysis


@pytest.mark.parametrize('evidence', ['execution', 'blocks', 'analysis', 'live'])
def test_no_pre_outcome_revision_after_old_execution(tmp_path, monkeypatch, evidence):
    monkeypatch.setattr(execute.original, 'OUT', tmp_path)
    processes = []
    if evidence == 'live':
        processes = [dict(CommandLine='python '+execute.original.SCRIPT+' worker')]
    else:
        (tmp_path/evidence).mkdir()
    monkeypatch.setattr(execute.io, 'native_snapshot', lambda: processes)
    with pytest.raises(ValueError, match='Original formal study'):
        execute.require_original_unstarted()


@pytest.mark.parametrize('command', ['watch', 'worker', 'summarize'])
def test_live_project_jobs_rejected_before_dispatch(monkeypatch, command):
    monkeypatch.setattr(execute.io, 'native_snapshot', lambda: [])
    def reject(*args):
        raise ValueError('Other project study processes remain live')
    monkeypatch.setattr(execute.isolation, 'require_quiet', reject)
    monkeypatch.setattr(execute, 'execution_function', lambda *args: pytest.fail('must not dispatch'))
    with pytest.raises(ValueError, match='Other project'):
        getattr(execute, command)()


@pytest.mark.parametrize('mutation', ['contract', 'sample', 'source'])
def test_revision_cannot_change_controller_schedule_or_failure_rule(tmp_path, monkeypatch, mutation):
    previous = dict(cases=[dict(case_id='original')], actual_sumo_runs=5120)
    expected = dict(contract='fixed', source_sha256={'new.py':'fixed'})
    plan = deepcopy(previous)
    plan['analysis_revision'] = deepcopy(expected)
    if mutation == 'contract':
        plan['analysis_revision']['contract'] = 'drop failures'
    elif mutation == 'sample':
        plan['cases'][0]['case_id'] = 'replacement'
    else:
        plan['analysis_revision']['source_sha256']['new.py'] = 'changed'
    target = tmp_path/'plan.json'
    execute.io.exclusive_json(target, plan)
    monkeypatch.setattr(execute, 'PLAN', target)
    monkeypatch.setattr(execute.io, 'committed', lambda *args: None)
    monkeypatch.setattr(execute, 'require_original_unstarted', lambda: None)
    monkeypatch.setattr(execute.original, 'bound_inputs', lambda: (previous, {}))
    monkeypatch.setattr(execute, 'revision', lambda: expected)
    with pytest.raises(ValueError, match='Changed analysis'):
        execute.plan_inputs()


def test_finalization_rejects_terminal_rewrite_during_replay(tmp_path, monkeypatch):
    directory = tmp_path/'execution'
    directory.mkdir()
    names = ('exit.json', 'spawn.json', 'launch.json', 'worker.native.json',
             'worker.acknowledged.json', 'collection_complete.json', 'blocks_manifest.json')
    for name in names:
        execute.io.exclusive_json(directory/name, {})
    monkeypatch.setattr(execute, 'OUT', tmp_path)
    monkeypatch.setattr(execute.io, 'native_snapshot', lambda: [])
    monkeypatch.setattr(execute.isolation, 'require_quiet', lambda *args: None)
    monkeypatch.setattr(execute, 'final_analysis', lambda *args: pytest.fail('must not report'))
    def function(name, finalizer):
        def replay():
            (directory/'exit.json').write_text('{"rewritten":true}')
            finalizer([], None)
        return replay
    monkeypatch.setattr(execute, 'execution_function', function)
    with pytest.raises(ValueError, match='Terminal identities changed'):
        execute.summarize()


@pytest.mark.skipif(os.environ.get('COOPGATE_RUN_SUMO_SMOKE') != '1', reason='Opt-in actual SUMO')
@pytest.mark.parametrize('broken', [False, True])
def test_real_full_family_persists_replays_and_reaches_new_analysis(tmp_path, broken):
    torch.set_num_threads(1)
    plan = execute.io.read(execute.original.PLAN)
    seed, scene, plant, mode, model_id = 139_885_001+int(broken), 'nominal', 'lag_strong', 'clean', '1802'
    design = execute.design
    initial = design.planned_initial(seed, scene)
    case = CaseIdentity(case_id='development-v44', independent_cluster='development-v44',
        policy_id=model_id, policy_sha256=plan['models'][model_id]['sha256'],
        scenario=scene, initial_id=seed, dynamics=plant, disturbance=mode,
        initial_sha256=design.array_sha256(initial, decimals=8),
        dynamics_sha256=design.canonical_sha256(matched_plant_parameters(seed, plant, .1)),
        disturbance_sha256=design.array_sha256(disturbance_tape(seed, mode, .1, 250)))
    class BrokenPolicy:
        def proposal(self, *args):
            raise RuntimeError('Injected shared pre-adoption proposal failure')
    model = BrokenPolicy() if broken else load_coupled(ROOT/plan['models'][model_id]['path'])[0]
    # Isolate mathematical/failure replay from development-machine timing.
    spec = design.protocol_from_dict(plan['protocol'])
    spec = replace(spec, budget=replace(spec.budget, decision_timeout_ms=1e9))
    plan['protocol'], plan['cases'] = asdict(spec), [asdict(case)]
    group = dict(block_index=0, case_ids=[case.case_id], actual_sumo_runs=8,
        assignments=[dict(case_id=a.case.case_id, method=a.method, order=a.order)
                     for a in fair_schedule([case])])
    calibrations = execute.io.read(ROOT/plan['calibration']['path'])
    target = tmp_path/'actual_sumo'
    result = runner.collect_block(plan, group, target, {model_id:model}, calibrations,
                                  'development-v44', 'development-v44')
    assert result['actual_sumo_runs'] == 8
    metadata = dict(plan_sha256='development-v44', binding_sha256='development-v44', block_index=0)
    stored = list(observations(target/'observations.sqlite', metadata, runner.block_schedule(group)))
    records = [runner.decode_episode(body['episode']) for _,body,_ in stored]
    report = summarize_matched_records(records, spec)
    assert report['cases'] == report['independent_clusters'] == 1
    assert report['deleted_records'] == 0
    if broken:
        assert report['selected_method'] is None
        for record, (_, body, trace) in zip(records, stored):
            assert record.aborted and not record.adopted and not trace
            assert record.per_vehicle_cost == (25.,)*3
            assert len(body['calls']) == 1
            row = report['methods'][record.method]
            assert row['failed_before_adoption_cases'] == row['baseline_failed_cases'] == 1
            assert not row['comparison_eligible']
