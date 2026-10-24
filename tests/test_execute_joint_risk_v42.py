"""Prerequisite, isolation and exact-terminal barriers for risk inference."""
from pathlib import Path
import os
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import execute_joint_risk_v42 as execute


def terminal(stage='calibration'):
    n = execute.design.COUNTS[stage]
    common = dict(stage=stage, plan_sha256='plan', binding_sha256='binding')
    receipt = dict(**common, exit_code=0, native_pid=123)
    complete = dict(**common, native_pid=123, actual_sumo_runs=4*n, clusters=n)
    manifest = dict(**common, actual_sumo_runs=4*n, entries=[dict(index=i) for i in range(n)])
    return receipt, complete, manifest


@pytest.mark.parametrize('stage', execute.design.STAGES)
def test_exact_full_stage_terminal(stage):
    execute.validate_terminal(*terminal(stage), stage, 'plan', 'binding')


@pytest.mark.parametrize('part,key,value', [
    (0, 'exit_code', 1), (0, 'stage', 'confirmation'), (0, 'plan_sha256', 'other'),
    (0, 'binding_sha256', 'other'), (1, 'native_pid', 124),
    (1, 'actual_sumo_runs', 12287), (1, 'clusters', 3071),
    (1, 'stage', 'confirmation'), (2, 'actual_sumo_runs', 12287),
    (2, 'entries', [dict(index=0)]*3072), (2, 'entries', [dict(index=i) for i in range(3071)]),
])
def test_partial_failed_rebound_or_duplicate_stage_cannot_finalize(part, key, value):
    data = terminal()
    data[part][key] = value
    with pytest.raises(ValueError, match='complete zero-exit'):
        execute.validate_terminal(*data, 'calibration', 'plan', 'binding')


def process(pid, parent, command):
    return dict(ProcessId=pid, ParentProcessId=parent, Name='python.exe', CommandLine=command)


def test_isolation_allows_own_observed_chain_but_rejects_sibling_and_orphan_studies():
    ours = [process(1, 0, 'outside'), process(2, 1, execute.SCRIPT+' watch calibration'),
            process(3, 2, execute.SCRIPT+' worker calibration')]
    execute.require_quiet(ours, own_pid=3, parent_pid=2)
    for extra in [process(4, 2, 'sumo'), process(5, 99, 'python tools/recover_block_grids_v38.py worker v32')]:
        with pytest.raises(ValueError, match='remain live'):
            execute.require_quiet(ours+[extra], own_pid=3, parent_pid=2)


def test_previous_attempt_missing_final_analysis_fails_before_effect_access(tmp_path, monkeypatch):
    monkeypatch.setattr(execute.predecessor, 'OUTPUT', tmp_path)
    def forbidden():
        pytest.fail('Read old attempt before terminal output exists')
    monkeypatch.setattr(execute.predecessor, 'configure', forbidden)
    with pytest.raises(ValueError, match='no complete confirmation'):
        execute.verify_previous_attempt()


def test_confirmation_refuses_to_start_before_calibration_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(execute, 'OUT', tmp_path/'study')
    monkeypatch.setattr(execute, 'bound_inputs', lambda: ({}, {}))
    monkeypatch.setattr(execute, 'require_quiet', lambda *a: None)
    monkeypatch.setattr(execute.io, 'native_snapshot', lambda: [])
    monkeypatch.setattr(execute.io, 'digest', lambda *a: 'hash')
    def incomplete(*args):
        raise ValueError('Calibration not terminal')
    monkeypatch.setattr(execute, 'stage_terminal', incomplete)
    def forbidden(*args, **kwargs):
        pytest.fail('Spawned before calibration completed')
    monkeypatch.setattr(execute.subprocess, 'Popen', forbidden)
    with pytest.raises(ValueError, match='not terminal'):
        execute.watch('confirmation')
    assert not execute.OUT.exists()


def test_both_stages_required_before_loading_models_or_inferring(tmp_path, monkeypatch):
    monkeypatch.setattr(execute, 'OUT', tmp_path/'study')
    monkeypatch.setattr(execute, 'bound_inputs', lambda: ({}, {}))
    monkeypatch.setattr(execute, 'require_quiet', lambda *a: None)
    monkeypatch.setattr(execute.io, 'native_snapshot', lambda: [])
    monkeypatch.setattr(execute.io, 'digest', lambda *a: 'hash')
    seen = []
    def terminal(stage, *args):
        seen.append(stage)
        if stage == 'confirmation':
            raise ValueError('Confirmation not terminal')
        return 'cal-exit'
    monkeypatch.setattr(execute, 'stage_terminal', terminal)
    def forbidden(*args, **kwargs):
        pytest.fail('Inference/model access before both stages completed')
    monkeypatch.setattr(execute, 'load_models', forbidden)
    monkeypatch.setattr(execute, 'certificate_from_complete_records', forbidden)
    with pytest.raises(ValueError, match='not terminal'):
        execute.finalize()
    assert seen == ['calibration', 'confirmation']
    assert not execute.OUT.exists()


def test_real_fixed_models_load_without_path_subscript_errors():
    plan = dict(models=execute.design.shared.load_models(ROOT))
    assert tuple(execute.load_models(plan)) == ('1802', '1812', '1822')


def test_real_memoization_completion_is_bound_without_retraining():
    hashes = execute.verify_memoization()
    assert len(hashes) == 3
    assert all(execute.io.digest(ROOT/name) == sha for name, sha in hashes.items())


def test_reset_preflight_covers_every_draw_without_rejection():
    plan = dict(stages={s: execute.design.stage_cases(s) for s in execute.design.STAGES})
    assert execute.reset_domain(plan) == dict(checked_initials=4096, reset_domain_failures=[],
                                            rejected_or_replaced_initials=0, outcome_free=True)


@pytest.mark.parametrize('change', ['live', 'hash', 'missing', 'launch', 'inventory'])
def test_stage_receipts_are_checked_before_accepting_completion(tmp_path, monkeypatch, change):
    monkeypatch.setattr(execute, 'OUT', tmp_path)
    directory = tmp_path/'calibration/execution'
    directory.mkdir(parents=True)
    receipt, complete, manifest = terminal()
    identity = dict(native_pid=123, launcher_pid=122, native_creation_date='original', command_line='worker')
    receipt.update(identity)
    execute.io.exclusive_json(directory/'blocks_manifest.json', manifest)
    complete['manifest_sha256'] = execute.io.digest(directory/'blocks_manifest.json')
    execute.io.exclusive_json(directory/'collection_complete.json', complete)
    execute.io.exclusive_json(directory/'launch.json', identity if change != 'launch' else {**identity, 'native_pid': 999})
    for name in ('spawn.json', 'worker.native.json', 'worker.acknowledged.json'):
        execute.io.exclusive_json(directory/name, {})
    receipt['hashes'] = {n: execute.io.digest(directory/n) for n in execute.TERMINAL_FILES}
    if change == 'missing':
        receipt['hashes'].pop('spawn.json')
    execute.io.exclusive_json(directory/'exit.json', receipt)
    if change == 'hash':
        (directory/'worker.native.json').write_text('changed')
    monkeypatch.setattr(execute.io, 'check_acknowledgment', lambda *a: None)
    monkeypatch.setattr(execute.io, 'native_snapshot', lambda: [dict(ProcessId=123, CreationDate='original')]
                        if change == 'live' else [])
    def inventory(*args):
        if change == 'inventory':
            return dict(index=0, files={'changed': True})
        pytest.fail('Block access before invalid terminal chain rejected')
    monkeypatch.setattr(execute, 'block_hashes', inventory)
    with pytest.raises(ValueError):
        execute.stage_terminal('calibration', 'plan', 'binding')


@pytest.mark.skipif(os.name != 'nt', reason='Native Windows process identity integration')
def test_watcher_keeps_real_native_nonzero_exit_without_restarting(tmp_path, monkeypatch):
    out = tmp_path/'native_failure'
    plan, binding = tmp_path/'plan.json', tmp_path/'binding.json'
    execute.io.exclusive_json(plan, {})
    execute.io.exclusive_json(binding, {})
    monkeypatch.setattr(execute, 'OUT', out)
    monkeypatch.setattr(execute, 'PLAN', plan)
    monkeypatch.setattr(execute, 'BINDING', binding)
    monkeypatch.setattr(execute, 'bound_inputs', lambda: ({}, {}))
    monkeypatch.setattr(execute, 'require_quiet', lambda *a: None)
    original = execute.subprocess.Popen
    spawned = []
    execution = out/'calibration/execution'
    code = (
        'import json,os,time; from pathlib import Path; '
        f'd=Path({str(execution)!r}); '
        '(d/"worker.native.json").write_text(json.dumps(dict(native_pid=os.getpid(),parent_pid=os.getppid()))); '
        'deadline=time.monotonic()+20\n'
        'while not (d/"worker.acknowledged.json").exists() and time.monotonic()<deadline: time.sleep(.05)\n'
        'raise SystemExit(7 if (d/"worker.acknowledged.json").exists() else 99)\n')
    def spawn(command, *args, **kwargs):
        if isinstance(command, list) and execute.SCRIPT in command and 'worker' in command:
            command = [sys.executable, '-c', code, execute.SCRIPT, 'worker', 'calibration']
            child = original(command, *args, **kwargs)
            spawned.append(child)
            return child
        return original(command, *args, **kwargs)  # Preserve the real CIM snapshot subprocess.
    monkeypatch.setattr(execute.subprocess, 'Popen', spawn)
    with pytest.raises(SystemExit) as stopped:
        execute.watch('calibration')
    assert stopped.value.code == 7
    assert len(spawned) == 1 and spawned[0].poll() == 7
    receipt = execute.io.read(execution/'exit.json')
    assert receipt['exit_code'] == 7 and receipt['launcher_pid'] == spawned[0].pid
    execute.io.check_acknowledgment(execution, 'worker', receipt['launcher_pid'], receipt['native_pid'])
    assert receipt['native_creation_date']
    assert not (execution/'collection_complete.json').exists()
    with pytest.raises(FileExistsError):
        execute.watch('calibration')
    assert len(spawned) == 1
