"""Reject incomplete or altered terminal studies before any effect analysis."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import execute_fair_comparison_v41 as execute


def terminal():
    receipt = dict(exit_code=0, native_pid=123, plan_sha256='plan', binding_sha256='binding')
    done = dict(native_pid=123, plan_sha256='plan', binding_sha256='binding',
                actual_sumo_runs=5120, blocks=160)
    manifest = dict(plan_sha256='plan', binding_sha256='binding', actual_sumo_runs=5120,
                    entries=[dict(block_index=i) for i in range(160)])
    return receipt, done, manifest


def test_exact_terminal_schedule_is_required():
    execute.validate_terminal(*terminal(), 'plan', 'binding')


@pytest.mark.parametrize('part,key,value', [
    (0, 'exit_code', 1), (0, 'plan_sha256', 'other'), (0, 'binding_sha256', 'other'),
    (1, 'native_pid', 456), (1, 'actual_sumo_runs', 5119), (1, 'blocks', 159),
    (1, 'plan_sha256', 'other'), (1, 'binding_sha256', 'other'),
    (2, 'actual_sumo_runs', 5119), (2, 'plan_sha256', 'other'),
    (2, 'binding_sha256', 'other'),
    (2, 'entries', [dict(block_index=i) for i in range(159)]),
    (2, 'entries', [dict(block_index=i) for i in reversed(range(160))]),
    (2, 'entries', [dict(block_index=0) for _ in range(160)]),
])
def test_partial_failed_reordered_and_rebound_studies_rejected(part, key, value):
    inputs = terminal()
    inputs[part][key] = value
    with pytest.raises(ValueError, match='complete zero-exit'):
        execute.validate_terminal(*inputs, 'plan', 'binding')


def test_bound_inventory_rejects_changes_but_allows_unrelated_additions(tmp_path, monkeypatch):
    monkeypatch.setattr(execute, 'ROOT', tmp_path)
    source = tmp_path/'controller.py'
    source.write_text('original')
    binding = dict(source_sha256={'controller.py': execute.io.digest(source)})
    (tmp_path/'unrelated.py').write_text('new')
    execute.verify_sources(binding)
    source.write_text('changed')
    with pytest.raises(ValueError, match='Bound source changed'):
        execute.verify_sources(binding)


@pytest.mark.parametrize('mutation', ['live', 'hash', 'missing_hash', 'manifest', 'ack'])
def test_terminal_chain_checks_precede_model_loading_or_analysis(tmp_path, monkeypatch, mutation):
    out = tmp_path/'study'
    execution = out/'execution'
    execution.mkdir(parents=True)
    monkeypatch.setattr(execute, 'OUT', out)
    monkeypatch.setattr(execute, 'PLAN', tmp_path/'plan.json')
    monkeypatch.setattr(execute, 'BINDING', tmp_path/'binding.json')
    execute.io.exclusive_json(execute.PLAN, {})
    execute.io.exclusive_json(execute.BINDING, {})
    plan_sha, binding_sha = execute.io.digest(execute.PLAN), execute.io.digest(execute.BINDING)
    receipt, done, manifest = terminal()
    for item in (receipt, done, manifest):
        item.update(plan_sha256=plan_sha, binding_sha256=binding_sha)
    receipt.update(launcher_pid=12, native_creation_date='original-creation')
    execute.io.exclusive_json(execution/'blocks_manifest.json', manifest)
    done['blocks_manifest_sha256'] = (execute.io.digest(execution/'blocks_manifest.json')
                                      if mutation != 'manifest' else 'changed')
    execute.io.exclusive_json(execution/'collection_complete.json', done)
    for name in ('spawn.json', 'launch.json', 'worker.native.json', 'worker.acknowledged.json'):
        execute.io.exclusive_json(execution/name, {})
    receipt['hashes'] = {p.name: execute.io.digest(p) for p in execution.glob('*.json')}
    if mutation == 'missing_hash':
        del receipt['hashes']['launch.json']
    execute.io.exclusive_json(execution/'exit.json', receipt)
    if mutation == 'hash':
        (execution/'launch.json').write_text('changed')
    monkeypatch.setattr(execute, 'bound_inputs', lambda: ({}, {}))
    def ack(*args):
        if mutation == 'ack':
            raise ValueError('Changed acknowledgment')
    monkeypatch.setattr(execute.io, 'check_acknowledgment', ack)
    monkeypatch.setattr(execute.io, 'native_snapshot', lambda: [
        dict(ProcessId=123, CreationDate='original-creation')])
    def forbidden(*args):
        pytest.fail('Model/effect access occurred before terminal chain validation')
    monkeypatch.setattr(execute, 'load_inputs', forbidden)
    with pytest.raises(ValueError):
        execute.summarize()
    assert not (out/'analysis').exists()


def test_source_inventory_includes_executable_runtime_and_tests():
    paths = {p.relative_to(ROOT).as_posix() for p in execute.source_paths()}
    assert {execute.SCRIPT, execute.TEST, 'tools/run_fair_runtime_v41.py',
            'src/coopgate/fair_runtime_v41.py', 'tools/execute_ood_validation_v39.py'} <= paths
