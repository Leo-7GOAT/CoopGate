"""Recovery tests use only synthetic identities, never partial efficacy."""
import ast
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import recover_memoized_recovery_v38 as recovery
from coopgate.episode_store_v21 import EpisodeStore


def groups():
    result = []
    for index in range(recovery.TOTAL_BLOCKS):
        case = [index,'scene',index,70000000+index,'nominal','bank_performance','1802']
        order = ('original','memoized') if index%2 == 0 else ('memoized','original')
        result.append(dict(block_index=index,case=case,
            executions=[case+[implementation] for implementation in order]))
    return result


def test_partition_keeps_exact_sealed_prefix_and_recovers_every_missing_pair():
    retained,missing = recovery.partition_groups(dict(groups=groups()))
    assert [group['block_index'] for group in retained] == list(range(74))
    assert [group['block_index'] for group in missing] == list(range(74,180))
    assert len(missing) == 106 and sum(len(group['executions']) for group in missing) == 212


@pytest.mark.parametrize('change',('short','gap','duplicate','one_execution'))
def test_partition_rejects_changed_or_incomplete_original_scope(change):
    schedule = groups()
    if change == 'short': schedule.pop()
    if change == 'gap': schedule[74]['block_index'] = 75
    if change == 'duplicate': schedule[74] = schedule[73]
    if change == 'one_execution': schedule[74]['executions'].pop()
    with pytest.raises(ValueError):
        recovery.partition_groups(dict(groups=schedule))


def test_old_process_chain_must_be_absent_and_terminal_receipts_forbid_recovery(tmp_path):
    prior = tmp_path/'prior'; execution = prior/'execution'; execution.mkdir(parents=True)
    (prior/'watcher_launch.json').write_text(json.dumps(dict(launcher_pid=11)),encoding='utf-8')
    (execution/'launch.json').write_text(json.dumps(dict(watcher_pid=12,launcher_pid=13,native_pid=14)),encoding='utf-8')
    proof = recovery.require_interrupted_processes_absent(prior,processes=[])
    assert proof['old_process_ids'] == [11,12,13,14] and proof['original_exit_status'] == 'unknown'
    with pytest.raises(ValueError,match='still be live'):
        recovery.require_interrupted_processes_absent(prior,processes=[dict(ProcessId=14,CommandLine='python')])
    with pytest.raises(ValueError,match='still be live'):
        recovery.require_interrupted_processes_absent(prior,processes=[dict(ProcessId=99,CommandLine='python audit_memoized_recovery_v37.py worker')])
    (execution/'exit.json').write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='terminal V37 receipt'):
        recovery.require_interrupted_processes_absent(prior,processes=[])


def make_block(path,metadata,cases,complete):
    path.mkdir(parents=True)
    store = EpisodeStore(path/'observations.sqlite',metadata)
    for case in cases:
        store.append(case,dict(row={'case':case},calls=[]),[{'command':[0,0,0]}])
    if complete:
        store.seal(cases)
        recovery.exclusive_json(path/'complete.json',dict(**metadata,
            reviews=[dict(case=list(case)) for case in cases],
            database_sha256=recovery.digest(path/'observations.sqlite'),
            seal_sha256=recovery.digest(path/'observations.sqlite.seal.json')))
    else:
        store.close()


def test_block_identity_accepts_complete_and_partial_prefix_without_decoding_outcomes(tmp_path):
    metadata = dict(plan_sha256='plan',block_index=0,stage='exact_memoization')
    cases = [('a','original'),('a','memoized')]
    complete = tmp_path/'complete';make_block(complete,metadata,cases,True)
    result = recovery.block_identity(complete,metadata,cases,complete=True)
    assert result['records'] == 2 and result['complete']
    partial = tmp_path/'partial';make_block(partial,metadata,cases[:1],False)
    result = recovery.block_identity(partial,metadata,cases,complete=False)
    assert result['records'] == 1 and not result['complete']
    assert set(result['files']) == {'observations.sqlite'}


@pytest.mark.parametrize('change',('metadata','schedule','partial_seal','bad_proof'))
def test_block_identity_rejects_changed_or_partially_sealed_evidence(tmp_path,change):
    metadata = dict(plan_sha256='plan',block_index=0,stage='exact_memoization')
    cases = [('a','original'),('a','memoized')]
    directory = tmp_path/'block';make_block(directory,metadata,cases,change != 'partial_seal')
    expected_metadata = dict(metadata)
    expected_cases = list(cases)
    complete = change != 'partial_seal'
    if change == 'metadata': expected_metadata['stage'] = 'changed'
    if change == 'schedule': expected_cases[1] = ('different','memoized')
    if change == 'partial_seal': (directory/'observations.sqlite.seal.json').write_text('{}',encoding='utf-8')
    if change == 'bad_proof':
        proof = recovery.read(directory/'complete.json');proof['reviews'].pop()
        (directory/'complete.json').write_text(json.dumps(proof),encoding='utf-8')
    with pytest.raises(ValueError):
        recovery.block_identity(directory,expected_metadata,expected_cases,complete=complete)


def test_full_coverage_requires_each_frozen_implementation_case_once():
    schedule = groups()
    cases = [dict(case=key) for group in schedule for key in group['executions']]
    assert recovery.require_full_coverage(cases,schedule)
    with pytest.raises(ValueError): recovery.require_full_coverage(cases[:-1],schedule)
    changed = [dict(case=list(item['case'])) for item in cases]
    changed[-1]['case'] = list(changed[0]['case'])
    with pytest.raises(ValueError): recovery.require_full_coverage(changed,schedule)


def test_worker_and_final_summary_reuse_v37_verification_and_exclude_partial_block():
    tree = ast.parse(Path(recovery.__file__).read_text(encoding='utf-8'))
    calls = [node for node in ast.walk(tree) if isinstance(node,ast.Call)]
    attributes = [(getattr(node.func.value,'id',None),node.func.attr)
                  for node in calls if isinstance(node.func,ast.Attribute)]
    assert attributes.count(('source','verify')) >= 2
    assert ('source','non_timing') in attributes
    text = Path(recovery.__file__).read_text(encoding='utf-8')
    assert "plan['missing_groups']" in text
    assert "partial_block_reused_for_results=False" in text
    assert "excluded_partial_records_are_engineering_only=True" in text
