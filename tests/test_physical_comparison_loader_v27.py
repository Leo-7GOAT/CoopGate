from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import analyze_physical_switch_v27 as analysis


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')


def completed_inputs(tmp_path):
    physical, common = tmp_path/'physical', tmp_path/'common'
    plan = dict(comparison_plan_sha256='physical-sha', adoption_plan_sha256='common-sha')
    launch = dict(launcher_pid=10, native_pid=11, plan_sha256='physical-sha')
    exit_ = dict(**launch, exit_code=0)
    save(physical/'full/execution/launch.json', launch)
    save(physical/'full/execution/exit.json', exit_)
    save(physical/'full/execution/collection_complete.json', dict(native_pid=11,
        plan_sha256='physical-sha', actual_sumo_runs=6912, blocks=288))
    save(physical/'full/analysis/verification.json', dict(stage='full', actual_sumo_runs=6912,
        plan_sha256='physical-sha', exit_receipt_sha256=analysis.digest(physical/'full/execution/exit.json')))
    save(common/'execution/start.json', dict(native_pid=20, plan_sha256='common-sha'))
    save(common/'execution/observation.json', dict(exit_code=0, native_pid=20, plan_sha256='common-sha'))
    save(common/'execution/collection_complete.json', dict(native_pid=20,
        plan_sha256='common-sha', old_episodes=18816, blocks=288))
    save(common/'analysis/report.json', dict(plan_sha256='common-sha', old_episodes=18816,
        exit_observation_sha256=analysis.digest(common/'execution/observation.json')))
    return plan, physical, common


def test_requires_both_actual_exits_and_committed_complete_reviews(tmp_path):
    plan, physical, common = completed_inputs(tmp_path)
    committed = []
    evidence = analysis.completion_gate(plan, physical, common, commit_check=committed.append)
    assert len(evidence) == 8
    assert committed == [physical/'full/analysis/verification.json', common/'analysis/report.json']


@pytest.mark.parametrize('name,field,value', [
    ('physical/full/execution/exit.json', 'exit_code', None),
    ('physical/full/execution/exit.json', 'exit_code', 1),
    ('physical/full/execution/exit.json', 'native_pid', 999),
    ('physical/full/execution/collection_complete.json', 'actual_sumo_runs', 144),
    ('physical/full/analysis/verification.json', 'stage', 'preflight'),
    ('common/execution/observation.json', 'exit_code', None),
    ('common/execution/observation.json', 'native_pid', 999),
    ('common/execution/collection_complete.json', 'old_episodes', 18815),
    ('common/execution/collection_complete.json', 'native_pid', 999),
    ('common/analysis/report.json', 'old_episodes', 18815),
    ('common/analysis/report.json', 'exit_observation_sha256', 'different')])
def test_partial_failed_substituted_receipts_are_rejected(tmp_path, name, field, value):
    plan, physical, common = completed_inputs(tmp_path)
    path = tmp_path/name
    data = analysis.read(path)
    data[field] = value
    save(path, data)
    with pytest.raises(ValueError):
        analysis.completion_gate(plan, physical, common, commit_check=lambda _:None)


def test_incomplete_inputs_prevent_any_outcome_read_or_report_directory(tmp_path, monkeypatch):
    plan = dict(comparison_plan_sha256='physical-sha', adoption_plan_sha256='common-sha')
    monkeypatch.setattr(analysis, 'inputs', lambda:plan)
    monkeypatch.setattr(analysis, 'OUT', tmp_path/'output')
    monkeypatch.setattr(analysis.physical, 'OUT', tmp_path/'physical')
    monkeypatch.setattr(analysis.common, 'OUT', tmp_path/'common')
    def forbidden(_):
        raise AssertionError('Read an incomplete outcome')
    monkeypatch.setattr(analysis, 'load_rows', forbidden)
    with pytest.raises(ValueError, match='Complete successful executions'):
        analysis.analyze()
    assert not (tmp_path/'output').exists()


def test_full_merged_grid_preserves_all_methods_models_and_plants():
    plan = analysis.read(analysis.physical.OUT/'plan.json')
    methods = analysis.method_lists(plan)
    assert len(methods['phase']) == 19 and len(methods['transfer']) == 25
    expected = {study:analysis.expected_cases(study, protocol['spec'], methods[study])
                for study, protocol in plan['protocols'].items()}
    assert len(expected['phase']) == 9728 and len(expected['transfer']) == 16000
    assert set(plan['models']) == {'1802', '1812', '1822'}
    for study in methods:
        assert set(plan['protocols'][study]['spec']['methods']) <= set(methods[study])
        for model in plan['models']:
            for rule in plan['rules']:
                assert f'{rule}_{model}' in methods[study]


def common_fixture():
    row = dict(controller='raw_1802', initial_index=0, initial=[[-80., 10.]]*3,
        control_steps=1, learned_action_steps=1, disturbance='clean',
        completed=True, aborted=False, collision=False, contract_violations=0,
        per_vehicle_finish=[10.]*3, per_vehicle_variation=[0.]*3, per_vehicle_cost=[10.]*3, team_cost=10.)
    case = (0, 'clean', 'raw_1802')
    fingerprint = analysis.common.object_hash(row)
    entry = dict(case=list(case), source_row_sha256=fingerprint, trace_path='old/trace.json', trace_sha256='trace')
    record = dict(original_row=row, source_row_sha256=fingerprint, source_trace_path='old/trace.json',
        source_trace_sha256='trace', original_learning_steps=1, common_learning_steps=0,
        common_episode_adopted=False)
    return dict(study='phase'), case, record, entry


def test_common_loader_preserves_source_and_distinct_event_counts():
    group, case, record, entry = common_fixture()
    result = analysis.validate_common_record(group, case, record, entry)
    assert result['common_learning_steps'] == 0 and result['original_learning_steps'] == 1
    assert 'common_learning_steps' not in record['original_row']


def test_self_consistent_modified_source_cannot_replace_frozen_observation():
    group, case, record, entry = common_fixture()
    record['original_row']['learned_action_steps'] = 0
    record['original_learning_steps'] = 0
    record['source_row_sha256'] = analysis.common.object_hash(record['original_row'])
    with pytest.raises(ValueError, match='frozen source row'):
        analysis.validate_common_record(group, case, record, entry)


def test_new_loader_reconstructs_every_completed_preflight_command_and_abort():
    plan = analysis.read(analysis.physical.OUT/'plan.json')
    executions = commands = failures = 0
    for group in analysis.physical.groups_for(plan, 'preflight'):
        path = analysis.physical.OUT/'preflight/blocks'/f"{group['block_index']:05d}"/'observations.sqlite'
        data = analysis.observations(path, analysis.physical.metadata('preflight', group), analysis.physical.schedule(group))
        for case, record, trace in data:
            row = analysis.validate_new_record(case, record, trace, group['study'])
            executions += 1
            commands += len(trace)
            failures += int(row['aborted'])
            if trace:
                corrupt = deepcopy(trace)
                corrupt[0]['actual_learning_used'] = not corrupt[0]['actual_learning_used']
                with pytest.raises(ValueError, match='actual-command adoption'):
                    analysis.validate_new_record(case, record, corrupt, group['study'])
    assert executions == 144 and commands == 3290 and failures == 2


def test_report_does_not_print_undefined_risk_as_zero():
    assert analysis.number(None) == 'undefined'
    assert analysis.number(0.) == '0.0000'
