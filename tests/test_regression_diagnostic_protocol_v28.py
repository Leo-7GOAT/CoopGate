from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import diagnose_regression_v28 as diagnostic


def test_complete_selection_keeps_all_nominal_cases_and_shared_failure_initial():
    rows = diagnostic.read(diagnostic.OLD/'episodes.json')
    selected, failed = diagnostic.select_scope(rows)
    assert len(selected) == 640 and len(failed) == 4
    assert sum(r['learned_action_steps'] for r in selected) == 87
    assert {r['seed'] for r in failed} == {63401031, 63402025}
    assert sum(r['seed'] == 63402025 for r in failed) == 3
    assert all(r['learned_action_steps'] == 1 for r in failed)
    victim = next(r for r in rows if r['plant_name'] == 'nominal' and r['controller'] == 'phase05_1802')
    with pytest.raises(ValueError, match='All nominal'):
        diagnostic.select_scope([r for r in rows if r is not victim])
    with pytest.raises(ValueError, match='All nominal'):
        diagnostic.select_scope(rows+[deepcopy(victim)])


def test_compact_inputs_keep_physics_and_admitted_forecast_evidence():
    item = dict(t=.5, x=[[0., 0.]]*3, command=[0.]*3, disturbance=[[0.]*3]*5,
        innovation=[[0.]*3]*5, actuator_states=[[0.]*3]*5, realized_error_in_contract=True,
        nominal_joint_action_admitted=True, baseline_command=[0.]*3,
        selected_bank_differences=[[[0.]*2]*3]*33, selected_phase_differences=[[0.]*3]*33,
        repair_weight=.5, discarded_large_candidate_search='Original archive is retained')
    result = diagnostic.compact_trace([item])[0]
    assert 'discarded_large_candidate_search' not in result
    for key in ('command', 'innovation', 'actuator_states', 'selected_bank_differences', 'selected_phase_differences'):
        assert result[key] == item[key]
    item.pop('innovation')
    with pytest.raises(KeyError):
        diagnostic.compact_trace([item])


def test_replay_roles_include_original_policy_and_counterfactual_baseline():
    assert diagnostic.ROLES == ('baseline', 'original_phase05', 'single_action_then_baseline')


def test_oracle_must_match_actual_original_metrics_and_interpolated_motion():
    oracle = dict(both_branches_complete=True, realized_difference=[[0., 0.], [0., 0.], [.4375, .5]],
                  realized_phase_delta=[0., 0., .012834])
    actual = dict(original_cost_delta=[0., 0., .4375], logged_finish_delta=[0., 0., .5],
                  interpolated_crossing_delta=[0., 0., .012834])
    assert diagnostic.oracle_replay_check(oracle, actual)['original_cost_and_time_matched']
    broken = deepcopy(oracle)
    broken['realized_difference'][2][0] = -.05
    with pytest.raises(AssertionError):
        diagnostic.oracle_replay_check(broken, actual)
    oracle['both_branches_complete'] = False
    with pytest.raises(ValueError, match='did not complete'):
        diagnostic.oracle_replay_check(oracle, actual)
