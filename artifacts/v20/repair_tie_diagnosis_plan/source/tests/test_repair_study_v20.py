import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import study_repair_v20 as study


def example():
    spec = dict(n_initials=2, modes=['clean', 'uniform'], methods=study.methods())
    rows = []
    for i in range(2):
        for mode in spec['modes']:
            for name, method in spec['methods'].items():
                old_used = method['kind'] != 'baseline' and i == 0
                new_used = method['kind'] == 'repair' or old_used
                gain = .2 if new_used else 0.
                rows.append(dict(initial_index=i, disturbance=mode, controller=name,
                                 completed=True, collision=False, aborted=False, abort_reason='', contract_violations=0,
                                 initial=[[-80., 12.], [-85., 13.], [-120., 11.]],
                                 per_vehicle_cost=[10. - gain] * 3, per_vehicle_finish=[9.] * 3,
                                 per_vehicle_variation=[20. - gain * 20] * 3,
                                 team_cost=10. - gain, learned_action_steps=int(new_used),
                                 chosen_compositions=[int(new_used)], forecast_transitions=30, internal_transitions=100,
                                 trajectory_sha256=str(new_used), decision_sha256=str(new_used),
                                 repair_action_steps=int(method['kind'] == 'repair')))
    return rows, spec


def test_complete_model_family_includes_old_controller_and_disabled_adapter():
    methods = study.methods()
    assert len(methods) == 10
    for seed in (1802, 1812, 1822):
        assert [methods[f'{prefix}_{seed}']['kind'] for prefix in ('sampled', 'reference', 'repair')] == ['temporal', 'reference', 'repair']


def test_actual_reference_identity_and_adoption_inclusion_have_separate_counts():
    rows, spec = example()
    result = study.compare_pairs(rows, spec)
    assert len(result) == 3
    for record in result.values():
        assert record['reference_episode_pairs'] == 4
        assert record['retained_adopted_pairs'] == record['added_adopted_pairs'] == 2
        assert record['added_harmful_pairs'] == 0
        assert record['all_mode_mean_cost_gain_over_original'] == pytest.approx(.1)


def test_reference_command_or_prediction_evidence_change_is_detected():
    rows, spec = example()
    for key in ('trajectory_sha256', 'decision_sha256', 'forecast_transitions', 'per_vehicle_cost'):
        wrong = copy.deepcopy(rows)
        target = next(row for row in wrong if row['controller'] == 'reference_1812')
        target[key] = 'different'
        with pytest.raises(AssertionError):
            study.compare_pairs(wrong, spec)


def test_lost_episode_adoption_is_rejected_even_if_team_cost_looks_better():
    rows, spec = example()
    target = next(row for row in rows if row['controller'] == 'repair_1802')
    target['learned_action_steps'] = 0
    target['team_cost'] = 0.
    with pytest.raises(ValueError, match='lost an episode adoption'):
        study.compare_pairs(rows, spec)


def test_added_individual_harm_is_not_hidden_by_positive_team_gain():
    rows, spec = example()
    target = next(row for row in rows if row['controller'] == 'repair_1822' and row['initial_index'] == 1 and row['disturbance'] == 'uniform')
    target['per_vehicle_cost'] = [9., 9., 10.1]
    target['team_cost'] = sum(target['per_vehicle_cost']) / 3
    result = study.compare_pairs(rows, spec)['1822']
    assert result['added_harmful_pairs'] == 1
    assert result['all_mode_mean_cost_gain_over_original'] > 0


def test_missing_or_duplicate_cases_are_never_summarized():
    rows, spec = example()
    for wrong in (rows[:-1], rows + [rows[0]]):
        with pytest.raises(ValueError, match='Missing or duplicate'):
            study.compare_pairs(wrong, spec)


def test_trace_identity_ignores_timing_but_includes_realized_noise_and_forecasts():
    trace = [dict(t=.5, x=[[-80., 12.]], command=[0., 0., 0.], disturbance=[[0., 0., 0.]],
                  gate_ms=100., selected_bank_differences=[[[0., 0.]]])]
    original = study.trace_metadata(trace)
    changed = copy.deepcopy(trace)
    changed[0]['gate_ms'] = 900.
    assert original == study.trace_metadata(changed)
    changed[0]['disturbance'][0][0] = .1
    assert original['trajectory_sha256'] != study.trace_metadata(changed)['trajectory_sha256']
    changed = copy.deepcopy(trace)
    changed[0]['selected_bank_differences'][0][0][0] = .1
    assert original['decision_sha256'] != study.trace_metadata(changed)['decision_sha256']
