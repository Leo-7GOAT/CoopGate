"""A vetoed candidate keeps its full evidence without claiming execution."""
from copy import deepcopy

import pytest
from experiments.joint_admissibility_recovery.audit import validate_joint_trace


def fixture():
    differences=[[[-1.,0.],[-1.,0.],[-1.,0.]]]*2
    return dict(development_controller='joint_candidate_admissibility',
        forecast_bank_size=2,gain_confidence_floor=.0025,gain_score_is_probability=False,
        performance_confidence_kind='normalized_worst_path_team_gain',
        confidence_pruning_kind='rejection_only_min_over_paths_upper_bound',
        gain_margin_prefix_rejections=[],
        candidate_evidence=[dict(candidate=0,passed=False,history_rejected=True,
            differences=differences),dict(candidate=1,passed=True,differences=differences)],
        gain_margin_evidence=[dict(candidate=i,composition=i+1,repair_weight=1.,
            baseline_team_costs=[100.,100.],worst_relative_team_gain=.01,
            floor=.0025,passed=True) for i in range(2)],
        history_candidate_evidence=[dict(candidate=0,passed=False,
            fitted_forecast_available=True,complete_pair=True,
            differences=[[.1,0.],[-1.,0.],[-1.,0.]],phase_differences=[0.,0.,0.],
            relative_gain=.01),dict(candidate=1,passed=True,fitted_forecast_available=False)],
        actual_learning_used=True,selected_gain_confidence_score=.01,
        selected_baseline_forecast_team_costs=[100.,100.],composition=2,repair_weight=1.,
        selected_phase_differences=[[0.,0.,0.]]*2)


def test_veto_and_final_admission_are_independently_validated_without_mutation():
    item=fixture();before=deepcopy(item)
    assert validate_joint_trace(item)==dict(history_candidates=2,history_vetoes=1)
    assert item==before


@pytest.mark.parametrize('mutation',('gain','history','missing','phase'))
def test_corrupted_veto_or_final_admission_is_rejected(mutation):
    item=fixture()
    if mutation=='gain':item['gain_margin_evidence'][0]['worst_relative_team_gain']=.02
    elif mutation=='history':item['history_candidate_evidence'][0]['passed']=True
    elif mutation=='missing':item['history_candidate_evidence'].pop(0)
    else:item['selected_phase_differences']=[[.001,0.,0.]]*2
    with pytest.raises(ValueError):validate_joint_trace(item)


def test_all_vetoes_are_validated_before_exact_fallback():
    item=fixture()
    item['candidate_evidence'][1].update(passed=False,history_rejected=True)
    item['history_candidate_evidence'][1]=dict(item['history_candidate_evidence'][0],candidate=1)
    item.update(actual_learning_used=False,selected_gain_confidence_score=None,
                selected_baseline_forecast_team_costs=[],composition=0,repair_weight=None,
                selected_phase_differences=[])
    assert validate_joint_trace(item)['history_vetoes']==2
