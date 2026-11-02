import copy
import numpy as np
import pytest
from coopgate.gain_margin_gate import ScalarGainMarginGate
from coopgate.pruned_gain_margin_gate import PrunedGainMarginGate
from coopgate.pruned_gain_margin_audit import validate_prefix_evidence
from coopgate.marl_v12 import LEVELS


class Coast:
    def proposal(self,obs,mask):return np.where(mask,-np.abs(LEVELS),-1e9).argmax(-1)


@pytest.mark.parametrize('floor',(0.,.0025,.005,.01))
def test_rejection_only_selector_preserves_commands_and_complete_selected_evidence(floor):
    state=np.array([[-140.,12.],[-220.,10.],[-190.,12.]]);prev=np.zeros(3)
    a=ScalarGainMarginGate(Coast(),'lag_strong',floor);b=PrunedGainMarginGate(Coast(),'lag_strong',floor)
    for _ in range(2):
        u,left=a.select(state,prev);v,right=b.select(state,prev)
        assert u.tobytes()==v.tobytes()
        for key in ('actual_learning_used','actual_learning_used_ever','selected_bank_differences','selected_phase_differences',
                    'selected_gain_confidence_score','selected_baseline_forecast_team_costs','composition','repair_weight','baseline_command','nominal_evidence'):
            assert left[key]==right[key]
        validate_prefix_evidence(right);prev=u


def test_prefix_checker_rejects_non_witness_and_wrong_path():
    delta=[[[0.,0.],[0.,0.],[0.,0.]]]
    item=dict(gain_confidence_floor=.005,gain_score_is_probability=False,
        performance_confidence_kind='normalized_worst_path_team_gain',forecast_bank_size=33,
        candidate_evidence=[dict(candidate=0,checked_paths=1,differences=delta,passed=False)],gain_margin_evidence=[],
        actual_learning_used=False,selected_baseline_forecast_team_costs=[],selected_gain_confidence_score=None,
        confidence_pruning_kind='rejection_only_min_over_paths_upper_bound',
        gain_margin_prefix_rejections=[dict(candidate=0,path=0,baseline_team_cost=10.,score_upper_bound=0.,floor=.005)])
    validate_prefix_evidence(item)
    bad=copy.deepcopy(item);bad['gain_margin_prefix_rejections'][0]['score_upper_bound']=.5
    with pytest.raises(ValueError,match='upper bound'):validate_prefix_evidence(bad)
    bad=copy.deepcopy(item);bad['gain_margin_prefix_rejections'][0]['path']=1
    with pytest.raises(ValueError,match='path identity'):validate_prefix_evidence(bad)
