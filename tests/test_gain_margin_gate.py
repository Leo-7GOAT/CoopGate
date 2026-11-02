import copy
import numpy as np
import pytest
from coopgate.gain_margin_gate import relative_gain_score,ScalarGainMarginGate
from coopgate.scalar_nominal_recovery import ScalarNominalParameterGate
from coopgate.gain_margin_audit import validate_gain_evidence
from coopgate.marl_v12 import LEVELS


class Coast:
    def proposal(self,obs,mask):return np.where(mask,-np.abs(LEVELS),-1e9).argmax(-1)


def test_score_uses_worst_normalized_team_gain_without_unaffected_coordinate_degeneracy():
    delta=np.zeros((2,3,2));delta[0,0,0]=-.3;delta[1,1,0]=-.3
    assert relative_gain_score(delta,[10.,20.])==pytest.approx(.005)
    assert relative_gain_score(delta,[10.,0.]) is None
    assert relative_gain_score(np.full((2,3,2),np.nan),[10.,20.]) is None


@pytest.mark.parametrize('profile',('nominal','gain_drag','lag_mild','lag_strong'))
def test_zero_floor_preserves_every_original_non_timing_field(profile):
    state=np.array([[-140.,12.],[-220.,10.],[-190.,12.]]);previous=np.zeros(3)
    a=ScalarNominalParameterGate(Coast(),profile);b=ScalarGainMarginGate(Coast(),profile,0.)
    for _ in range(2):
        old,old_info=a.select(state,previous);new,new_info=b.select(state,previous)
        assert old.tobytes()==new.tobytes()
        for key,value in old_info.items():
            if key!='gate_ms':assert new_info[key]==value
        validate_gain_evidence(new_info);previous=old


def test_independent_score_checker_rejects_tampering():
    d=np.zeros((33,3,2));d[:,0,0]=-.3
    entry=dict(candidate=0,composition=1,repair_weight=1.,baseline_team_costs=[10.]*33,
        worst_relative_team_gain=.01,floor=.005,passed=True)
    item=dict(gain_confidence_floor=.005,gain_score_is_probability=False,
        performance_confidence_kind='normalized_worst_path_team_gain',forecast_bank_size=33,
        candidate_evidence=[dict(candidate=0,differences=d.tolist(),passed=True)],gain_margin_evidence=[entry],
        actual_learning_used=True,composition=1,repair_weight=1.,selected_baseline_forecast_team_costs=[10.]*33,
        selected_gain_confidence_score=.01)
    validate_gain_evidence(item)
    changed=copy.deepcopy(item);changed['gain_margin_evidence'][0]['worst_relative_team_gain']=.99
    with pytest.raises(ValueError,match='score differs'):validate_gain_evidence(changed)
    changed=copy.deepcopy(item);changed['gain_confidence_floor']=.01;changed['gain_margin_evidence'][0]['passed']=False
    with pytest.raises(ValueError):validate_gain_evidence(changed)
