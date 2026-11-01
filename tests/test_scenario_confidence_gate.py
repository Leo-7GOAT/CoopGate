import numpy as np
import pytest

from coopgate.marl_v12 import LEVELS
from coopgate.memoized_recovery_v37 import MemoizedRecoveryGate
from coopgate.scenario_confidence_gate import ScenarioConfidenceGate

STATE=np.array([[-140.,12.],[-220.,10.],[-190.,12.]])


class CoastingActor:
    def proposal(self,obs,mask):
        return np.where(mask,-np.abs(LEVELS),-1e9).argmax(-1)


class FixedForecastFixture(ScenarioConfidenceGate):
    def __init__(self,bad_paths,budget):
        super().__init__(CoastingActor(),'nominal',budget);self.bad_paths=bad_paths

    def predict(self,state,previous,mlo,mhi,command,path):
        value=10. if command is None else (10.1 if path in self.bad_paths else 9.)
        values=np.full((3,2),value);crossing=np.full(3,value)
        return (values,crossing,True,0,0,0,0,0.,0,state,np.zeros(3),mlo,mhi,0,
                None,None,None,None,None,None,None,True)


@pytest.mark.parametrize('bad,budget,adopt',[(1,0,False),(1,1,True),(2,1,False),(2,3,True)])
def test_score_threshold_changes_admission_with_unchanged_physical_certificate(bad,budget,adopt):
    gate=FixedForecastFixture(set(range(1,bad+1)),budget)
    _,info=gate.select(STATE,np.zeros(3))
    assert info['actual_learning_used']==adopt
    assert info['certified'] and info['certificate_kind']=='bounded_actuator_memory_stopping'
    assert not info['scenario_score_is_probability'] and not info['conditional_performance_risk_certified']
    if adopt:assert info['scenario_pass_fraction']==(33-bad)/33


@pytest.mark.parametrize('profile',('nominal','gain_drag','lag_mild','lag_strong'))
def test_zero_failure_budget_preserves_original_real_predictor_command_and_admission(profile):
    original=MemoizedRecoveryGate(CoastingActor(),profile,'bank_performance')
    score=ScenarioConfidenceGate(CoastingActor(),profile,0)
    expected,old=original.select(STATE,np.zeros(3));actual,new=score.select(STATE,np.zeros(3))
    np.testing.assert_array_equal(actual,expected)
    for field in ('actual_learning_used','composition','repair_weight','selected_bank_differences',
                  'selected_phase_differences','baseline_command','baseline_evidence','finite_bank_performance_margin'):
        assert new[field]==old[field]
