import numpy as np
import pytest
from coopgate.actuator_recovery_v34 import class_bounds
from coopgate.parameter_scenario_gate import parameter_bank,ParameterScenarioGate
from coopgate.scenario_confidence_gate import ScenarioConfidenceGate
from coopgate.marl_v12 import LEVELS


class Coast:
    def proposal(self,obs,mask):return np.where(mask,-np.abs(LEVELS),-1e9).argmax(-1)


@pytest.mark.parametrize('profile',('nominal','gain_drag','lag_mild','lag_strong'))
def test_bank_retains_original_paths_and_uses_only_declared_parameters(profile):
    old=ScenarioConfidenceGate(Coast(),profile);new=ParameterScenarioGate(Coast(),profile)
    np.testing.assert_array_equal(old.tapes,new.tapes[:33])
    bounds=class_bounds(profile)
    assert len(new.tapes)==(33 if profile=='nominal' else 41)
    for gain,drag,rho in parameter_bank(profile):
        for value,left,right in ((gain,0,1),(drag,2,3),(rho,4,5)):
            assert np.all(value>=bounds[:,left]) and np.all(value<=bounds[:,right])


def test_scenario_memory_is_reconstructed_from_executed_history_not_selected_output():
    gate=ParameterScenarioGate(Coast(),'lag_strong');state=np.array([[-140.,12.],[-220.,10.],[-190.,12.]])
    gate.select(state,np.zeros(3));previous=np.array([.625,1.25,-.625])
    _,info=gate.select(state,previous)
    for i,(_,_,rho) in enumerate(gate.parameter_scenarios):
        expected=np.zeros(3)
        for _ in range(5):expected=rho*expected+(1-rho)*previous
        np.testing.assert_array_equal(gate.parameter_memories[i],expected)
    assert info['forecast_bank_size']==41 and not info['full_parameter_box_certified']
