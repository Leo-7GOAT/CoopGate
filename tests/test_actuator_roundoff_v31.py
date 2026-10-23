import numpy as np
import pytest

from coopgate.actuator_prediction_v30 import predict_suffix as strict
from coopgate.actuator_prediction_roundoff_v31 import predict_suffix
from coopgate.matched_shift_v21 import matched_plant_parameters
from coopgate.disturbance_v9 import disturbance_tape
from coopgate.shifted_sumo_v19 import PLANTS

STATE = np.array([[-80.,12.],[-85.,13.],[-120.,11.]])


@pytest.mark.parametrize('plant',PLANTS)
def test_every_output_is_identical_for_previously_admitted_exact_input(plant):
    tape = disturbance_tape(31037,'uniform',.1,250)
    args = (STATE,np.zeros(3),np.array([.5,1.,-1.]),0,tape,matched_plant_parameters(31037,plant))
    assert predict_suffix(*args,record_trace=True) == strict(*args,record_trace=True)


@pytest.mark.parametrize('endpoint,direction',[(2.5,np.inf),(-5.,-np.inf)])
def test_one_ulp_convex_recurrence_roundoff_is_preserved_without_clipping(endpoint,direction):
    memory = np.full(3,np.nextafter(endpoint,direction))
    original = memory.copy()
    args = (STATE,np.zeros(3),memory,50,np.zeros((250,3)),matched_plant_parameters(31037,'lag_mild'))
    with pytest.raises(ValueError,match='bounded actuator memory'):
        strict(*args)
    result = predict_suffix(*args)
    assert result['final_actuator'] == original.tolist()
    np.testing.assert_array_equal(memory,original)
    assert not result['complete']


@pytest.mark.parametrize('memory',[np.full(3,2.5+1e-10),np.full(3,-5.-1e-10),np.full(3,np.inf)])
def test_actual_bound_violations_are_still_rejected(memory):
    with pytest.raises(ValueError,match='bounded actuator memory'):
        predict_suffix(STATE,np.zeros(3),memory,0,np.zeros((250,3)),matched_plant_parameters(31037,'lag_mild'))
