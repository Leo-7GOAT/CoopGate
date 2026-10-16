import numpy as np
import pytest
from coopgate.profile_prediction_v16 import predict_profile


class Constant:
    def __init__(self,value):self.value=value
    def select(self,x,previous):return np.full(3,self.value),{}


def test_forecast_preserves_individual_arrival_cost_accounting():
    initial=np.array([[250.,2.],[59.9,2.],[200.,2.]])
    prediction,metadata=predict_profile(initial,lambda x:Constant(2.5))
    np.testing.assert_allclose(prediction,[[0.,0.],[.625,.5],[0.,0.]],atol=1e-10)
    assert metadata['completed'] and metadata['forecast_transitions']==1
    np.testing.assert_array_equal(initial,[[250.,2.],[59.9,2.],[200.,2.]])


def test_forecast_does_not_trust_invalid_controller_outputs():
    initial=np.array([[-80.,12.],[-85.,13.],[-120.,11.]])
    with pytest.raises(ValueError,match='controller failed'):predict_profile(initial,lambda x:Constant(np.nan))
    with pytest.raises(ValueError,match='acceleration limit'):predict_profile(initial,lambda x:Constant(100.))
