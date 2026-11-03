import numpy as np
import pytest
from coopgate.observed_lag_bias_fit import fit_observed_history
from coopgate.actuator_prediction_roundoff_v31 import advance_actuator


@pytest.mark.parametrize('profile,rho,bias',[
    ('nominal',[0.,0.,0.],[.07,-.08,.025]),
    ('lag_mild',[.04,.06,.12],[.03,-.06,.09]),
    ('lag_strong',[.73,.78,.81],[-.09,.05,.02]),
])
def test_recovers_excited_hidden_lag_and_bias_from_boundary_measurements(profile,rho,bias):
    x=np.array([[-100.,10.],[-130.,9.],[-165.,11.]])
    states=[x.copy()];commands=[];memory=np.zeros(3)
    for command in ([1.,.4,-.8],[1.7,-.5,.2],[-.4,1.1,.6],[.3,-.7,1.2]):
        command=np.array(command);commands.append(command)
        for _ in range(5):
            x,memory,_=advance_actuator(x,memory,command,np.array(bias),(np.ones(3),np.zeros(3),np.array(rho)),.1,20.)
        states.append(x.copy())
    fit=fit_observed_history(states,commands,profile)
    np.testing.assert_allclose(fit['retention'],rho,atol=2e-6,rtol=0)
    np.testing.assert_allclose(fit['bias'],bias,atol=1e-6,rtol=0)
    np.testing.assert_allclose(fit['memory'],memory,atol=2e-6,rtol=0)
    assert fit['fit_squared_error'].max()<1e-12
    assert not fit['uses_true_parameters'] and not fit['uses_future_innovations']


def test_no_identification_at_reset_or_with_only_clipped_windows():
    x=np.array([[-100.,0.],[-130.,0.],[-165.,0.]])
    assert fit_observed_history([x],np.zeros((0,3)),'lag_strong') is None
    assert fit_observed_history([x,x],[np.zeros(3)],'lag_strong') is None
    with pytest.raises(ValueError):fit_observed_history([x,x],[np.zeros(3)],'gain_drag')
