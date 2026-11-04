"""The reserve is a local TV identity, not an assumed global error bound."""
import numpy as np
import pytest
from experiments.switching_reserve.gate import variation_reserve,SwitchingReserveGate
from experiments.joint_admissibility.gate import JointAdmissibilityGate


def test_triangle_excess_equals_twice_distance_to_the_backup_segment():
    rng=np.random.default_rng(119)
    for _ in range(200):
        p,b,u=rng.uniform(-5,2.5,size=(3,3))
        distance=np.maximum(np.minimum(p,b)-u,0)+np.maximum(u-np.maximum(p,b),0)
        np.testing.assert_allclose(variation_reserve(p,b,u),.1*distance,atol=2e-16,rtol=1e-12)


def test_intermediate_commands_have_zero_two_command_variation_reserve():
    p=np.array([-5.,0.,2.5]);b=np.array([2.5,-5.,0.])
    np.testing.assert_array_equal(variation_reserve(p,b,.25*p+.75*b),np.zeros(3))


@pytest.mark.parametrize('delta,command,passed',[(.1,-2.5,False),(.5,-2.5,True),(.1,1.25,True)])
def test_reserve_rejects_small_backtracking_gain_but_keeps_sufficient_gain(monkeypatch,delta,command,passed):
    obj=SwitchingReserveGate.__new__(SwitchingReserveGate)
    obj.tapes=np.zeros((2,250,3));obj.gain_floor=.0025
    obj._current_public_baseline=np.full(3,2.5)
    chosen=np.full(3,command);obj._reserve_predictions={}
    for path in range(2):
        before=[np.column_stack((np.full(3,10.),np.full(3,10.)))]
        after=[before[0]-np.column_stack((np.full(3,delta),np.zeros(3)))]
        obj._reserve_predictions[None,path]=before
        obj._reserve_predictions[chosen.tobytes(),path]=after
    monkeypatch.setattr(JointAdmissibilityGate,'_history_admissible',
        lambda *args:(True,dict(passed=True,fitted_forecast_available=False)))
    accepted,evidence=obj._history_admissible(None,np.zeros(3),None,None,chosen)
    assert accepted is passed and evidence['passed'] is passed
    assert evidence['switching_reserve']['full_closed_loop_error_bound'] is False


def test_nonfinite_commands_are_rejected():
    with pytest.raises(ValueError):variation_reserve(np.zeros(3),np.zeros(3),[0.,np.nan,0.])
