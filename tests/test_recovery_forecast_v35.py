import numpy as np
import pytest

from coopgate.actuator_recovery_v34 import ActuatorRecoveryBaseline, certificate, class_bounds
from coopgate.disturbance_v9 import disturbance_tape
from coopgate.online_actuator_gate_v32 import reference_parameters
from coopgate.recovery_forecast_v35 import forecast, recovery_command
from coopgate.shifted_sumo_v19 import apply_actuator


STATE = np.array([[-140.,12.],[-220.,10.],[-190.,12.]])
PROFILES = ('nominal','gain_drag','lag_mild','lag_strong')


def independent_continuation(profile,mode,first,prefix):
    """Original Python V34 controller and independent simulator recurrence."""
    plant = reference_parameters(profile)
    tape = disturbance_tape(35007,mode,.1,250)
    baseline = ActuatorRecoveryBaseline(profile)
    x,prev,memory = STATE.copy(),np.zeros(3),np.zeros(3)
    finish,variation,crossing = np.full(3,25.-.5*prefix),np.zeros(3),np.full(3,np.nan)
    trace = []
    inputs = None
    for step in range(50):
        command,info = baseline.select(x,prev)
        active = x[:,0] < 60.
        if step == prefix:
            finish[~active],crossing[~active] = 0.,0.
            inputs = dict(state=x.copy(),previous=prev.copy(),memory=memory.copy(),
                observer_lower=np.array(info['actuator_memory_lower']),
                observer_upper=np.array(info['actuator_memory_upper']),step=step,tape=tape,
                plant=plant,bounds=class_bounds(profile),first_command=first,record_trace=True)
            if first is not None:
                command = first.copy()
                assert certificate(x,prev,command,inputs['observer_lower'],inputs['observer_upper'],inputs['bounds'])['certified']
        if step >= prefix:
            variation += active*np.abs(command-prev)
        memories,errors = [],[]
        for sub in range(5):
            before = x[:,0].copy()
            applied,memory = apply_actuator(command,x[:,1],memory,tape[step*5+sub],plant)
            memories.append(memory.tolist());errors.append((applied-command).tolist())
            x[:,1] = np.clip(x[:,1]+.1*applied,0.,20.)
            x[:,0] += .1*x[:,1]
            if step >= prefix:
                hit = np.isnan(crossing) & (before < 60.) & (x[:,0] >= 60.)
                crossing[hit] = (step-prefix)*.5+sub*.1+.1*(60.-before[hit])/(x[hit,0]-before[hit])
        if step >= prefix:
            finish[active & (x[:,0] >= 60.)] = (step-prefix+1)*.5
            trace.append(dict(t=(step+1)*.5,x=x.tolist(),command=command.tolist(),actuator_states=memories,
                disturbance=errors,actuator_memory_lower=info['actuator_memory_lower'],
                actuator_memory_upper=info['actuator_memory_upper'],
                recovery_applied=False if step == prefix and first is not None else info['recovery_applied']))
        prev = command
        if np.all(x[:,0] >= 60.): break
    return inputs,dict(values=np.column_stack((finish+.05*variation,finish)),crossing=crossing,
                       final_state=x,final_actuator=memory,trace=trace,complete=bool(np.all(x[:,0] >= 60.)))


@pytest.mark.parametrize('profile',PROFILES)
@pytest.mark.parametrize('mode',('clean','uniform','bias','pulse'))
@pytest.mark.parametrize('prefix,override',((0,False),(0,True),(3,False),(3,True)))
def test_full_forecast_matches_independent_controller_and_dynamics(profile,mode,prefix,override):
    args,expected = independent_continuation(profile,mode,np.zeros(3) if override else None,prefix)
    result = forecast(**args)
    assert result['complete'] == expected['complete'] and result['error_code'] == 0
    for name in ('values','final_state','final_actuator'):
        np.testing.assert_allclose(result[name],expected[name],rtol=0,atol=1e-9)
    np.testing.assert_allclose([np.nan if v is None else v for v in result['crossing']],
                               expected['crossing'],rtol=0,atol=1e-9,equal_nan=True)
    assert len(result['trace']) == len(expected['trace'])
    for a,b in zip(result['trace'],expected['trace'],strict=True):
        assert a['command'] == b['command'] and a['recovery_applied'] == b['recovery_applied']
        for name in ('x','actuator_states','disturbance','actuator_memory_lower','actuator_memory_upper'):
            np.testing.assert_allclose(a[name],b[name],rtol=0,atol=1e-9)


@pytest.mark.parametrize('profile',PROFILES)
def test_compiled_baseline_agrees_with_python_recovery(profile):
    gate = ActuatorRecoveryBaseline(profile)
    expected,info = gate.select(STATE,np.zeros(3))
    result = recovery_command(STATE,np.zeros(3),0,np.zeros(3),np.zeros(3),gate.bounds,gate.packed)
    np.testing.assert_array_equal(result[0],expected)
    assert result[1] and result[2] == info['recovery_applied']
    assert result[3] == (info['nominal_proposal_error'] is not None)


def valid_args():
    return dict(state=STATE,previous=np.zeros(3),memory=np.zeros(3),observer_lower=np.zeros(3),
        observer_upper=np.zeros(3),step=0,tape=np.zeros((250,3)),plant=reference_parameters('nominal'),
        bounds=class_bounds('nominal'))


@pytest.mark.parametrize('change',('memory','clock','path','coefficients','interval','first','trace'))
def test_invalid_forecast_context_fails_closed(change):
    args = valid_args()
    if change == 'memory': args['memory'] = np.ones(3)
    if change == 'clock': args['step'] = True
    if change == 'path': args['tape'] = np.full((250,3),.11)
    if change == 'coefficients': args['plant']['gain'][0] = 1.1
    if change == 'interval': args['observer_lower'] = np.ones(3)
    if change == 'first': args['first_command'] = np.full(3,np.nan)
    if change == 'trace': args['record_trace'] = 1
    with pytest.raises(ValueError): forecast(**args)


def test_invalid_first_command_and_exhausted_horizon_never_approve_performance():
    args = valid_args()
    result = forecast(**args,first_command=np.full(3,-5.))
    assert not result['complete'] and result['error_code'] == 3 and result['steps'] == 0
    args['step'] = 50
    result = forecast(**args)
    assert not result['complete'] and result['steps'] == 0


def test_reference_point_is_not_a_replacement_for_observer_uncertainty():
    args = valid_args()
    args.update(plant=reference_parameters('lag_strong'),bounds=class_bounds('lag_strong'),
                observer_lower=np.full(3,-.25),observer_upper=np.full(3,.25),record_trace=True)
    result = forecast(**args)
    assert result['trace']
    assert result['trace'][0]['actuator_memory_lower'] == [-.25]*3
    assert result['trace'][0]['actuator_memory_upper'] == [.25]*3
