import ast
import inspect

import numpy as np
import pytest

from coopgate import buffered_actuator_v33 as fast
from coopgate import online_actuator_gate_v32 as original
from coopgate.joint_planner_v15 import joint_rollout


@pytest.mark.parametrize('reference,counterpart',[(joint_rollout,fast.joint_rollout),
    (original.forecast_kernel,fast.forecast_kernel),(original.forecast,fast.forecast)])
def test_forecast_arithmetic_and_original_search_work_are_unchanged(reference,counterpart):
    ref=getattr(reference,'py_func',reference);other=getattr(counterpart,'py_func',counterpart)
    assert ast.dump(ast.parse(inspect.getsource(ref)),include_attributes=False)==ast.dump(
        ast.parse(inspect.getsource(other)),include_attributes=False)
    assert fast.forecast_kernel.targetoptions['fastmath'] is False


def test_actual_gate_inherits_selection_and_memory_rules_without_modification():
    assert fast.BufferedActuatorGate.select is original.ReferenceActuatorGate.select
    assert fast.BufferedActuatorGate.__init__ is original.ReferenceActuatorGate.__init__


@pytest.mark.parametrize('profile',['nominal','gain_drag','lag_mild','lag_strong'])
@pytest.mark.parametrize('first',[None,np.array([2.5,0.,2.5])])
def test_full_suffix_fields_and_commands_match_exactly(profile,first):
    state=np.array([[-80.,12.],[-85.,13.],[-120.,11.]])
    tape=np.random.default_rng(33031).uniform(-.1,.1,(250,3))
    args=(state,np.zeros(3),np.zeros(3),0,tape,original.reference_parameters(profile),first)
    before=[v.copy() for v in args[:3]]
    a=original.forecast(*args,record_trace=True);b=fast.forecast(*args,record_trace=True)
    np.testing.assert_equal(a,b)
    for x,old in zip(args[:3],before):np.testing.assert_array_equal(x,old)


@pytest.mark.parametrize('step',[0,17,50])
def test_unsafe_and_completed_boundaries_keep_failure_identity_and_work(step):
    for state in (np.array([[0.,10.],[0.,10.],[-1.,10.]]),np.array([[70.,0.],[71.,0.],[72.,0.]])):
        args=(state,np.zeros(3),np.zeros(3),step,np.zeros((250,3)),original.reference_parameters('lag_strong'))
        np.testing.assert_equal(original.forecast(*args,record_trace=True),fast.forecast(*args,record_trace=True))


@pytest.mark.parametrize('bad',['memory','clock','path','command'])
def test_validation_keeps_rejecting_invalid_forecast_inputs(bad):
    args=[np.array([[-80.,12.],[-85.,13.],[-120.,11.]]),np.zeros(3),np.zeros(3),0,
        np.zeros((250,3)),original.reference_parameters('nominal'),None]
    if bad=='memory':args[2]=np.array([2.5+1e-10,0.,0.])
    elif bad=='clock':args[3]=.5
    elif bad=='path':args[4]=np.zeros((249,3))
    else:args[-1]=np.array([2.6,0.,0.])
    for fn in (original.forecast,fast.forecast):
        with pytest.raises(ValueError):fn(*args)
