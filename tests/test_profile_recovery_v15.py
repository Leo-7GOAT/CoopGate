import numpy as np
import pytest
from coopgate.profile_recovery_v15 import RecoveringProfile
from coopgate.ego_v13 import EgoRolloutGate
from coopgate.fast_v8 import advance_one,parameters
from coopgate.merge_v4 import MergeConfig


class Faulty:
    def __init__(self,mode):self.mode=mode;self.calls=0;self.step=0
    def select(self,x,prev):
        self.calls+=1
        if self.mode=='mutate':
            x[:]=1e9;prev[:]=1e9;raise RuntimeError('Injected input mutation')
        if self.mode=='raise':raise RuntimeError('Injected actor failure')
        if self.mode=='nan':return np.full(3,np.nan),{}
        if self.mode=='slew':return np.full(3,-5.),{}
        if self.mode=='unsafe':return np.full(3,2.5),{}
        raise ValueError(self.mode)


@pytest.mark.parametrize('mode',['raise','nan','slew','mutate'])
def test_faults_return_to_same_state_baseline_without_erasing_admission(mode):
    primary=Faulty(mode);wrapped=RecoveringProfile(primary,EgoRolloutGate(),True);reference=EgoRolloutGate()
    x=np.array([[-80.,12.],[-85.,13.],[-120.,11.]]);previous=np.zeros(3);c=parameters(MergeConfig())
    for _ in range(4):
        expected,_=reference.select(x,previous);command,info=wrapped.select(x,previous)
        np.testing.assert_array_equal(command,expected)
        assert info['fault_fallback'] and info['performance_admitted'] and wrapped.accept
        assert np.max(np.abs(command-previous))<=2.5+1e-9
        assert advance_one(x,command,c);previous=command
    assert primary.calls==1


def test_rejected_profile_never_calls_the_learning_controller():
    primary=Faulty('raise');wrapped=RecoveringProfile(primary,EgoRolloutGate(),False)
    x=np.array([[-80.,12.],[-85.,13.],[-120.,11.]])
    command,info=wrapped.select(x,np.zeros(3));expected,_=EgoRolloutGate().select(x,np.zeros(3))
    np.testing.assert_array_equal(command,expected)
    assert primary.calls==0 and not info['performance_admitted'] and not info['fault_fallback']


def test_physical_failure_is_detected_even_when_actuator_limits_are_met():
    # This initial is recoverable, but allowing both approaches to accelerate
    # together has no robust backup certificate.
    x=np.array([[-50.,12.],[-51.,12.],[-90.,11.]])
    primary=Faulty('unsafe');wrapped=RecoveringProfile(primary,EgoRolloutGate(),True)
    command,info=wrapped.select(x,np.zeros(3))
    expected,_=EgoRolloutGate().select(x,np.zeros(3))
    np.testing.assert_array_equal(command,expected)
    assert info['fault_fallback'] and 'certificate' in info['fault_reason']
