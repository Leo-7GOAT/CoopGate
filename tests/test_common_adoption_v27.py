import numpy as np
import pytest

from coopgate.common_adoption_v27 import BaselineReplayCache, audit_trace, changed_active_commands
from coopgate.joint_planner_v15 import JointRolloutGate


def test_adoption_ignores_policy_labels_and_arrived_vehicles():
    state=np.array([[-20.,10.],[60.,10.],[59.9,10.]])
    np.testing.assert_array_equal(changed_active_commands(state,[0,2.5,.5],[0,0,0]),[False,False,True])
    assert not changed_active_commands(state,[0,2.5,0],[0,0,0]).any()


def test_strict_threshold_is_retained():
    state=np.zeros((3,2))
    np.testing.assert_array_equal(changed_active_commands(state,[1e-9,1.1e-9,-1.1e-9],[0,0,0]),[False,True,True])


class Probe:
    def __init__(self):
        self.step=0
        self.calls=0
    def select(self,state,previous):
        self.calls+=1
        return previous+state[:,0]+self.step,{}


def test_cache_includes_state_previous_and_time_and_returns_copies():
    probe=Probe()
    cache=BaselineReplayCache(probe)
    state,previous=np.zeros((3,2)),np.zeros(3)
    first=cache.command(state,previous,0)
    first[:]=999
    np.testing.assert_array_equal(cache.command(state,previous,0),np.zeros(3))
    cache.command(state,previous,1)
    cache.command(state,np.ones(3),0)
    altered=state.copy();altered[0,0]=1
    cache.command(altered,previous,0)
    assert probe.calls==cache.evaluations==4 and cache.hits==1
    assert not state.any() and not previous.any()


def test_trace_uses_measured_post_step_state_and_executed_previous_command():
    cache=BaselineReplayCache(Probe())
    trace=[dict(command=[1.,0.,0.],x=[[2.,0.],[0.,0.],[0.,0.]]),
           dict(command=[4.,1.,1.],x=[[3.,0.],[1.,0.],[1.,0.]])]
    records=audit_trace(np.zeros((3,2)),trace,cache)
    assert records[0]['adopted']
    assert records[1]['baseline_command']==[4.,1.,1.] and not records[1]['adopted']


def test_actual_baseline_cached_commands_equal_fresh_reconstruction():
    cache=BaselineReplayCache()
    state=np.array([[-80.,12.],[-85.,13.],[-120.,11.]])
    previous=np.zeros(3)
    for step in (0,1,7,0):
        other=JointRolloutGate();other.step=step
        expected,_=other.select(state,previous)
        np.testing.assert_array_equal(cache.command(state,previous,step),expected)
    assert cache.evaluations==3 and cache.hits==1


@pytest.mark.parametrize('step',[-1,True,.5])
def test_invalid_step_is_not_silently_coerced(step):
    with pytest.raises(ValueError):
        BaselineReplayCache(Probe()).command(np.zeros((3,2)),np.zeros(3),step)


@pytest.mark.parametrize('bad',[np.zeros(2),np.array([np.nan,0.,0.]),np.array([np.inf,0.,0.])])
def test_bad_commands_cannot_be_treated_as_no_adoption(bad):
    with pytest.raises(ValueError):
        changed_active_commands(np.zeros((3,2)),bad,np.zeros(3))


def test_baseline_failure_propagates_without_false_rejection():
    class Broken:
        def select(self,*args):
            raise RuntimeError('No valid baseline')
    with pytest.raises(RuntimeError,match='No valid baseline'):
        audit_trace(np.zeros((3,2)),[dict(command=[0.,0.,0.],x=np.zeros((3,2)))],BaselineReplayCache(Broken()))
