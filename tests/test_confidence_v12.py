import numpy as np
import pytest
from coopgate.confidence_v12 import (conformal_upper,EpisodeConfidenceGate,
    conflict_group,in_training_domain,predict_costs)
from coopgate.marl_v12 import MARLGate,LEVELS
from coopgate.training_v12 import SingleVectorMerge


def initial():return np.array([[-80.,12.],[-85.,13.],[-120.,11.]])


def calibration(q=0.):
    return {'minimum_gain_s':.05,'baseline':'constant','groups':{g:{'q':q,'n':100} for g in ('high_conflict','low_conflict')}}


def test_quantile_uses_n_plus_one_order_statistic():
    assert conformal_upper(np.arange(100),.1)==90
    assert np.isinf(conformal_upper([0.,1.,2.],.1))
    with pytest.raises(ValueError):conformal_upper([np.nan])


def test_single_learning_keeps_other_proposals_fixed():
    env=SingleVectorMerge(3,11);obs,state,mask=env.inputs()
    assert obs.shape[1]==1 and mask.shape[1]==1
    nxt,reward,terminal,boundary,rows=env.step(np.full((3,1),12))
    assert nxt[0].shape==(3,1,18)
    assert reward.shape==(3,)


def test_missing_policy_and_domain_shift_use_baseline():
    for x in (initial(),np.array([[-80.,18.],[-85.,13.],[-120.,11.]])):
        gate=EpisodeConfidenceGate(None,x,calibration())
        assert not gate.accept
        u,info=gate.select(x,np.zeros(3))
        expected,_=MARLGate(None,mode='constant').select(x,np.zeros(3))
        assert np.array_equal(u,expected) and info['certified']


def test_confidence_does_not_label_equal_prediction_as_advantage():
    class Constant:
        def proposal(self,obs,mask):
            return np.where(mask,np.arange(len(LEVELS)),-1).max(-1)
    gate=EpisodeConfidenceGate(Constant(),initial(),calibration())
    assert abs(gate.predicted_delta)<1e-9
    assert not gate.accept


def test_policy_prediction_failure_falls_back():
    class Broken:
        def proposal(self,obs,mask):raise ValueError('Injected policy fault')
    gate=EpisodeConfidenceGate(Broken(),initial(),calibration())
    assert gate.reason=='prediction_failure' and not gate.accept
    _,info=gate.select(initial(),np.zeros(3))
    assert info['certified']


def test_conflict_group_uses_current_arrival_estimate():
    assert conflict_group(initial())=='high_conflict'
    assert in_training_domain(initial())
    assert not in_training_domain(np.array([[-80.,18.],[-85.,13.],[-120.,11.]]))
