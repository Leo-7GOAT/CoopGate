import itertools

import numpy as np
import pytest

from coopgate.aged_observation_envelope import (
    AgeAwareStateObserver, PublicEnvelope, delayed_interval, estimate_frames,
    memory_interval, state_interval, timestamp_vector,
)
from coopgate.ood_mismatch_v39 import compound_substep


def test_interval_dynamics_cover_corners_and_random_compound_transitions():
    rng = np.random.default_rng(460100)
    b = PublicEnvelope()
    for trial in range(130):
        lo = np.array([-100., rng.uniform(0, 20)])
        hi = lo+np.array([rng.uniform(0, 5), rng.uniform(0, 20-lo[1])])
        m = sorted(rng.uniform(-5, 2.5, 2)); q = sorted(rng.uniform(-5, 2.5, 2))
        ml, mh = memory_interval(np.full(3,m[0]), np.full(3,m[1]),
                                np.full(3,q[0]), np.full(3,q[1]), b)
        lower, upper = state_interval(lo,hi,ml[0],mh[0],b)
        samples = (itertools.product((0.,1.),repeat=7) if trial < 30
                   else rng.uniform(size=(16,7)))
        for fv,fm,fq,ft,fg,fd,fw in samples:
            velocity = lo[1]+fv*(hi[1]-lo[1])
            plant = dict(tau=np.full(3,b.tau_lower+ft*(b.tau_upper-b.tau_lower)),
                gain=np.full(3,b.gain_lower+fg*(b.gain_upper-b.gain_lower)),
                drag=np.full(3,b.drag_lower+fd*(b.drag_upper-b.drag_lower)))
            acceleration, actual_memory = compound_substep(np.full(3,q[0]+fq*(q[1]-q[0])),
                np.full(3,velocity),np.full(3,m[0]+fm*(m[1]-m[0])),
                np.full(3,(2*fw-1)*b.innovation_radius),plant)
            assert np.all(actual_memory >= ml) and np.all(actual_memory <= mh)
            v = np.clip(velocity+b.dt*acceleration[0],0,20)
            assert lower[0] <= lo[0]+b.dt*v <= hi[0]+b.dt*v <= upper[0]
            assert lower[1] <= v <= upper[1]


def test_delayed_observer_tracks_hidden_fifo_and_memory_without_receiving_them():
    rng = np.random.default_rng(460101)
    observer = AgeAwareStateObserver('stale_dropout')
    x = np.array([[-120.,12.],[-150.,10.],[-175.,16.]])
    states=[x.copy()]; commands=[]; memory=np.zeros(3)
    plant=dict(tau=np.array([.12,.65,1.25]),gain=np.array([.78,1.04,1.22]),
               drag=np.array([.001,.00065,.0001]))
    delays=np.array([1,2,3]);previous=np.zeros(3)
    for step in range(40):
        source=np.maximum(0,step-np.array([1,2,3]))
        measurement=np.array([states[s][i] for i,s in enumerate(source)])
        result=observer.update(measurement,source,previous)
        assert np.all(x>=np.asarray(result['lower'])) and np.all(x<=np.asarray(result['upper']))
        assert np.all(memory>=result['memory_lower']) and np.all(memory<=result['memory_upper'])
        command=rng.uniform(-5,2.5,3);commands.append(command)
        delayed=np.array([commands[step-delays[i]][i] if step>=delays[i] else 0 for i in range(3)])
        for _ in range(5):
            a,memory=compound_substep(delayed,x[:,1],memory,rng.uniform(-.1,.1,3),plant)
            x[:,1]=np.clip(x[:,1]+.1*a,0,20);x[:,0]+=.1*x[:,1]
        states.append(x.copy());previous=command


def test_public_facade_cannot_read_truth_future_innovations_or_realized_parameters():
    class DenyTruth(dict):
        def __getitem__(self,key):
            assert key in {'observed_state','perception','previous_command'}
            return super().__getitem__(key)
    frames=[DenyTruth(observed_state=[[-100.,12.]]*3,
        perception=dict(profile='clean_reference',decision=0,source_decision=0),previous_command=[0.]*3)]
    assert len(estimate_frames(frames,'clean_reference'))==1


def test_bias_jitter_domain_intersection_and_inconsistent_history_fail_explicitly():
    observer=AgeAwareStateObserver('bounded_bias_jitter')
    result=observer.update([[-100.,20.199]]*3,0,[0.]*3)
    assert np.all(np.asarray(result['upper'])[:,1]==20)
    assert not result['original_exact_state_certificate_restored']
    with pytest.raises(ValueError,match='contradicts'):
        observer.update([[500.,20.]]*3,1,[0.]*3)


@pytest.mark.parametrize('source', [[0,1,2],-1,[0.,0.,0.],[True]*3,[0,0]])
def test_invalid_or_future_timestamps_rejected(source):
    with pytest.raises(ValueError):timestamp_vector(source,0)


def test_delayed_interval_includes_initial_zero_and_every_possible_delay():
    lo,hi=delayed_interval([np.array([2.5,-5.,1.])],3)
    np.testing.assert_array_equal(lo,0);np.testing.assert_array_equal(hi,0)
    lo,hi=delayed_interval([np.array([2.5,-5.,1.]),np.ones(3)],3)
    np.testing.assert_array_equal(lo,[0,-5,0]);np.testing.assert_array_equal(hi,[2.5,0,1])


def test_stale_point_is_not_mistaken_for_the_current_state():
    observer=AgeAwareStateObserver('stale_dropout')
    measurement=np.array([[-100.,10.]]*3)
    observer.update(measurement,0,[0.]*3)
    current=observer.update(measurement,[0,0,0],[2.5]*3)
    assert np.all(np.asarray(current['lower'])[:,0] > measurement[:,0]+4.9)
    assert current['age_decisions']==[1,1,1]
    # Exact zero innovation/initial memory gives x=-95, v=10 at t=0.5.
    truth=np.array([[-95.,10.]]*3)
    assert np.all(truth>=current['lower']) and np.all(truth<=current['upper'])
