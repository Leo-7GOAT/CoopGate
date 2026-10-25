from copy import deepcopy

import numpy as np
import pytest

from coopgate.aged_observation_envelope import AgeAwareStateObserver, estimate_frames


def test_every_public_prefix_matches_the_complete_streams_same_prefix():
    frames=[]
    for step in range(10):
        source=max(0,step-2)
        frames.append(dict(observed_state=[[-100.+5*source,10.]]*3,
            perception=dict(profile='stale_dropout',decision=step,source_decision=[source]*3),
            previous_command=[0.]*3))
    full=estimate_frames(frames,'stale_dropout')
    for length in range(1,11):
        assert estimate_frames(frames[:length],'stale_dropout')==full[:length]


def test_receipt_time_cannot_silently_replace_a_stale_measurement_source_time():
    observer=AgeAwareStateObserver('stale_dropout')
    raw=[[-100.,10.]]*3
    observer.update(raw,0,[0.]*3)
    with pytest.raises(ValueError,match='contradicts'):
        observer.update(raw,1,[0.]*3)  # False claim that the old sample is current.


def test_received_state_timestamps_and_commands_are_never_mutated():
    observed=np.array([[-100.,20.199],[-130.,-.1],[-160.,10.]])
    source=np.zeros(3,dtype=int);previous=np.zeros(3)
    before=deepcopy((observed,source,previous))
    AgeAwareStateObserver('bounded_bias_jitter').update(observed,source,previous)
    for actual,original in zip((observed,source,previous),before):
        np.testing.assert_array_equal(actual,original)
