import numpy as np
from numba import types
from numba.typed import Dict

from coopgate import buffered_baseline_v25 as original
from coopgate import certificate_cached_recovery as cached
from coopgate.fast_v8 import parameters
from coopgate.merge_v4 import MergeConfig


def test_cached_planner_preserves_exact_commands_costs_work_and_capacity_behavior():
    packed=parameters(MergeConfig())
    for capacity in (0,2,cached.CERTIFICATE_CAPACITY):
        cache=Dict.empty(key_type=cached.CERTIFICATE_KEY,value_type=types.boolean)
        stats=np.zeros(4,np.int64)
        for offset in (0.,20.,60.):
            state=np.array([[-140.+offset,12.],[-220.+offset,10.],[-190.+offset,12.]])
            expected=original.deduplicated_joint_rollout(state,np.zeros(3),0,packed)
            actual=cached.deduplicated_joint_rollout(state,np.zeros(3),0,packed,cache,stats,capacity)
            for a,b in zip(actual,expected):np.testing.assert_equal(a,b)
        assert stats[0]==stats[1]+stats[2] and len(cache)<=capacity
        if capacity>2:assert stats[1]>0


def test_certificate_key_separates_bit_changes_in_each_input_family():
    state=np.array([[-140.,12.],[-220.,10.],[-190.,12.]])
    previous=np.zeros(3);command=np.zeros(3);packed=parameters(MergeConfig())
    cache=Dict.empty(key_type=cached.CERTIFICATE_KEY,value_type=types.boolean);stats=np.zeros(4,np.int64)
    cached.cached_certificate(state,previous,command,packed,.5,cache,stats,100)
    for array in (state,previous,command,packed):
        for index in range(array.size):
            old=array.flat[index];array.flat[index]=np.nextafter(old,np.inf)
            before=stats[2]
            cached.cached_certificate(state,previous,command,packed,.5,cache,stats,100)
            assert stats[2]==before+1
            array.flat[index]=old
    before=stats[2]
    cached.cached_certificate(state,previous,command,packed,np.nextafter(.5,np.inf),cache,stats,100)
    assert stats[2]==before+1
