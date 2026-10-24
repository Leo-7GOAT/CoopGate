from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from probe_dmps_sumo_snapshot_v2 import trajectory_equivalent


def test_128bit_rng_states_compare_exactly_without_float_cast():
    state={'rng':{'state':2**127+12345},'clock':1.5,'flag':False}
    trajectory_equivalent(state,state)
    with pytest.raises(AssertionError,match='rng.state'):
        trajectory_equivalent(state,{**state,'rng':{'state':2**127+12346}})


def test_integer_counters_are_not_tolerance_compared():
    with pytest.raises(AssertionError):trajectory_equivalent(1,1.)
    with pytest.raises(AssertionError):trajectory_equivalent(True,1)


def test_float_tolerance_and_complete_nested_shape_remain_enforced():
    trajectory_equivalent({'x':[1.,2.]},{'x':[1.+1e-10,2.]})
    with pytest.raises(AssertionError):trajectory_equivalent({'x':[1.,2.]},{'x':[1.+1e-6,2.]})
    with pytest.raises(AssertionError):trajectory_equivalent({'x':[1.,2.]},{'x':[1.]})
