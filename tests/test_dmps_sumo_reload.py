from pathlib import Path
from types import SimpleNamespace

import pytest

from coopgate.dmps_sumo_reload import DMPSReloadEnv,ControlledSnapshot
from coopgate.merge_v4 import MergeConfig


def fixture():
    env=object.__new__(DMPSReloadEnv)
    env.owner='owner';env.generation=1;env.seed=139892001
    env.raw=SimpleNamespace(c=MergeConfig(),d=Path('test-placeholder'))
    return env


def test_reload_restores_original_clock_and_keeps_physical_options():
    env=fixture();state=SimpleNamespace(simulator_time=.1)
    args=env.reload_args(state,Path('saved.xml'))
    assert args[args.index('--begin')+1]=='0.1'
    assert args[args.index('--load-state')+1]=='saved.xml'
    assert args[args.index('--step-length')+1]=='0.1'
    assert args[args.index('--collision.action')+1]=='warn'
    assert args[args.index('--time-to-teleport')+1]=='-1'


def test_missing_control_modes_are_not_silently_defaulted():
    env=fixture()
    state=ControlledSnapshot('owner',1,b'',{'ids':['ego0','ego1','ego2']},{},env.raw.c.dict(),.1,(96,),(0,))
    with pytest.raises(ValueError,match='Incomplete TraCI'):env.set_state(state)
