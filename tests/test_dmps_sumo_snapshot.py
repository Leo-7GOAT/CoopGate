from types import SimpleNamespace

import numpy as np
import pytest

from coopgate.dmps_sumo_snapshot import DMPSSumoSnapshotEnv, Snapshot, _ConfiguredMatched, _ConfiguredReset
from coopgate.disturbance_v9 import DisturbedMergeEnv
from coopgate.merge_v4 import MergeConfig


def fixture():
    env=object.__new__(DMPSSumoSnapshotEnv)
    env.raw=SimpleNamespace(prev=np.zeros(3),c=MergeConfig())
    env.owner='owner';env.generation=1
    return env


def test_reset_dependency_binding_is_on_actual_super_lookup_path():
    chain=_ConfiguredMatched.__mro__
    assert chain.index(DisturbedMergeEnv)<chain.index(_ConfiguredReset)
    assert next(cls for cls in chain[chain.index(DisturbedMergeEnv)+1:] if 'reset' in cls.__dict__) is _ConfiguredReset


def test_action_scaling_and_slew_preserve_physical_limits():
    env=fixture()
    np.testing.assert_array_equal(env.action_to_command([-1,0,1]),[-2.5,-1.25,2.5])
    env.raw.prev=np.array([-5.,2.5,0.])
    np.testing.assert_array_equal(env.action_to_command([1,-1,-1]),[-2.5,0.,-2.5])
    np.testing.assert_allclose(env.action_to_command(env.backup()),[-5.,0.,-2.5],atol=1e-15)


@pytest.mark.parametrize('bad',[[2,0,0],[np.nan,0,0],[0,0],[np.inf,0,0]])
def test_invalid_actions_are_rejected(bad):
    with pytest.raises(ValueError):fixture().action_to_command(bad)


@pytest.mark.parametrize('owner,generation',[('other',1),('owner',0)])
def test_snapshot_from_another_environment_or_reset_is_rejected(owner,generation):
    snapshot=Snapshot(owner,generation,b'',{}, {},MergeConfig().dict(),0.)
    with pytest.raises(ValueError,match='another simulator or reset'):fixture().set_state(snapshot)


def test_changed_configuration_is_rejected_before_native_load():
    snapshot=Snapshot('owner',1,b'',{}, {},dict(horizon=1),0.)
    with pytest.raises(ValueError,match='configuration'):fixture().set_state(snapshot)
