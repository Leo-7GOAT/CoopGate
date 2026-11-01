import json
from pathlib import Path

import numpy as np
import pytest
import torch

from coopgate.mappo_ctde import PPOConfig
from coopgate.mappo_deployment import load_deployment_actor,make_recovery_gate
from coopgate.actuator_recovery_v34 import ActuatorRecoveryBaseline


def test_baseline_adapter_preserves_actual_fallback_command_and_history():
    state=np.array([[-140.,12.],[-220.,10.],[-190.,12.]])
    for profile in ('nominal','gain_drag','lag_mild','lag_strong'):
        original=ActuatorRecoveryBaseline(profile)
        adapted=make_recovery_gate(None,profile,'baseline')
        command,_=original.select(state,np.zeros(3))
        converted,info=adapted.select(state,np.zeros(3))
        np.testing.assert_array_equal(converted,command)
        assert not info['actual_learning_used']
        assert not info['conditional_performance_risk_certified']


def test_loader_requires_actual_joint_ppo_audit_and_detects_changed_weights(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'tools'))
    from train_mappo_ctde import train
    from audit_mappo_ctde import audit
    out=tmp_path/'actual_joint_smoke'
    cfg=PPOConfig(width=16,n_envs=4,rollout=16,epochs=2,minibatch=32,role_actors=True)
    train(out,128,77,'joint',config=cfg)
    with pytest.raises(ValueError,match='audit required'):
        load_deployment_actor(out/'policy.pt')
    audit(out)
    model,proof=load_deployment_actor(out/'policy.pt')
    assert proof['joint_steps']==128 and not proof['inherited_risk_certificate']
    gate=make_recovery_gate(model,'nominal','mamps_shared')
    _,info=gate.select(np.array([[-140.,12.],[-220.,10.],[-190.,12.]]),np.zeros(3))
    assert not info['conditional_performance_risk_certified']
    data=(out/'policy.pt').read_bytes();(out/'policy.pt').write_bytes(data+b'changed')
    with pytest.raises(ValueError,match='checkpoint changed'):
        load_deployment_actor(out/'policy.pt')
