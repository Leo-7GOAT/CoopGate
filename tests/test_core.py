import numpy as np
import pytest

from coopgate.config import Config
from coopgate.controllers import SpeedMPC,Intent,decode_action
from coopgate.mappo import compute_gae,MAPPO
from coopgate.shield import Supervisor,rollout_margin
from coopgate.experiment import wilson


def states():
    return np.array([[50,0,12,0,1],[100,3.2,12,0,1],[10,3.2,12,0,1],
                     [170,3.2,12,0,1],[0,0,8,0,1]],float)


def test_mpc_acceleration_speed_and_slew():
    cfg=Config();mpc=SpeedMPC(cfg)
    v,previous=0.,0.
    for _ in range(40):
        a=mpc.control(v,18,previous)
        assert -cfg.max_brake-1e-4 <= a <= cfg.max_accel+1e-4
        assert abs(a-previous) <= cfg.max_jerk*cfg.decision_dt+1e-3
        v+=a*cfg.decision_dt
        assert -1e-3<=v<=cfg.max_speed+1e-3
        previous=a
    assert abs(v-18)<.2
    assert mpc.failures==0


def test_collision_is_rejected_at_current_state():
    s=states();s[1,:3]=s[0,:3]
    margins=rollout_margin(s,[Intent(0,0)]*3,Config())
    assert margins[0]<0 and margins[1]<0


def test_joint_gate_never_accepts_failed_rollout():
    cfg=Config();s=states()
    proposed=[Intent(20,1)]*3
    result=Supervisor(cfg,'joint_robust').select(s,proposed)
    assert result.checks>=8*3
    if not result.emergency:
        for a in (-cfg.background_brake,0,1.5):
            assert rollout_margin(s,result.intents,cfg,a).min()>=0


def test_no_feasible_mode_marks_emergency_not_certificate():
    s=states();s[1,:3]=s[0,:3]
    r=Supervisor(Config(),'joint_robust').select(s,[Intent(20,1)]*3)
    assert r.emergency
    assert not r.modes.any()
    assert all(i.speed==0 for i in r.intents)


def test_gae_bootstraps_timeout_but_not_terminal():
    a,_=compute_gae(np.array([1.,1.]),np.array([0.,0.]),np.array([10.,10.]),
                    np.array([False,True]),np.array([True,True]),gamma=.9)
    np.testing.assert_allclose(a,[10,1])


def test_policy_shapes_and_valid_actions():
    policy=MAPPO();a,lp,v=policy.act(np.zeros((3,25),np.float32))
    assert a.shape==(3,) and lp.shape==(3,)
    assert ((a>=0)&(a<9)).all() and np.isfinite(v)


def test_zero_collisions_does_not_mean_zero_risk():
    low,high=wilson(0,30)
    assert abs(low)<1e-12 and high>.10


def test_decoder_respects_lane_boundaries():
    assert decode_action(0,0,10,Config()).lane==0
    assert decode_action(8,1,10,Config()).lane==1
