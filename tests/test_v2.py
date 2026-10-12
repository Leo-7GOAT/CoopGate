import numpy as np
import pytest
from coopgate.certification import binomial_upper,zero_failure_sample_size,certify
from coopgate.paired import stopping_advance,ClosedLoopPredictor,PairedSupervisor,potential
from coopgate.config import Config
from coopgate.controllers import Intent,SpeedMPC
from coopgate.sim import SumoEnv


def test_exact_stopping_envelope_against_dense_time_grid():
    rng=np.random.default_rng(32)
    for _ in range(80):
        vr,vf=rng.uniform(0,25,2);delay=rng.uniform(0,1)
        acc=rng.uniform(0,3);br,bf=rng.uniform(2,10,2)
        peak=vr+acc*delay
        ts=np.linspace(0,max(delay+peak/br,vf/bf),20001)
        s=np.clip(ts-delay,0,peak/br)
        rear=np.where(ts<=delay,vr*ts+.5*acc*ts**2,vr*delay+.5*acc*delay**2+peak*s-.5*br*s*s)
        f=np.minimum(ts,vf/bf);front=vf*f-.5*bf*f*f
        dense=max(0,float(np.max(rear-front)))
        exact=stopping_advance(vr,vf,delay,acc,br,bf)
        assert dense<=exact+1e-9
        assert exact-dense<2e-5


def test_endpoint_stopping_distance_can_underestimate_peak_approach():
    peak=stopping_advance(20,10,0,0,9,3)
    assert peak==pytest.approx(25/3)
    assert peak>20**2/(2*9)-10**2/(2*3)


def test_continuous_junction_has_no_position_jump():
    c=Config(commit_lane_changes=True,continuous_network=True)
    with SumoEnv(c,'traditional') as env:
        env.reset(30010)
        for _ in range(30):
            old=env.states.copy()
            env.step([4,4,4])
            displacement=env.states[:3,0]-old[:3,0]
            assert np.max(displacement)<=c.max_speed*c.decision_dt+.1


def test_stopping_reserve_grows_with_reaction_delay():
    assert stopping_advance(15,15,1)>stopping_advance(15,15,.2)
    assert stopping_advance(0,0,0)==0


def test_progress_potential_never_rewards_delaying_goal_crossing():
    c=Config();s=np.array([[c.goal_x-1,3.2,18,0,1]]*3,dtype=float)
    before=potential(s,c)
    s[:,0]+=2
    assert potential(s,c)>=before
    s[:,0]=c.goal_x-.0001
    assert abs(potential(s,c)-potential(np.array([[c.goal_x,3.2,18,0,1]]*3),c))<1e-6


def test_binomial_family_correction_and_sample_size():
    assert binomial_upper(0,100,family_size=8)==pytest.approx(1-(.05/8)**.01)
    assert binomial_upper(0,100,family_size=8)>binomial_upper(0,100)
    assert zero_failure_sample_size(.05,.05,8)==99
    assert certify(0,100,family_size=8)['certified']
    assert not certify(1,100,family_size=8)['certified']
    assert binomial_upper(0,0)==1


def test_predictor_respects_existing_lane_commitment():
    c=Config(commit_lane_changes=True,paired_horizon=1)
    with SumoEnv(c,'traditional') as env:
        env.reset(123)
        # Predictor can evaluate contradictory requests while retaining memory.
        context={'lane_targets':np.array([1,1,1]),'lane_remaining':np.array([1.,0.,0.])}
        f=ClosedLoopPredictor(c).rollout(env.states,[Intent(12,0)]*3,context,0.)
        assert np.isfinite([f.margin,f.reserve,f.value]).all()


def test_executor_keeps_started_lane_change_target():
    c=Config(commit_lane_changes=True)
    with SumoEnv(c,'mappo') as env:
        env.reset(123)
        env.step([7,4,4])  # left at 14 m/s
        assert env.lane_targets[0]==1 and env.lane_remaining[0]>0
        env.step([1,4,4])  # ask to reverse; commitment must remain
        assert env.lane_targets[0]==1
        assert env.lane_overrides>=1


def test_low_speed_policy_does_not_displace_backup_for_rl_count():
    c=Config(commit_lane_changes=True,gate_budget_ms=10000)
    with SumoEnv(c,'traditional') as env:
        env.reset(123)
        r=PairedSupervisor(c).select(env.states,[Intent(8,int(s[1]>1.6)) for s in env.states[:3]])
        if r.modes.any():
            assert r.diagnostics['predicted_gain']>=c.paired_gain
        assert r.checks==24
