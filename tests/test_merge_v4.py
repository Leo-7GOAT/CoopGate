import numpy as np
from coopgate.merge_v4 import MergeConfig, MergeGate, MergeEnv, transition_safe, backup, certify, advance


def test_zone_overlap_between_endpoints_is_detected():
    c=MergeConfig()
    # Vehicle 0 exits while vehicle 1 enters: both endpoints alone look exclusive.
    p=np.array([[5.6,-26.,-100.]])
    q=np.array([[8.,-24.,-100.]])
    assert not transition_safe(p,q,c)[0]


def test_safe_temporally_separated_zone_handoff():
    c=MergeConfig()
    p=np.array([[6.7,-27.,-100.]])
    q=np.array([[8.7,-25.,-100.]])
    assert transition_safe(p,q,c)[0]


def test_joint_certificate_preserves_recovery_under_random_proposals():
    rng=np.random.default_rng(2901);c=MergeConfig();g=MergeGate(c,'joint_rl')
    for _ in range(5):
        x=np.array([[-80.,12.],[-85.,13.],[-120.,13.]])
        previous=np.zeros(3)
        for _ in range(35):
            command,info=g.select(x,previous,rng.uniform(-5,2.5,3))
            assert info['in_set'] and info['certified']
            assert np.all(np.abs(command-previous)<=2.5+1e-9)
            y=x[None,:,:].copy();assert advance(y,command[None,:],c)[0]
            x=y[0];previous=command
            assert certify(x,previous,backup(previous,c),c)[0]


def test_actual_sumo_merge_matches_declared_route_model():
    with MergeEnv(method='eta') as env:
        env.reset(100)
        while True:
            _,_,t,u,m=env.step([4,4,4])
            if t or u:break
        assert m['completed'] and not m['collision']
        assert m['contract_violations']==m['outside_steps']==m['uncertified_steps']==0
        assert m['max_prediction_error']<1e-6
