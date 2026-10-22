import numpy as np
import pytest

from coopgate.selective_transfer_v25 import empirical_coupling,transfer_bound


@pytest.mark.parametrize('values',[(.05,.2,0),(.03,.2,.004),(.05,.01,.009)])
def test_bound_and_positive_adoption(values):
    r,a,e=values
    result=transfer_bound(r,a,e)
    assert result['conditional_risk_upper']==min(1,r+e/a)
    assert result['target_adoption_lower']==a-e
    assert result['positive_target_adoption_guaranteed']


def test_zero_adoption_is_not_zero_risk():
    with pytest.raises(ValueError): transfer_bound(0,0,0)
    result=empirical_coupling([0,0,0],[0,1,2])
    assert result['source_risk'] is None and result['empirical_tv_envelope'] is None
    assert result['target_risk']==.5


def test_shift_can_remove_all_adoption():
    result=transfer_bound(.05,.2,.2)
    assert not result['positive_target_adoption_guaranteed']
    assert result['target_adoption_lower']==0
    assert empirical_coupling([1,1,0],[0,0,0])['target_risk'] is None


@pytest.mark.parametrize('position',[0,1,2])
@pytest.mark.parametrize('bad',[-.1,1.1,float('inf'),float('nan'),True])
def test_invalid_supplied_bounds_rejected(position,bad):
    values=[.05,.2,.01];values[position]=bad
    with pytest.raises(ValueError):transfer_bound(*values)


def test_bound_is_sharp_when_good_adoption_mass_moves_to_harmful():
    # Adoption 20%, source risk 5%, and 1% mass transfer imply 10% target risk.
    source=[0]*80+[1]*19+[2]
    target=[0]*80+[1]*18+[2]*2
    result=empirical_coupling(source,target)
    assert result['source_risk']==pytest.approx(.05)
    assert result['target_risk']==pytest.approx(.1)
    assert result['event_total_variation']==pytest.approx(.01)
    assert result['empirical_tv_envelope']['conditional_risk_upper']==pytest.approx(.1)


def test_zero_marginal_shift_does_not_imply_matching_individual_events():
    result=empirical_coupling([0,1,2],[2,0,1])
    assert result['event_total_variation']==0
    assert result['coupled_event_mismatch']==1
    assert result['source_risk']==result['target_risk']==.5


def test_both_bounds_hold_for_5000_arbitrary_joint_probability_tables():
    rng=np.random.default_rng(250251)
    for _ in range(5000):
        joint=rng.dirichlet(np.full(9,.3)).reshape(3,3)
        p,q=joint.sum(1),joint.sum(0)
        a,b=p[1:].sum(),q[1:].sum()
        tv=.5*np.abs(p-q).sum()
        mismatch=1-np.trace(joint)
        bound=transfer_bound(p[2]/a,a,tv)
        assert tv<=mismatch+1e-12
        assert q[2]/b<=bound['conditional_risk_upper']+1e-12
        # The reverse bound also holds; no symmetry of the coupling is assumed.
        assert p[2]/a<=transfer_bound(q[2]/b,b,tv)['conditional_risk_upper']+1e-12


@pytest.mark.parametrize('source,target',[
    ([],[]),([0],[0,1]),([0.,1.],[0,1]),([True],[0]),([3],[0]),([-1],[0])])
def test_invalid_event_pairs_rejected(source,target):
    with pytest.raises(ValueError):empirical_coupling(source,target)
