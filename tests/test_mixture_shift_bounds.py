import numpy as np
import pytest
from scipy.optimize import linprog

from coopgate.mixture_shift_bounds import mixture_box_envelope, conditional_risk_under_mixture_shift
from coopgate.stratified_risk import conditional_risk_upper


def test_single_fixed_mixture_matches_frozen_bound():
    fixed=conditional_risk_upper([1,0,0,0],[300]*4,[768]*4,[.25]*4,alpha=.05/6)
    shifted=conditional_risk_under_mixture_shift([1,0,0,0],[300]*4,[768]*4,
        [.25]*4,[.25]*4,alpha=.05/6)
    assert shifted['risk_upper']==pytest.approx(fixed['risk_upper'],abs=1e-15)
    assert shifted['adoption_lower']==pytest.approx(fixed['adoption_lower'],abs=1e-15)
    assert shifted['vertices_checked']==1
    assert not shifted['gain_certificate_transferred']
    assert not shifted['joint_deployment_certified']


def test_entire_simplex_and_zero_adoption_fail_closed():
    result=mixture_box_envelope([.01,.04],[.5,.25],[0,0],[1,1])
    assert result['risk_upper']==pytest.approx(.16)
    assert result['adoption_lower']==.25
    zero=mixture_box_envelope([0,0],[.5,0],[0,0],[1,1])
    assert zero['risk_upper']==1 and zero['adoption_lower']==0
    assert zero['zero_denominator_feasible']


@pytest.mark.parametrize('low,high', [([.6,.6],[1,1]),([0,0],[.4,.4]),
    ([.7,0],[.6,1]),([float('nan'),0],[1,1]),([0],[1,1])])
def test_invalid_or_infeasible_weight_sets_rejected(low,high):
    with pytest.raises(ValueError):
        mixture_box_envelope([.01,.02],[.4,.5],low,high)


def test_vertex_result_matches_independent_linear_fractional_program():
    rng=np.random.default_rng(460047)
    for _ in range(12):
        q=rng.uniform(.001,.05,4); a=rng.uniform(.2,.8,4)
        low=rng.uniform(0,.1,4); high=rng.uniform(.5,.9,4)
        result=mixture_box_envelope(q,a,low,high)
        # Charnes-Cooper variables: y=w/(a.w), t=1/(a.w).
        objective=np.r_[-q,0.]
        equality=np.array([np.r_[a,0.],np.r_[np.ones(4),-1.]])
        constraints=np.vstack([np.column_stack([np.eye(4),-high]),
                               np.column_stack([-np.eye(4),low])])
        lp=linprog(objective,A_ub=constraints,b_ub=np.zeros(8),
            A_eq=equality,b_eq=[1.,0.],bounds=[(0,None)]*5,method='highs')
        assert lp.success
        assert result['risk_upper']==pytest.approx(-lp.fun,abs=1e-10)
        adoption=linprog(a,A_eq=np.ones((1,4)),b_eq=[1.],bounds=list(zip(low,high)),method='highs')
        assert adoption.success
        assert result['adoption_lower']==pytest.approx(adoption.fun,abs=1e-10)
        for _ in range(20):
            w=rng.dirichlet(np.ones(4))
            if np.all(w>=low) and np.all(w<=high):
                assert w@q/(w@a) <= result['risk_upper']+1e-14
                assert w@a >= result['adoption_lower']-1e-14


def test_nominal_mixture_success_does_not_cover_different_frequencies():
    k,a,n=[50,0,0,0],[768]*4,[768]*4
    nominal=conditional_risk_under_mixture_shift(k,a,n,[.25]*4,[.25]*4,alpha=.05/6)
    shifted=conditional_risk_under_mixture_shift(k,a,n,[.75,0,0,0],[.75,.25,.25,.25],alpha=.05/6)
    assert nominal['risk_and_adoption_bounds_pass']
    assert shifted['risk_upper']>.05
    assert not shifted['risk_and_adoption_bounds_pass']


def test_adoption_must_also_survive_weight_change():
    shifted=conditional_risk_under_mixture_shift([0]*4,[768,768,768,16],[768]*4,
        [0,0,0,.9],[.1,.1,.1,.9],alpha=.05/6)
    assert shifted['adoption_lower']<.2
    assert not shifted['risk_and_adoption_bounds_pass']


def test_stratum_permutation_preserves_envelope():
    q=np.array([.01,.02,.03,.04]); a=np.array([.3,.4,.5,.6])
    low=np.array([.05,.10,.15,.20]); high=np.array([.6,.5,.4,.5])
    original=mixture_box_envelope(q,a,low,high)
    order=[3,0,2,1]
    other=mixture_box_envelope(q[order],a[order],low[order],high[order])
    assert original['risk_upper']==other['risk_upper']
    assert original['adoption_lower']==other['adoption_lower']
