import numpy as np
import pytest
from scipy.stats import binom
from coopgate.admission_risk_v15 import binomial_upper,validate_gate_family


def test_exact_bound_covers_the_binomial_population_without_monte_carlo():
    n=20;delta=.025
    bounds=np.array([binomial_upper(k,n,delta) for k in range(n+1)])
    for risk in (.01,.05,.2,.7,.95):
        failure_probability=binom.pmf(np.arange(n+1),n,risk)[bounds<risk].sum()
        assert failure_probability<=delta+1e-12


def test_few_admissions_cannot_hide_behind_a_large_rejected_denominator():
    admitted=np.zeros((1000,2),bool);admitted[:10,0]=True;admitted[:300,1]=True
    harmful=np.zeros_like(admitted);gain=admitted.astype(float);gain[:10,0]=100.
    result=validate_gate_family(range(1000),('sparse','supported'),admitted,harmful,gain)
    assert result['gates'][0]['all_scene_harm_rate']==0.
    assert not result['gates'][0]['certified']
    assert result['gates'][1]['certified'] and result['selected_gate']=='supported'


def test_zero_admission_is_not_a_zero_conditional_risk_certificate():
    result=validate_gate_family(range(100),('none',),np.zeros((100,1),bool),np.zeros((100,1),bool),np.zeros((100,1)))
    assert result['selected_gate'] is None
    assert result['gates'][0]['conditional_empirical_risk'] is None
    assert result['gates'][0]['conditional_risk_upper']==1.


def test_repeated_initials_and_gain_from_rejected_scenes_are_rejected():
    with pytest.raises(ValueError,match='Unique independent'):
        validate_gate_family([1,1],('g',),np.ones((2,1),bool),np.zeros((2,1),bool),np.ones((2,1)))
    with pytest.raises(ValueError,match='gated gain'):
        validate_gate_family([1,2],('g',),np.zeros((2,1),bool),np.zeros((2,1),bool),np.ones((2,1)))
