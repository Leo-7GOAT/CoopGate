import copy

import numpy as np
import pytest

from cspi_factorial_analysis import evaluate,fit_observed,frame
from design_cspi_factorial_comparison import assignments


def test_assignments_are_complete_reproducible_and_outcome_free():
    cases=[dict(cluster_id=str(i),seed=i) for i in range(640)]
    a=assignments(cases)
    assert a==assignments(cases)
    assert {r['block_index'] for r in a}==set(range(640))
    assert [sum(r['split']==s for r in a) for s in ('fit','tune','safety_test','evaluation')]==[256,128,128,128]
    changed=[dict(c,unused_outcome=1000-i) for i,c in enumerate(cases)]
    assert a==assignments(changed)
    with pytest.raises(ValueError):assignments(cases[:-1])


def test_prediction_points_never_change_fitted_scaler_or_nuisance_models():
    rng=np.random.default_rng(905700);x=rng.normal(size=(256,13));t=np.arange(256)%2;y=x[:,0]+t
    points=rng.normal(size=(128,13));changed=points.copy();changed[1:]*=1e6
    predictions,parameters=fit_observed(x,t,y,points)
    other,other_parameters=fit_observed(x,t,y,changed)
    assert parameters==other_parameters
    np.testing.assert_array_equal([p[0] for p in predictions],[p[0] for p in other])
    with pytest.raises(ValueError):fit_observed(x,np.zeros(256),y,points)


def vault():
    costs=np.full((3,3,3),10.)
    return dict(team_cost=costs.mean(axis=-1),costs=costs,arrivals=costs.copy(),
        failed=np.zeros((3,3),bool),adopted=np.array([[0,1,1]]*3,bool))


def test_pre_adoption_failure_stays_in_penalized_loss_and_blocks_gain():
    data=vault();data['failed'][0,1]=True;data['adopted'][0,1]=False
    result=evaluate(data,[1,1,1])
    assert result['controller_failures']==result['failures_before_any_adoption']==1
    assert result['failed_pairs_removed']==0 and result['all_scope_cost_gain_percent'] is None
    assert result['mean_failure_penalized_loss']==pytest.approx(10+25/3)


def test_failed_reference_is_unassessable_even_if_controller_finishes():
    data=vault();data['failed'][0,0]=True
    result=evaluate(data,[1,1,1])
    assert result['baseline_failed_unassessable']==1 and result['controller_failures']==0
    assert result['harmful_and_adopted']==1 and result['all_scope_cost_gain_percent'] is None


def test_team_improvement_does_not_hide_one_vehicle_harm():
    data=vault();data['costs'][0,1]=[5.,5.,11.];data['team_cost']=data['costs'].mean(axis=-1)
    result=evaluate(data,[1,1,1])
    assert result['all_scope_cost_gain_percent']>0 and result['harmful_and_adopted']==1


def test_invalid_or_missing_paired_outcome_cannot_disappear():
    data=vault()
    with pytest.raises(ValueError):evaluate(data,[1.9,1,1])
    bad=copy.deepcopy(data);bad['costs']=bad['costs'][:-1]
    with pytest.raises(ValueError):evaluate(bad,[1,1,1])
    bad=copy.deepcopy(data);bad['team_cost'][0,1]=np.nan
    with pytest.raises(ValueError):evaluate(bad,[1,1,1])


def test_expected_gain_does_not_identify_individual_harm_from_randomized_logs():
    # Identical arm marginals, different admissible joint potential outcomes.
    diagonal=np.array([[0.,0.],[1.,1.]])
    crossed=np.array([[0.,1.],[1.,0.]])
    np.testing.assert_array_equal(np.sort(diagonal,axis=0),np.sort(crossed,axis=0))
    assert np.mean(diagonal[:,1]-diagonal[:,0])==np.mean(crossed[:,1]-crossed[:,0])==0
    assert np.mean(diagonal[:,1]>diagonal[:,0])==0
    assert np.mean(crossed[:,1]>crossed[:,0])==.5


def test_fixed_split_core_runs_pinned_original_selection_and_safety_tests():
    from pathlib import Path
    import cspi_supported_candidates as adapter
    import run_cspi_author_pipeline as author
    source=Path(__file__).resolve().parents[2]/'reference_sources/cspi-mt-b5af605/Figures_2_and_3_for_KDD.py'
    if not source.exists():pytest.skip('Pinned original CSPI source unavailable')
    functions=author.functions(source);rng=np.random.default_rng(905710)
    x=rng.normal(size=(512,13));t=rng.binomial(1,.5,512)
    y=-10+.2*x[:,0]+t*(2+.2*x[:,1])+rng.normal(0,.1,512)
    predictions,_=fit_observed(x[:256],t[:256],y[:256],x)
    fit_scores=predictions[1][:256]-predictions[0][:256]
    tune=frame(dict(treatment=t[256:384],outcome=y[256:384]),[p[256:384] for p in predictions])
    selected=adapter.choose(functions,fit_scores,tune,905720)
    test=frame(dict(treatment=t[384:],outcome=y[384:]),[p[384:] for p in predictions])
    result=adapter.test_selected(functions,selected,test,905730)
    assert {m['method'] for m in result['methods']}=={'cspi','cspi_mt'}
    assert all(m['status']=='tested' for m in result['methods'])
    assert not result['new_risk_certificate']
