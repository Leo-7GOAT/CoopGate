"""Fit-only nuisance learning and failure-preserving evaluation for CSPI."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def fit_observed(features,treatment,outcomes,prediction_features):
    features=np.asarray(features,dtype=float);prediction_features=np.asarray(prediction_features,dtype=float)
    treatment=np.asarray(treatment);outcomes=np.asarray(outcomes,dtype=float)
    if (features.ndim!=2 or prediction_features.ndim!=2 or features.shape[1]!=13
            or prediction_features.shape[1]!=13 or treatment.shape!=outcomes.shape
            or treatment.shape!=(len(features),) or not np.isin(treatment,[0,1]).all()
            or not np.isfinite(features).all() or not np.isfinite(prediction_features).all()
            or not np.isfinite(outcomes).all()):raise ValueError('Finite observed fit data required')
    scaler=StandardScaler().fit(features);x=scaler.transform(features);xp=scaler.transform(prediction_features)
    predictions=[];parameters=[]
    for arm in (0,1):
        mask=treatment==arm
        if mask.sum()<8:raise ValueError('Insufficient assigned fitting arm; never redraw')
        model=Ridge(alpha=1.).fit(x[mask],outcomes[mask]);predictions.append(model.predict(xp))
        parameters.append(dict(arm=arm,training_count=int(mask.sum()),coef=model.coef_.tolist(),intercept=float(model.intercept_)))
    return predictions,dict(models=parameters,scaler_mean=scaler.mean_.tolist(),scaler_scale=scaler.scale_.tolist())


def frame(data,predictions):
    m0,m1=predictions;t=np.asarray(data['treatment']);y=np.asarray(data['outcome'])
    if m0.shape!=m1.shape or m0.shape!=t.shape or t.shape!=y.shape:
        raise ValueError('Observed-label and prediction rows differ')
    return pd.DataFrame(dict(treatment=t,propensity=np.full(len(t),.5),outcome=y,
        mu_0_est=m0,mu_1_est=m1,ite_est=m1-m0))


def evaluate(vault,choices):
    """Choices index baseline/nominal/bank in an untouched paired audit table."""
    choices=np.asarray(choices);n=len(choices);indices=np.arange(n)
    if n==0 or choices.shape!=(n,) or not np.isin(choices,[0,1,2]).all():raise ValueError('Fixed complete deployment choices required')
    choices=choices.astype(int)
    expected=dict(team_cost=(n,3),costs=(n,3,3),arrivals=(n,3,3),failed=(n,3),adopted=(n,3))
    for name,shape in expected.items():
        if vault[name].shape!=shape or not np.isfinite(vault[name]).all():raise ValueError('Complete finite paired outcomes required')
    if (not np.isin(vault['failed'],[0,1]).all() or not np.isin(vault['adopted'],[0,1]).all()
            or np.any(vault['costs']<0) or np.any(vault['arrivals']<0) or np.any(vault['arrivals']>25+1e-9)
            or not np.allclose(vault['team_cost'],vault['costs'].mean(axis=-1),atol=1e-9,rtol=0)):
        raise ValueError('Consistent capped physical outcomes required')
    base_cost=vault['team_cost'][:,0];cost=vault['team_cost'][indices,choices]
    if np.any(base_cost<=0):raise ValueError('Positive reference costs required')
    failed=vault['failed'][indices,choices].astype(bool);base_failed=vault['failed'][:,0].astype(bool)
    adopted=vault['adopted'][indices,choices].astype(bool)
    if vault['adopted'][:,0].any():raise ValueError('Baseline cannot adopt learning')
    harm=(failed|base_failed|np.any(vault['costs'][indices,choices]>vault['costs'][:,0]+1e-9,axis=1)
        |np.any(vault['arrivals'][indices,choices]>vault['arrivals'][:,0]+1e-9,axis=1))
    all_complete=not (failed.any() or base_failed.any())
    return dict(evaluation_units=n,controller_selected=int(np.count_nonzero(choices)),
        actual_adopted=int(adopted.sum()),harmful_and_adopted=int((harm&adopted).sum()),
        controller_failures=int(failed.sum()),baseline_failures=int(base_failed.sum()),
        failures_before_any_adoption=int((failed&~adopted).sum()),failed_pairs_removed=0,
        adopted_harm_fraction=float((harm&adopted).sum()/adopted.sum()) if adopted.any() else None,
        baseline_failed_unassessable=int(base_failed.sum()),
        mean_failure_penalized_loss=float(np.mean(cost+25*failed)),
        mean_paired_failure_penalized_gain=float(np.mean(base_cost+25*base_failed-cost-25*failed)),
        all_scope_cost_gain_percent=100*float((base_cost.sum()-cost.sum())/base_cost.sum()) if all_complete else None)
