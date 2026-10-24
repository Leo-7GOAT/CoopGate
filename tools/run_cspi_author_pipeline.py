"""Run all three author synthetic DGPs once through the pinned CSPI pipeline.

This is a development integration check, not a paper-scale replication or a
CoopGate efficacy study. No upstream function body is edited or vendored.
"""
import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits


BLOB = '84323a8cf90351cf69fbbc59300523ddca54f625'
COMMIT = 'b5af605a38a7866c52745c153e02cb4bf049ea01'
ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def functions(source):
    raw = source.read_bytes()
    assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() == BLOB
    tree = ast.parse(raw.decode('utf-8'))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    assert len(selected) == 13
    namespace = dict(np=np, pd=pd, stats=stats, LinearRegression=LinearRegression)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace


def verify_estimator(f, data, cutoffs, baseline):
    # Verify independently from the policy indicators, not the interval code.
    treatment, propensity = data.treatment.to_numpy(), data.propensity.to_numpy()
    outcome = data.outcome.to_numpy()
    m0, m1 = data.mu_0_est.to_numpy(), data.mu_1_est.to_numpy()
    influence = m1-m0+treatment/propensity*(outcome-m1)-(1-treatment)/(1-propensity)*(outcome-m0)
    scores = np.array([((data.ite_est.to_numpy() >= c).astype(float)
                       -(data.ite_est.to_numpy() >= baseline).astype(float))*influence for c in cutoffs])
    estimate, covariance = f['PDE'](cutoffs, baseline, data, by_quantile=False, dr=True)
    np.testing.assert_allclose(estimate, scores.mean(axis=1), atol=1e-12, rtol=0)
    np.testing.assert_allclose(covariance, np.cov(scores)/len(data), atol=1e-12, rtol=0)


def run(source, output):
    f = functions(source)
    for name in ('tools/run_cspi_author_pipeline.py','docs/cspi_author_pipeline_development.md'):
        committed = subprocess.run(['git','show','HEAD:'+name],cwd=ROOT,check=True,capture_output=True).stdout
        if committed != (ROOT/name).read_bytes():
            raise ValueError('Commit the exact pipeline and protocol before execution')
    cutoffs, baseline, gamma = np.linspace(2., -1.9, 50), -2., .05
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(upstream_commit=COMMIT, upstream_git_blob=BLOB,
        upstream_sha256=digest(source), author_function_count=13, function_bodies_modified=False,
        driver_sha256=digest(Path(__file__)), protocol_sha256=digest(ROOT/'docs/cspi_author_pipeline_development.md'),
        python=sys.version, runtime={p:importlib.metadata.version(p) for p in ('numpy','scipy','pandas','scikit-learn','threadpoolctl')},
        seed=42, settings=['linear','roof','negative_constant'], repetitions_per_setting=1,
        tuning_rows=2000, testing_rows=8000, gamma=gamma, cutoffs=cutoffs.tolist(), baseline_cutoff=baseline,
        gaussian_draws=10000, nuisance_folds=5, sumo_runs=0,
        selection_and_test_rows_disjoint=True, nuisance_fits_use_pooled_cross_folds=True,
        finite_sample_risk_claimed=False, comparative_efficacy_claimed=False)
    write(output/'protocol.json',protocol)
    reports=[]
    for setting in protocol['settings']:
        np.random.seed(42)
        if setting == 'roof':
            data=f['generate_data_roof'](n=10000)
        else:
            data=f['generate_data_lin'](n=10000, effect_coef=1 if setting=='linear' else 0,
                                       constant_eff=0 if setting=='linear' else -.25)
        tune, test=data.iloc[:2000].copy(),data.iloc[2000:].copy()
        assert set(tune.index).isdisjoint(test.index)
        assert not np.isin(data.ite_est.to_numpy(), np.append(cutoffs,baseline)).any()
        np.savez_compressed(output/(setting+'_data.npz'),**{c:data[c].to_numpy() for c in data.columns})
        verify_estimator(f,test,cutoffs,baseline)
        single, single_dr, mt_e, mt = f['cutoff_selection'](cutoffs,baseline,tune,gamma=gamma,n_sim=10000,test_ratio=4)
        finite, asymp=f['cutoff_selection_HCPI'](cutoffs,baseline,tune,gamma=gamma,test_ratio=4)
        methods=[]
        def add(name, selected, passed, chosen=None):
            selected=np.asarray(selected,dtype=float)
            passed=np.asarray(passed,dtype=bool)
            assert selected.shape == passed.shape and np.isfinite(selected).all()
            if chosen is None:
                assert len(selected)==1
                chosen=float(selected[0]) if passed[0] else baseline
            if setting=='linear': truth=.5-chosen**2/8
            elif setting=='roof': truth=float(f['true_policy_diff_roof'](np.array([chosen]))[0])
            else: truth=.25*(.5+chosen/4)
            methods.append(dict(method=name,selected_cutoffs=selected.tolist(),test_passes=passed.tolist(),
                                deployed_cutoff=chosen,true_population_gain=float(truth)))
        for suffix, selected, dr in [('ipw',single,False),('dr',single_dr,True)]:
            passed=f['safety_test_single'](selected,baseline,test,gamma=gamma,dr=dr)
            for j,label in enumerate(('passing_probability','expected_improvement')):
                add(suffix+'_'+label,selected[j:j+1],passed[j:j+1])
        for name, selected in [('cspi_mt_e',mt_e),('cspi_mt',mt)]:
            passed=f['safety_test'](selected,baseline,test,gamma=gamma,by_quantile=False,nsim=10000,dr=True)
            choices=np.asarray(selected)[np.asarray(passed,dtype=bool)]
            if len(choices):
                values,_=f['PDE'](choices,baseline,data,by_quantile=False,dr=True)
                chosen=float(choices[np.argmax(values)])
            else: chosen=baseline
            add(name,selected,passed,chosen)
        add('hcpi_finite',finite,f['safety_test_finite_sample'](finite,baseline,test,gamma=gamma))
        add('hcpi_asymptotic',asymp,f['safety_test_single'](asymp,baseline,test,gamma=gamma,dr=False))
        values,_=f['PDE'](cutoffs,baseline,data,by_quantile=False,dr=True)
        idx=int(np.argmax(values)); add('naive',cutoffs[idx:idx+1],np.array([values[idx]>0]))
        report=dict(setting=setting,methods=methods,independent_indicator_estimator_verified=True,
                    data_sha256=digest(output/(setting+'_data.npz')))
        write(output/(setting+'_result.json'),report)
        reports.append(report)
        print(json.dumps(dict(setting=setting,methods_completed=len(methods))),flush=True)
    write(output/'complete.json',dict(settings_completed=3,method_outputs=27,
        data_rows=30000,not_independent_repetitions=27,substantive_error_rate_estimated=False,
        files_sha256={p.name:digest(p) for p in output.iterdir() if p.is_file()}))
    return reports


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.source,args.output)
