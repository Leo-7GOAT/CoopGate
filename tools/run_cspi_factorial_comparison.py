"""Run unchanged author selection/tests on the prospectively split export.

Use the separate baseline Python runtime. Evaluation outcomes are decoded only
after selection, safety testing and deployment thresholds are written to disk.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

import cspi_factorial_analysis as core
import cspi_supported_candidates as adapter
import design_cspi_factorial_comparison as design
import run_cspi_author_pipeline as author


def load(path):
    with np.load(path,allow_pickle=False) as archive:return {name:archive[name] for name in archive.files}


def run(source):
    plan=design.inputs();design.verify_upstream(source);functions=author.functions(source)
    data=design.OUT/'data';complete=design.read(data/'complete.json');design.committed(data/'complete.json')
    assert complete['supplement_plan_sha256']==design.digest(design.PLAN)
    assert complete['parent_plan_sha256']==design.PARENT_SHA and complete['parent_runs_verified']==3200
    for name,expected in complete['files_sha256'].items():
        if design.digest(data/name)!=expected:raise ValueError('Export changed: '+name)
    output=design.OUT/'analysis';output.mkdir(exist_ok=False)
    design.write(output/'protocol.json',dict(plan_sha256=design.digest(design.PLAN),export_sha256=design.digest(data/'complete.json'),
        upstream_sha256=design.digest(source),python=sys.version,executable=str(Path(sys.executable).resolve()),
        runtime={p:importlib.metadata.version(p) for p in ('numpy','scipy','pandas','scikit-learn','threadpoolctl')},
        methods=plan['contract']['methods'],new_risk_certificate=False))
    fit=load(data/'fit.npz');tune=load(data/'tune.npz')
    # Safety/evaluation labels are not passed to fit_observed or choose.
    predictions,parameters=core.fit_observed(fit['features'],fit['treatment'],fit['outcome'],
        np.concatenate((fit['features'],tune['features'])))
    design.write(output/'nuisance.json',parameters)
    fit_scores=predictions[1][:256]-predictions[0][:256]
    tuning=core.frame(tune,[p[256:] for p in predictions])
    try:selection=adapter.choose(functions,fit_scores,tuning,805720)
    except (ValueError,RuntimeWarning,np.linalg.LinAlgError) as error:
        payload=dict(selected=dict(cspi=[],cspi_mt=[]),route='untestable_no_replacement',reason=str(error))
        selection=dict(payload=payload,sha256=adapter.fingerprint(payload))
    design.write(output/'selection.json',selection)
    frozen=design.read(output/'selection.json')
    test=load(data/'safety_test.npz')
    test_predictions,repeated_parameters=core.fit_observed(fit['features'],fit['treatment'],fit['outcome'],test['features'])
    assert parameters==repeated_parameters
    try:test_result=adapter.test_selected(functions,frozen,core.frame(test,test_predictions),805730)
    except np.linalg.LinAlgError as error:
        test_result=dict(selection_sha256=frozen['sha256'],new_risk_certificate=False,methods=[
            dict(method=name,selected=values,status='untestable',passed=[],deployed_cutoff=adapter.BASELINE,
                reason='Selected family linear algebra failure; no replacement: '+str(error))
            for name,values in frozen['payload']['selected'].items()])
    design.write(output/'safety_test.json',test_result)
    deployment={m['method']:m['deployed_cutoff'] for m in test_result['methods']}
    design.write(output/'deployment.json',deployment)
    # All decisions are frozen before loading ANY paired evaluation outcomes.
    eval_features=load(data/'evaluation_features.npz')['features']
    eval_predictions,repeated_parameters=core.fit_observed(fit['features'],fit['treatment'],fit['outcome'],eval_features)
    assert parameters==repeated_parameters
    scores=eval_predictions[1]-eval_predictions[0]
    if not np.isfinite(scores).all() or np.any(scores>=adapter.BASELINE):raise ValueError('Evaluation score domain exceeded')
    choices=dict(always_baseline=np.zeros(128,int),always_projected_nominal=np.ones(128,int),projected_bank=np.full(128,2,int),
        **{name:(scores>=cutoff).astype(int) for name,cutoff in deployment.items()})
    design.write(output/'evaluation_choices.json',{name:value.tolist() for name,value in choices.items()})
    vault=load(data/'paired_audit_potential.npz')
    results={name:core.evaluate(vault,value) for name,value in choices.items()}
    design.write(output/'evaluation.json',results)
    assert design.inputs()==plan
    for name,expected in complete['files_sha256'].items():assert design.digest(data/name)==expected
    assert complete==design.read(data/'complete.json')
    design.write(output/'complete.json',dict(plan_sha256=design.digest(design.PLAN),export_sha256=design.digest(data/'complete.json'),
        evaluation=results,original_author_tests=test_result,new_sumo_runs=0,additional_independent_clusters=0,
        evaluated_independent_parent_clusters=128,finite_sample_risk_certificate=False,
        pure_online_gate_comparison=False,equal_compute_or_data_budget_claimed=False,
        files_sha256={p.name:design.digest(p) for p in output.iterdir() if p.is_file()}))
    print(json.dumps(dict(evaluation=results,original_tests=test_result)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True,type=Path);args=parser.parse_args()
    with threadpool_limits(limits=1):run(args.source)
