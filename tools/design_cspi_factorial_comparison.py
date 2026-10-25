"""Outcome-free supplement to the unchanged, unstarted complete factorial."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'artifacts/candidate_factorial/complete_fair_cases'
PARENT_SHA='437c5386f0c6962e214c3f21d2b997ca40e6f6b6ba336068b7d519823e1830b1'
OUT=ROOT/'artifacts/literature_reproduction/cspi_factorial_prospective'
PLAN=OUT/'plan.json'
BLOB='84323a8cf90351cf69fbbc59300523ddca54f625'
UPSTREAM_SHA='e05c904b87b58163ecc78e3907cb16aca49eb62f7ebd8fe2f69fc03abf2b2fa5'
SOURCES=('tools/design_cspi_factorial_comparison.py','tools/export_cspi_factorial_comparison.py',
    'tools/run_cspi_factorial_comparison.py','tools/cspi_factorial_analysis.py',
    'tests/test_cspi_factorial_comparison.py','docs/cspi_factorial_prospective_protocol.md',
    'tools/cspi_supported_candidates.py','tools/run_cspi_author_pipeline.py')
PLANTS=('nominal','gain_drag','lag_mild','lag_strong')
MODELS=('1802','1812','1822')
VARIANTS=('baseline','projected_nominal','projected_bank')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as handle:
        json.dump(value,handle,indent=2,sort_keys=True,allow_nan=False);handle.write('\n')


def committed(name):
    relative=Path(name).resolve().relative_to(ROOT).as_posix()
    value=subprocess.check_output(['git','show','HEAD:'+relative],cwd=ROOT)
    if value!=(ROOT/relative).read_bytes():raise ValueError('Commit unchanged source first: '+relative)


def assignments(cases):
    if len(cases)!=640 or len({c['cluster_id'] for c in cases})!=640:
        raise ValueError('All 640 independent parent cases required')
    order=np.random.default_rng(805700).permutation(640)
    treatment=np.random.default_rng(805701).binomial(1,.5,640)
    split=['fit']*256+['tune']*128+['safety_test']*128+['evaluation']*128
    return [dict(position=i,block_index=int(index),cluster_id=cases[index]['cluster_id'],
        treatment=int(treatment[i]),split=split[i]) for i,index in enumerate(order)]


def contract():
    return dict(independent_parent_clusters=640,additional_independent_clusters=0,
        split_sizes=dict(fit=256,tune=128,safety_test=128,evaluation=128),
        candidate_controller='projected_nominal',reference_controller='baseline',
        proposed_controller='projected_bank',episode_level_selection=True,
        feature_fields=['initial_position_velocity_6','public_plant_onehot_4','policy_identity_onehot_3'],
        excluded_features=['true_actuator_parameters','future_disturbance','disturbance_mode','outcome_labels'],
        observed_reward='-(team_cost + 25 * failure)',propensity=.5,
        nuisance='Fit-only StandardScaler and per-arm Ridge(alpha=1)',
        candidate_quantiles=[.1,.3,.5,.7,.9],gamma=.05,gaussian_draws=10000,
        selector_seed=805720,test_seed=805730,
        methods=['always_baseline','always_projected_nominal','projected_bank','cspi','cspi_mt'],
        predeclared_support_adapter=True,untestable_policy='Whole selected family falls back; no resampling',
        no_new_sumo_execution=True,no_equal_compute_or_data_budget_claim=True,
        author_tests_asymptotic=True,new_finite_sample_risk_certificate=False,
        evaluation_descriptive_only=True,parent_controller_or_analysis_changed=False)


def verify_upstream(source):
    raw=Path(source).read_bytes()
    if (digest(source)!=UPSTREAM_SHA
            or hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!=BLOB):
        raise ValueError('Pinned original CSPI source required')


def freeze(source):
    if OUT.exists():raise FileExistsError('Preserve first supplement plan')
    if digest(PARENT/'plan.json')!=PARENT_SHA:raise ValueError('Parent plan changed')
    if any((PARENT/name).exists() for name in ('execution','blocks','analysis')):
        raise ValueError('Freeze before any formal factorial outcome')
    parent=read(PARENT/'plan.json');committed(PARENT/'plan.json');verify_upstream(source)
    for name,expected in parent['source_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Frozen parent source changed: '+name)
    for name in SOURCES:committed(ROOT/name)
    design=dict(contract=contract(),parent_plan_sha256=PARENT_SHA,
        assignments=assignments(parent['cases']),upstream_git_blob=BLOB,upstream_sha256=UPSTREAM_SHA,
        source_sha256={name:digest(ROOT/name) for name in SOURCES},
        frozen_before_parent_formal_outcomes=True,partial_efficacy_outcomes_read=False)
    OUT.mkdir(parents=True,exist_ok=False);write(PLAN,design)
    print(json.dumps(dict(plan_sha256=digest(PLAN),formal_outcomes_read=False,launched=False)))


def inputs():
    committed(PLAN);plan=read(PLAN)
    if (digest(PARENT/'plan.json')!=PARENT_SHA or plan['parent_plan_sha256']!=PARENT_SHA
            or plan['contract']!=contract() or plan['assignments']!=assignments(read(PARENT/'plan.json')['cases'])):
        raise ValueError('Fixed supplement design changed')
    for name,expected in plan['source_sha256'].items():
        committed(ROOT/name)
        if digest(ROOT/name)!=expected:raise ValueError('Supplement source changed: '+name)
    return plan


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('freeze','check'))
    parser.add_argument('--source',type=Path);args=parser.parse_args()
    if args.command=='freeze':freeze(args.source)
    else:inputs();print('unchanged')
