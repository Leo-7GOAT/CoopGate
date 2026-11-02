import ast
from copy import deepcopy
from collections import Counter
from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.stats import beta,t
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import mappo_risk_attempt3 as stats
import mappo_joint_data as data
import mappo_formal_design as design
from coopgate import recovery_risk_v39 as old


def test_statistical_function_bodies_and_original_thresholds_are_preserved():
    def functions(module):return {n.name:ast.dump(ast.Module(body=n.body,type_ignores=[])) for n in ast.parse(Path(module.__file__).read_text()).body if isinstance(n,ast.FunctionDef)}
    assert functions(stats)==functions(old)
    assert stats.MODELS==(4703,4704,4705) and stats.ATTEMPT==3
    assert stats.ATTEMPT_ALPHA==.05/12 and stats.RISK_LIMIT==.05 and stats.ADOPTION_MIN==.2 and stats.GAIN_MIN_PERCENT==.5
    with pytest.raises(ValueError):stats.adoption_certificate(np.ones((1024,3),bool),model_ids=old.MODELS)


def test_full_sampling_and_controller_coin_stream_are_fixed_without_balancing_confirmation():
    cal=design.stage_cases('calibration');confirm=design.stage_cases('confirmation',dict.fromkeys(('4703','4704','4705'),.4))
    assert len(cal)==3072 and len(confirm)==1024
    assert Counter(c['plant'] for c in cal)==dict.fromkeys(design.PLANTS,768)
    assert Counter(c['plant'] for c in confirm)!=dict.fromkeys(design.PLANTS,256)
    assert len({c['seed'] for c in cal+confirm})==4096
    assert sum(len(c['methods']) for c in cal+confirm)==37888
    assert sum(sum(m=='baseline' or m in data.METHODS[1:] for m in c['methods']) for c in cal+confirm)==16384
    assert all('controller_coins' not in c for c in cal)
    assert confirm==design.stage_cases('confirmation',dict.fromkeys(('4703','4704','4705'),.4))
    design.validate_candidate()


def row(value=11.,adopted=True,failed=False):
    return dict(per_vehicle_cost=[value]*3,per_vehicle_finish=[9.]*3,team_cost=value,learned_action_steps=int(adopted),
        completed=not failed,aborted=failed,collision=False,contract_violations=0)


def test_failures_before_adoption_and_baseline_failure_are_never_dropped():
    records={'baseline':row(adopted=False),**{m:row(10.8) for m in data.METHODS[1:]}}
    records['4703']=row(25.,False,True);actual=data.paired_cluster(records)
    assert actual['failed'][0] and actual['harmful'][0] and not actual['adopted'][0]
    records['baseline']=row(25.,False,True);actual=data.paired_cluster(records)
    assert actual['baseline_failed'] and all(actual['harmful']) and all(actual['baseline_unassessable'])


def test_one_vehicle_harm_and_zero_adoption_are_not_hidden_by_team_gain():
    records={'baseline':row(adopted=False),**{m:row(10.8) for m in data.METHODS[1:]}}
    records['4703']['per_vehicle_cost']=[11.1,10.,10.];records['4703']['team_cost']=np.mean([11.1,10.,10.])
    actual=data.paired_cluster(records);assert actual['harmful'][0] and actual['gain_percent'][0]>0
    records['4703']=deepcopy(records['baseline']);assert not data.paired_cluster(records)['adopted'][0]
    records['4703']['per_vehicle_cost'][0]+=.1;records['4703']['team_cost']=np.mean(records['4703']['per_vehicle_cost'])
    with pytest.raises(ValueError,match='Zero adoption'):data.paired_cluster(records)


def test_actual_attempt3_inference_matches_independent_quantiles_and_keeps_failure_block():
    a=np.ones((3072,3),bool);h=np.zeros_like(a);f=np.zeros_like(a);plants=np.array(design.PLANTS*768)
    ca=np.ones((1024,3),bool);cf=np.zeros_like(ca);g=np.random.default_rng(81011).normal(1.5,1.,(1024,3))
    result=stats.joint_deployment_certificate(a,h,f,plants,ca,g,cf,
        calibration_baseline_failed=np.zeros(3072,bool),confirmation_baseline_failed=np.zeros(1024,bool))
    assert result['fixed_models']==[4703,4704,4705] and result['joint_deployment_certified']
    np.testing.assert_allclose(result['adoption']['members']['4703']['adoption_lower'],beta.ppf(.05/12,1024,1))
    np.testing.assert_allclose(result['paired_gain']['members']['4703']['gain_lower_percent'],
        g[:,0].mean()-t.ppf(1-.05/12,1023)*g[:,0].std(ddof=1)/32)
    cf[0,0]=True
    result=stats.joint_deployment_certificate(a,h,f,plants,ca,g,cf,
        calibration_baseline_failed=np.zeros(3072,bool),confirmation_baseline_failed=np.zeros(1024,bool))
    assert not result['joint_deployment_certified'] and result['observed_execution_failures']==1
