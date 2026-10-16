"""The formal prerequisite must not be replaced by a favorable average alone."""
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('report_single_v14',Path(__file__).resolve().parents[1]/'tools/report_single_v14.py')
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)


def fixture():
    seeds=[901,911,921];methods=['rollout']+[f'{kind}_{s}' for kind in ('learned','last') for s in seeds]
    p=dict(training_seeds=seeds,methods=methods,initials=[dict(seed=s) for s in range(4)],
           disturbances=['clean','uniform'],expected_runs=4*2*len(methods),
           bootstrap=dict(seed=14,draws=1000),criteria=dict(ego_cost_gain_percent_min=1.,ego_time_gain_percent_min=.5,ego_variation_gain_percent_min=10.))
    rows=[]
    for initial in p['initials']:
        for mode in p['disturbances']:
            for method in methods:
                baseline=method=='rollout';time=10. if baseline else 9.6;variation=8. if baseline else 6.
                rows.append(dict(seed=initial['seed'],disturbance=mode,controller=method,
                    ego_time=time,ego_variation=variation,ego_cost=time+.05*variation,other_time=11.,
                    completed=True,collision=False,aborted=False,contract_violations=0,outside_steps=0,
                    uncertified_steps=0,controller_ms=[.8,1.],learned_tail_fraction=0.,
                    group='low_conflict' if initial['seed']%2==0 else 'high_conflict'))
    return p,rows


def test_consistent_individual_gain_passes_all_frozen_prerequisite_checks():
    p,rows=fixture();s=report.summarize(p,rows)
    assert s['individual_ego_verified'] and s['marl_expansion_allowed']
    assert s['acceptance']['passed']==s['acceptance']['total']==9


def test_one_regressing_seed_cannot_hide_in_positive_average():
    p,rows=fixture()
    for row in rows:
        if row['controller']=='learned_921':
            row['ego_time']=10.1;row['ego_cost']=10.5
    result=report.summarize(p,rows)
    assert result['gain_percent']['learned']['ego_cost']>1.
    assert not result['acceptance']['checks']['every_seed_cost_improves']
    assert not result['marl_expansion_allowed']


def test_positive_point_estimate_without_positive_interval_is_insufficient():
    p,rows=fixture()
    for row in rows:
        if row['controller'].startswith('learned_'):
            row['ego_time']=7. if row['seed']%2==0 else 12.6
            row['ego_cost']=row['ego_time']+.05*row['ego_variation']
    result=report.summarize(p,rows)
    assert result['gain_percent']['learned']['ego_time']>.5
    assert not result['acceptance']['checks']['time_interval_positive']
    assert not result['marl_expansion_allowed']


def test_incomplete_or_repeated_executions_are_rejected():
    p,rows=fixture()
    with pytest.raises(AssertionError):report.summarize(p,rows[:-1])
    rows[-1]=dict(rows[0])
    with pytest.raises(AssertionError):report.summarize(p,rows)


def test_safety_failure_or_online_performance_search_prevents_unlocking():
    p,rows=fixture();rows[1]['contract_violations']=1
    assert not report.summarize(p,rows)['marl_expansion_allowed']
    p,rows=fixture();next(r for r in rows if r['controller'].startswith('learned_'))['learned_tail_fraction']=.1
    assert not report.summarize(p,rows)['marl_expansion_allowed']
