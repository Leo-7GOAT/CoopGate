from copy import deepcopy

import numpy as np
import pytest

from coopgate.performance_identity_v31 import audit_identity


def outcome(cost,finish,admissions=0):
    return dict(per_vehicle_cost=cost,per_vehicle_finish=finish,learned_action_steps=admissions,
        control_steps=30,completed=True,aborted=False,collision=False,contract_violations=0)


def admission(step,cost,time):
    return dict(step=step,pair=dict(both_complete=True,difference=np.column_stack((cost,time)).tolist()))


def test_local_losses_can_cancel_without_being_removed_from_the_identity():
    base = outcome([10.]*3,[10.]*3)
    source = outcome([9.5,10.,9.],[9.5,10.,9.],2)
    local = [admission(0,[.25,0.,-.5],[.5,0.,-.5]),admission(5,[-.75,0.,-.5],[-1.,0.,-.5])]
    result = audit_identity(source,base,local)
    assert result['identity_matches'] and result['episode_nonworsening']
    assert not result['all_local_nonworsening']
    assert result['positive_local_contributions'][0] == [.25,.5]


def test_zero_adoption_is_an_exact_baseline_case_not_a_missing_sample():
    base = outcome([10.]*3,[10.]*3)
    result = audit_identity(deepcopy(base),base,[])
    assert result['assessable'] and result['identity_matches'] and result['admissions'] == 0
    changed = deepcopy(base)
    changed['per_vehicle_cost'][0] += .01
    assert not audit_identity(changed,base,[])['identity_matches']


@pytest.mark.parametrize('change',['missing','duplicate','wrong_clock'])
def test_no_omitted_or_duplicated_learning_action_can_satisfy_the_identity(change):
    base = outcome([10.]*3,[10.]*3)
    source = outcome([9.]*3,[9.]*3,2)
    records = [admission(1,[-.5]*3,[-.5]*3),admission(3,[-.5]*3,[-.5]*3)]
    if change == 'missing': records.pop()
    elif change == 'duplicate': records[1]['step'] = 1
    else: records[1]['step'] = 30
    with pytest.raises(ValueError,match='exactly once'):
        audit_identity(source,base,records)


@pytest.mark.parametrize('change',['source_abort','baseline_abort','branch_failure'])
def test_incomplete_low_costs_are_retained_but_cannot_verify_identity(change):
    base = outcome([10.]*3,[10.]*3)
    source = outcome([0.]*3,[0.]*3,1)
    records = [admission(0,[-10.]*3,[-10.]*3)]
    if change == 'source_abort': source['aborted'] = True
    elif change == 'baseline_abort': base['completed'] = False
    else: records[0]['pair'] = dict(both_complete=False,difference=None)
    result = audit_identity(source,base,records)
    assert not result['assessable'] and result['identity_matches'] is None
    assert result['observed_capped_difference'] == [[-10.,-10.]]*3


def test_team_gain_does_not_hide_one_vehicle_regression():
    base = outcome([10.]*3,[10.]*3)
    source = outcome([9.,9.,10.2],[9.,9.,10.],1)
    result = audit_identity(source,base,[admission(0,[-1.,-1.,.2],[-1.,-1.,0.])])
    assert result['identity_matches'] and result['episode_mean_cost_difference'] < 0
    assert not result['episode_nonworsening']
