from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from extend_online_actuator_v32 import full_groups,relative_outcome,summarize_cell
from coopgate.shifted_sumo_v19 import PLANTS,SCENARIOS


def spec():
    return dict(n_initials_per_scenario=32,scenarios=list(SCENARIOS),plants=list(PLANTS),seed_base=63400001)


def original():
    return [(32*si+i,scene,i,63400001+1000*si+i,plant,'phase05_'+model)
        for si,scene in enumerate(SCENARIOS) for i in range(32) for plant in PLANTS
        for model in ('1802','1812','1822')]


def test_full_grid_retains_every_original_condition_and_balances_execution_order():
    groups = full_groups(original(),spec())
    assert len(groups)==640
    cases = [tuple(k) for g in groups for k in g['cases']]
    assert len(cases)==len(set(cases))==3840
    assert {k[:-1] for k in cases}==set(original())
    assert sum(g['cases'][i][-1]=='nominal' for g in groups for i in (0,2,4))==960
    assert sum(g['cases'][i][-1]=='reference' for g in groups for i in (0,2,4))==960
    assert {g['condition'][2] for g in groups}==set(range(32))


@pytest.mark.parametrize('change',['missing','duplicate','wrong_seed','wrong_model','wrong_initial','missing_strong_lag'])
def test_full_grid_rejects_missing_or_substituted_conditions(change):
    values = [list(k) for k in original()]
    if change=='missing':values.pop()
    elif change=='duplicate':values[-1]=deepcopy(values[0])
    elif change=='wrong_seed':values[0][3]+=1
    elif change=='wrong_model':values[0][5]='phase05_9999'
    elif change=='wrong_initial':values[0][2]=32
    else:values=[k for k in values if k[4]!='lag_strong']
    with pytest.raises(ValueError,match='Complete original'):
        full_groups(values,spec())


def row(cost=10.,complete=True):
    return dict(completed=complete,collision=False,aborted=not complete,contract_violations=0,
        per_vehicle_cost=[cost]*3,per_vehicle_finish=[cost-.5]*3,team_cost=cost,
        error_contract_exceeded_substeps=int(not complete))


def test_incomplete_baseline_cannot_supply_false_relative_gain_or_harm_evidence():
    result = relative_outcome(row(3.),row(4.,False),True)
    assert result['baseline_failed'] and result['adopted']
    assert result['relative_harm'] is None and result['paired_cost_gain_percent'] is None


def test_failed_candidate_keeps_failure_even_if_truncated_cost_is_small():
    result = relative_outcome(row(2.,False),row(),True)
    assert result['controller_failed'] and result['relative_harm']
    assert result['paired_cost_gain_percent'] is None


def test_one_vehicle_loss_remains_harmful_when_team_mean_improves():
    current = row(9.)
    current['per_vehicle_cost']=[7.,9.,11.]
    result = relative_outcome(current,row(),True)
    assert result['paired_cost_gain_percent']==10. and result['relative_harm']


def test_all_scope_gain_is_withheld_on_failure_and_terminal_calls_are_included():
    cases = []
    for current,baseline in [(row(9.),row()),(row(3.,False),row()),(row(9.),row(4.,False))]:
        cases.append(dict(row=current,baseline_team_cost=baseline['team_cost'],calls=[dict(wall_ms=600.)],
            **relative_outcome(current,baseline,True)))
    summary = summarize_cell(cases)
    assert summary['all_scope_cost_gain_percent'] is None
    assert summary['complete_pairs_descriptive_gain_percent']==10.
    assert summary['paired_complete']==1 and summary['runs']==3
    assert summary['completed']==2 and summary['baseline_complete']==2
    assert summary['comparable_adopted']==2 and summary['harmful_and_adopted']==1
    assert summary['adopted_with_incomplete_baseline']==1
    assert summary['callback_count']==3 and summary['callback_over_500ms']==3
