from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from pair_dmps_recovery_baseline import paired_outcome


def base():
    return dict(completed=True,aborted=False,collision=False,contract_violations=0,
        per_vehicle_cost=[10.,10.,10.],per_vehicle_finish=[10.,10.,10.],team_cost=10.)


def test_incomplete_learning_cannot_be_reported_as_cost_gain_even_with_small_partial_cost():
    result=paired_outcome(base(),dict(completed=False,per_vehicle_cost=[1.,1.,1.]),[1.,1.,1.])
    assert result['learned_failed'] and result['relative_harm'] and result['paired_cost_gain_percent'] is None


def test_failed_baseline_is_unassessable_rather_than_harmless_learning():
    row=base();row['aborted']=True
    result=paired_outcome(row,dict(completed=True,per_vehicle_cost=[9.,9.,9.]),[9.,9.,9.])
    assert result['baseline_failed'] and result['relative_harm'] is None
    assert result['paired_cost_gain_percent'] is None


def test_team_improvement_can_still_harm_an_individual_vehicle():
    result=paired_outcome(base(),dict(completed=True,per_vehicle_cost=[7.,8.,11.]),[7.,8.,10.])
    assert result['paired_cost_gain_percent']>0 and result['relative_harm']
