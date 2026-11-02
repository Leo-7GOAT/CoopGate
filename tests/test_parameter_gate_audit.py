import copy
import numpy as np
import pytest
from coopgate.parameter_gate_audit import validate_scenario_evidence
from coopgate.parameter_scenario_gate import parameter_bank


def test_parameter_score_reviewer_checks_count_bounds_budget_and_every_path():
    item=dict(actuator_class='lag_strong',forecast_bank_size=41,original_midpoint_paths=33,
        parameter_scenario_count=8,full_parameter_box_certified=False,
        parameter_scenarios=[dict(gain=g.tolist(),drag=d.tolist(),retention=r.tolist()) for g,d,r in parameter_bank('lag_strong')],
        scenario_max_failed_paths=0,scenario_score_is_probability=False,confidence_threshold=1.,
        scenario_pass_fraction=1.,selected_bank_differences=np.full((41,3,2),-.1).tolist(),
        selected_phase_differences=np.zeros((41,3)).tolist())
    validate_scenario_evidence(item,True)
    for kind in ('count','parameter','budget','missing','regression','box'):
        bad=copy.deepcopy(item)
        if kind=='count':bad['forecast_bank_size']=33
        if kind=='parameter':bad['parameter_scenarios'][0]['retention'][0]=1.
        if kind=='budget':bad['scenario_max_failed_paths']=1
        if kind=='missing':bad['selected_bank_differences'].pop()
        if kind=='regression':bad['selected_bank_differences'][-1][2][0]=.1
        if kind=='box':bad['full_parameter_box_certified']=True
        with pytest.raises(ValueError):validate_scenario_evidence(bad,True)
