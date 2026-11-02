import copy
import numpy as np
import pytest
from coopgate.scenario_gate_audit import validate_scenario_evidence


def test_independent_score_checker_rejects_tampered_budget_score_and_missing_path():
    d=np.full((33,3,2),-.1);d[2]=.1
    item=dict(scenario_max_failed_paths=1,scenario_score_is_probability=False,
              confidence_threshold=32/33,scenario_pass_fraction=32/33,
              selected_bank_differences=d.tolist(),selected_phase_differences=np.zeros((33,3)).tolist())
    validate_scenario_evidence(item,True)
    for key,value in [('scenario_pass_fraction',1.),('scenario_max_failed_paths',0),
                      ('selected_bank_differences',d[:-1].tolist()),('scenario_score_is_probability',True)]:
        bad=copy.deepcopy(item);bad[key]=value
        with pytest.raises(ValueError):validate_scenario_evidence(bad,True)
