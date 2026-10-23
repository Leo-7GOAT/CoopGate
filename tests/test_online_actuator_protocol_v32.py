from copy import deepcopy
import sys
from pathlib import Path

import pytest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from audit_online_actuator_v32 import compare_kernel,integration_scope
from coopgate.actuator_prediction_roundoff_v31 import predict_suffix
from coopgate.online_actuator_gate_v32 import forecast,reference_parameters
from coopgate.shifted_sumo_v19 import PLANTS,SCENARIOS


def cases():
    return [[32*si+i,scene,i,32000000+1000*si+i,plant,'phase05_'+model]
        for si,scene in enumerate(SCENARIOS) for i in range(32) for plant in PLANTS for model in ('1802','1812','1822')]


def test_integration_includes_all_cells_and_both_models_without_outcome_filtering():
    scope = integration_scope(cases())
    assert len(scope) == 120 and len({tuple(k) for k in scope}) == 120
    assert {k[4] for k in scope} == set(PLANTS)
    assert {k[1] for k in scope} == set(SCENARIOS)
    assert {k[-1] for k in scope} == {'nominal','reference'}
    assert all(k[2] == 0 for k in scope)


@pytest.mark.parametrize('change',['missing','duplicate','wrong_plant','wrong_model'])
def test_fixed_cell_coverage_cannot_drop_strong_lag_or_a_model(change):
    values = cases()
    if change == 'missing': values.pop(0)
    elif change == 'duplicate': values[1] = deepcopy(values[0])
    elif change == 'wrong_plant': values[0][4] = 'lag_strong'
    else: values[0][5] = 'phase05_9999'
    with pytest.raises(ValueError,match='all 60'):
        integration_scope(values)


def test_kernel_comparison_cannot_hide_a_different_failure_reason_or_clock():
    state = np.array([[0.,10.],[0.,10.],[-1.,10.]])
    args = (state,np.zeros(3),np.zeros(3),3,np.zeros((250,3)),reference_parameters('lag_strong'))
    expected = predict_suffix(*args,record_trace=True)
    actual = forecast(*args,record_trace=True)
    assert compare_kernel(actual,expected,expected['trace'])['all_match']
    changed = deepcopy(actual)
    changed['error_code'] = 2
    assert not compare_kernel(changed,expected,expected['trace'])['all_match']
    changed = deepcopy(actual)
    changed['error_step'] += 1
    assert not compare_kernel(changed,expected,expected['trace'])['all_match']
