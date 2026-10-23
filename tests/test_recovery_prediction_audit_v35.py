import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from audit_recovery_forecast_v35 import compare_prediction
from coopgate.recovery_forecast_v35 import forecast
from test_recovery_forecast_v35 import independent_continuation


@pytest.fixture(scope='module')
def example():
    args,expected = independent_continuation('lag_strong','uniform',None,0)
    trace = expected['trace']
    for step,item in enumerate(trace):
        item['innovation'] = args['tape'][step*5:step*5+5].tolist()
    errors = np.asarray([t['disturbance'] for t in trace])
    row = dict(initial=args['state'].tolist(),plant=args['plant'],plant_name='lag_strong',
        completed=expected['complete'],per_vehicle_cost=expected['values'][:,0].tolist(),
        per_vehicle_finish=expected['values'][:,1].tolist(),contract_violations=0,
        error_contract_exceeded_substeps=int((np.max(np.abs(errors),axis=2) > .5+1e-9).sum()),
        max_actual_error=float(np.max(np.abs(errors))))
    return forecast(**args),row,trace


def test_full_prediction_review_matches_independent_fixture(example):
    predicted,row,trace = example
    assert compare_prediction(predicted,row,trace)['all_match']


@pytest.mark.parametrize('change',('prefix','command','memory','observer','final_observer','crossing','state','clock','error_count'))
def test_full_prediction_review_rejects_altered_or_incomplete_output(example,change):
    predicted,row,trace = copy.deepcopy(example)
    if change == 'prefix': predicted['trace'].pop()
    if change == 'command': predicted['trace'][0]['command'][0] += .1
    if change == 'memory': predicted['trace'][0]['actuator_states'][0][0] += .1
    if change == 'observer': predicted['trace'][0]['actuator_memory_upper'][0] += .1
    if change == 'final_observer': predicted['final_observer_lower'][0] += .1
    if change == 'crossing': predicted['crossing'][0] += .1
    if change == 'state': predicted['final_state'][0][0] += .1
    if change == 'clock': predicted['trace'][0]['t'] += .1
    if change == 'error_count': predicted['legacy_error_exceeded_substeps'] += 1
    result = compare_prediction(predicted,row,trace)
    assert not result['all_match']
