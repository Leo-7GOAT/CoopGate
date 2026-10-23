from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import audit_actuator_prediction_v30 as audit


def test_scope_keeps_all_four_plants_all_scenes_and_every_known_baseline_abort():
    rows = audit.read(audit.OLD/'episodes.json')
    selected,replay = audit.scope(rows)
    assert len(selected) == 640
    assert sum(r['aborted'] for r in selected) == 10
    assert {audit.case(r) for r in selected if r['aborted']} <= {audit.case(r) for r in replay}
    assert {(r['scenario'],r['plant_name']) for r in replay} == {(s,p) for s in audit.SCENARIOS for p in audit.PLANTS}
    assert sum(r['initial_index']==0 for r in replay) == 20
    with pytest.raises(ValueError,match='All 640'):
        audit.scope([r for r in rows if audit.case(r) != audit.case(selected[-1])])
    with pytest.raises(ValueError,match='All 640'):
        audit.scope(rows+[deepcopy(selected[0])])


def test_comparison_does_not_hide_wrong_abort_cost_command_or_actuator_memory():
    step = dict(x=[[0.,1.]]*3,command=[0.]*3,actuator_states=[[0.]*3]*5,disturbance=[[0.]*3]*5)
    prediction = dict(trace=[step],values=[[1.,1.]]*3,error=None,complete=True,route_violations=0,
                      error_contract_exceeded_substeps=0,max_actual_error=0.)
    row = dict(per_vehicle_cost=[1.]*3,per_vehicle_finish=[1.]*3,aborted=False,abort_reason='',
               completed=True,contract_violations=0,error_contract_exceeded_substeps=0,max_actual_error=0.)
    assert audit.comparison(prediction,row,[step])['all_match']
    for kind in ('cost','command','actuator','abort'):
        changed = deepcopy(prediction)
        if kind == 'cost': changed['values'][0][0] += .5
        elif kind == 'command': changed['trace'][0]['command'][0] += .1
        elif kind == 'actuator': changed['trace'][0]['actuator_states'][0][0] += .1
        else: changed['error'] = dict(message='Unexpected abort')
        assert not audit.comparison(changed,row,[step])['all_match']
