from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import audit_common_adoption_v27 as audit


def fixture():
    row=dict(initial=[[0.,0.]]*3,controller='raw_1802',learned_action_steps=1,control_steps=1)
    trace=[dict(step=0,command=[0.,0.,0.],x=[[1.,0.]]*3,baseline_command=[0.,0.,0.],
                changed_active_vehicles=[False]*3,adopted=False)]
    record=dict(source_row_sha256=audit.object_hash(row),original_learning_steps=1,
                common_learning_steps=0,common_episode_adopted=False)
    return row,trace,record


def test_original_adoption_can_differ_without_overwriting_it():
    row,trace,record=fixture()
    audit.check_compact(row,trace,record)
    assert row['learned_action_steps']==1 and not record['common_episode_adopted']


@pytest.mark.parametrize('field,value',[('common_learning_steps',1),('common_episode_adopted',True),('original_learning_steps',0)])
def test_changing_either_event_is_detected(field,value):
    row,trace,record=fixture()
    record[field]=value
    with pytest.raises(ValueError):
        audit.check_compact(row,trace,record)


def test_missing_control_step_or_modified_input_is_detected():
    row,trace,record=fixture()
    with pytest.raises(ValueError,match='Incomplete'):
        audit.check_compact(row,[],record)
    row['initial'][0][0]=99.
    with pytest.raises(ValueError,match='Original row'):
        audit.check_compact(row,trace,record)


def test_claimed_base_controller_must_match_actual_reconstruction():
    row,trace,record=fixture()
    row.update(controller='baseline',learned_action_steps=0)
    record.update(source_row_sha256=audit.object_hash(row),original_learning_steps=0,
                  common_learning_steps=1,common_episode_adopted=True)
    trace[0].update(command=[1.,0.,0.],changed_active_vehicles=[True,False,False],adopted=True)
    with pytest.raises(ValueError,match='baseline differs'):
        audit.check_compact(row,trace,record)


def test_grid_refuses_deleted_models_and_duplicate_cases():
    scope=audit.read(audit.SCOPE)
    rows=audit.read(audit.OLD/'phase/episodes.json')
    spec=scope['protocols']['phase']['spec']
    audit.validate_cases('phase',rows,spec)
    with pytest.raises(ValueError,match='All original'):
        audit.validate_cases('phase',rows[:-1],spec)
    with pytest.raises(ValueError,match='All original'):
        audit.validate_cases('phase',rows+[deepcopy(rows[0])],spec)


def test_compact_trace_keeps_original_states_and_executed_commands():
    _,trace,_=fixture()
    original=[dict(command=trace[0]['command'],x=trace[0]['x'],unneeded_prediction='full archive retained')]
    result=audit.compact_trace(original,[{k:v for k,v in trace[0].items() if k not in ('command','x')}])
    assert result==trace
    with pytest.raises(ValueError):
        audit.compact_trace(original,[])
