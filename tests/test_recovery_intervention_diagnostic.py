import copy

import numpy as np
import pytest

from coopgate.recovery_intervention_diagnostic import ROLES,RecordedRecoveryBranch,adoption_state,boundary,pair
import diagnose_recovery_regressions as diagnostic


@pytest.fixture(scope='module')
def source_records():
    records,_,_=diagnostic.source_cases()
    return records


def test_scope_reads_every_harm_from_completed_sealed_blocks(source_records):
    assert len(source_records)==5 and sum(len(r['steps']) for r in source_records)==8
    assert len({tuple(r['case'][:4]) for r in source_records})==4


def test_every_original_prefix_and_boundary_base_replays_with_unchanged_v34(source_records):
    for record in source_records:
        row,trace=record['source'],record['trace']
        for target in record['steps']:
            for role in ROLES:
                controller=RecordedRecoveryBranch(row,trace,target,role)
                for step in range(target+1):
                    x=np.asarray(row['initial'] if step==0 else trace[step-1]['x'])
                    previous=np.zeros(3) if step==0 else np.asarray(trace[step-1]['command'])
                    command,info=controller.select(x,previous)
                    expected=trace[step]['baseline_command'] if step==target and role=='baseline_suffix' else trace[step]['command']
                    np.testing.assert_array_equal(command,expected)
                    assert info['diagnostic_intervention']==(step==target and role=='candidate_then_baseline')


def test_native_controller_does_not_read_hidden_plant_memory_or_future_innovation(source_records):
    class Forbidden:
        def __array__(self,*args,**kwargs):raise AssertionError('Hidden array read')
        def __getitem__(self,key):raise AssertionError('Hidden parameter read')
    record=source_records[0];row=copy.deepcopy(record['source']);row['plant']=Forbidden()
    trace=copy.deepcopy(record['trace']);target=record['steps'][0]
    for entry in trace:
        for name in ('actuator_states','innovation','disturbance'):entry[name]=Forbidden()
    controller=RecordedRecoveryBranch(row,trace,target,'candidate_then_baseline')
    for step in range(target+1):
        x=np.asarray(row['initial'] if step==0 else trace[step-1]['x'])
        prev=np.zeros(3) if step==0 else np.asarray(trace[step-1]['command'])
        controller.select(x,prev)


def test_nonadoption_cannot_be_promoted_to_an_intervention(source_records):
    record=source_records[0];trace=copy.deepcopy(record['trace']);step=record['steps'][0]
    trace[step]['actual_learning_used']=False
    with pytest.raises(ValueError):adoption_state(record['source'],trace,step)
    trace=copy.deepcopy(record['trace']);trace[step]['command']=trace[step]['baseline_command']
    with pytest.raises(ValueError):boundary(record['source'],trace,step)


def test_incomplete_suffix_cannot_authorize_a_local_performance_difference():
    result=pair(dict(baseline_suffix=dict(complete=False),candidate_then_baseline=dict(complete=True)))
    assert not result['both_complete'] and result['difference'] is None and result['phase_difference'] is None
