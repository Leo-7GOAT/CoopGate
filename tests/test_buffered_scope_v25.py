"""Protect full-grid coverage, error timing and strict replay semantics."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import audit_buffered_scope_v25 as audit


class FakeGate:
    step = 7
    def __init__(self, error=None): self.error=error
    def select(self, x):
        if self.error: raise self.error
        return x, {'certified':True,'gate_ms':1.2}


def test_wrapper_preserves_values_and_records_success():
    gate=audit.TimedGate(FakeGate())
    assert gate.select(3)==(3,{'certified':True,'gate_ms':1.2})
    assert gate.step==7 and gate.calls[0]['error'] is None
    audit.check_calls({'aborted':False},[{}],gate.calls)


@pytest.mark.parametrize('kind',[ValueError,RuntimeError])
def test_terminal_call_is_counted_and_original_error_propagates(kind):
    gate=audit.TimedGate(FakeGate(kind('physical backup unavailable')))
    with pytest.raises(kind,match='physical backup'): gate.select(3)
    audit.check_calls({'aborted':True,'abort_reason':'physical backup unavailable'},[],gate.calls)
    assert gate.calls[0]['error']['type']==kind.__name__ and gate.calls[0]['wall_ms']>=0


@pytest.mark.parametrize('calls',[
    [], [{'wall_ms':float('nan'),'error':None}],
    [{'wall_ms':-1,'error':None}],
    [{'wall_ms':1,'error':{'type':'ValueError','message':'missing'}}]])
def test_omitted_or_invalid_calls_rejected(calls):
    with pytest.raises(ValueError): audit.check_calls({'aborted':False},[{}],calls)


def fixture_scope(study):
    if study=='phase':
        spec=dict(shards=1,n_initials=2,seed_base=10,modes=['clean','bias'],
                  methods={f'phase05_{s}':{} for s in audit.MODELS})
        rows=[dict(initial_index=i,disturbance=m,controller=c) for i,m,c in audit.expected_grid(study,spec)]
    else:
        spec=dict(shards=1,n_initials_per_scenario=2,seed_base=10,scenarios=['nominal','slow_yield'],
                  plants=['nominal','lag_strong'],methods={f'phase05_{s}':{} for s in audit.MODELS})
        rows=[dict(global_initial_index=g,scenario=sc,initial_index=i,seed=s,plant_name=p,controller=c)
              for g,sc,i,s,p,c in audit.expected_grid(study,spec)]
    return spec,rows


@pytest.mark.parametrize('study',['phase','transfer'])
def test_full_scope_accepts_all_cases(study):
    spec,rows=fixture_scope(study)
    audit.validate_scope(study,list(reversed(rows)),spec)


@pytest.mark.parametrize('study',['phase','transfer'])
@pytest.mark.parametrize('mutation',['remove','duplicate','replace_model'])
def test_full_scope_rejects_cherry_picked_or_substituted_cases(study,mutation):
    spec,rows=fixture_scope(study)
    if mutation=='remove': rows.pop()
    elif mutation=='duplicate': rows.append(copy.deepcopy(rows[0]))
    else: rows[0]['controller']='phase05_9999'
    with pytest.raises(ValueError): audit.validate_scope(study,rows,spec)


def test_every_case_gets_both_orders_and_both_implementations():
    group=dict(cases=[dict(case=[0,'clean','phase05_1802']),dict(case=[0,'clean','phase05_1812'])])
    assert audit.schedule(group)==[(0,'clean','phase05_1802','reference'),(0,'clean','phase05_1802','buffered'),
                                  (0,'clean','phase05_1812','buffered'),(0,'clean','phase05_1812','reference')]


def test_group_order_uses_indices_not_outcomes():
    records=[dict(study='transfer',case=[32,'slow_yield',0,1010,'lag_strong','phase05_1802'],
                  local_index=0,scenario='slow_yield'),
             dict(study='phase',case=[1,'clean','phase05_1802'],local_index=1,scenario='nominal'),
             dict(study='phase',case=[0,'bias','phase05_1802'],local_index=0,scenario='dense')]
    result=audit.ordered_groups(records)
    assert [(g['study'],g['initial_index']) for g in result]==[('phase',0),('transfer',32),('phase',1)]
    assert [g['block_index'] for g in result]==[0,1,2]


@pytest.mark.parametrize('changed_field',['command','portfolio_joint_action_admitted','performance_model_transitions'])
def test_existing_replay_comparison_rejects_non_timing_changes(changed_field):
    row={'aborted':False,'controller_ms':[1],'setup_ms':1}
    trace=[dict(command=[1,2,3],portfolio_joint_action_admitted=True,performance_model_transitions=9,gate_ms=1)]
    other=copy.deepcopy(trace);other[0][changed_field]=0
    with pytest.raises(AssertionError): audit.original.compare_replay(row,row,trace,other)


def test_existing_replay_comparison_accepts_only_timing_changes():
    a={'aborted':False,'controller_ms':[1],'setup_ms':1};b={**a,'controller_ms':[2],'setup_ms':2}
    audit.original.compare_replay(a,b,[{'command':[1,2,3],'gate_ms':1}],[{'command':[1,2,3],'gate_ms':2}])


@pytest.mark.parametrize('target,field,value',[
    ('receipt','exit_code',1),('receipt','launcher_pid',999),('receipt','native_pid',999),
    ('receipt','plan_sha256','other'),('launch','plan_sha256','other'),
    ('done','plan_sha256','other'),('done','native_pid',999),
    ('done','actual_sumo_runs',1),('done','blocks',1)])
def test_successful_complete_native_exit_cannot_be_substituted(target,field,value):
    plan=dict(actual_sumo_runs=6912,blocks=288)
    launch=dict(launcher_pid=10,native_pid=11,plan_sha256='expected')
    receipt=dict(**launch,exit_code=0)
    done=dict(**plan,native_pid=11,plan_sha256='expected')
    audit.validate_completion(plan,'expected',receipt,launch,done)
    {'receipt':receipt,'launch':launch,'done':done}[target][field]=value
    with pytest.raises(ValueError):audit.validate_completion(plan,'expected',receipt,launch,done)
