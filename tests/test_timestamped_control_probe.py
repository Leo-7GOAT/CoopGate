from copy import deepcopy
from datetime import datetime,timezone
from types import SimpleNamespace

import numpy as np
import pytest

import coopgate.timestamped_control_probe as probe
import execute_timestamped_control_probe as run
import watch_timestamped_control_probe as observer
import verify_timestamped_control_probe as verifier


def evidence(decision,source=0,profile='stale_dropout'):
    return dict(profile=profile,decision=decision,source_decision=source)


def test_sensor_bridge_exports_only_public_current_source_metadata():
    class Sensor:
        def observe(self,history):
            return np.ones((3,2)),dict(**evidence(0),true_parameters='forbidden',future_tape=[123],
                                      true_state=history,actuator_memory='forbidden')
    port=probe.TimestampPort();bridge=probe.PublicSensorBridge(Sensor(),port)
    bridge.observe('sensor-internal-truth')
    assert port.current(0)==dict(profile='stale_dropout',decision=0,source_decision=[0]*3)
    payload=port.current(0);payload['source_decision'][0]=99
    assert port.current(0)['source_decision']==[0]*3


def test_port_rejects_undelivered_reordered_and_future_sources():
    port=probe.TimestampPort()
    with pytest.raises(ValueError):port.current(0)
    with pytest.raises(ValueError):port.publish(evidence(1))
    with pytest.raises(ValueError):port.publish(evidence(0,1))
    port.publish(evidence(0))
    with pytest.raises(ValueError):port.publish(evidence(0))
    with pytest.raises(ValueError):port.current(1)


@pytest.mark.parametrize('variant',probe.VARIANTS)
def test_all_variants_receive_common_envelope_and_route_correct_current_point(monkeypatch,variant):
    class Inner:
        def __init__(self,*args):self.c=SimpleNamespace(decision_dt=.5);self.calls=[]
        def select(self,state,previous,unused):
            self.calls.append(state.copy());return np.zeros(3),dict(baseline_command=[0.]*3)
    monkeypatch.setattr(probe,'RecoveryJointGate',Inner)
    port=probe.TimestampPort();gate=probe.TimestampedGate(None,'lag_strong','baseline',variant,port)
    raw=np.array([[-100.,10.]]*3);before=deepcopy(raw)
    port.publish(evidence(0));gate.select(raw,np.zeros(3))
    port.publish(evidence(1));_,info=gate.select(raw,np.zeros(3))
    expected=(raw if variant=='projected_raw' else raw+np.array([5.,0.]) if variant=='constant_velocity'
        else (np.asarray(gate.receipts[-1]['envelope']['lower'])+np.asarray(gate.receipts[-1]['envelope']['upper']))/2)
    np.testing.assert_array_equal(gate.inner.calls[-1],expected)
    np.testing.assert_array_equal(raw,before)
    assert info['age_adapter']['envelope']['age_decisions']==[1]*3
    assert not info['age_adapter']['restores_physical_certificate']
    assert len(gate.receipts)==2


def test_failed_inner_call_keeps_complete_public_input_receipt(monkeypatch):
    class Inner:
        def __init__(self,*args):self.c=SimpleNamespace(decision_dt=.5)
        def select(self,*args):raise RuntimeError('preserved inner failure')
    monkeypatch.setattr(probe,'RecoveryJointGate',Inner)
    port=probe.TimestampPort();gate=probe.TimestampedGate(None,'lag_strong','baseline','envelope_midpoint',port)
    port.publish(evidence(0))
    with pytest.raises(RuntimeError,match='preserved'):gate.select([[-100.,10.]]*3,[0.]*3)
    assert gate.receipts[0]['inner_error']==dict(type='RuntimeError',message='preserved inner failure')
    assert gate.receipts[0]['envelope'] is not None


def test_private_adapters_keep_original_code_and_globals():
    original=run.original_audit.audit_episode
    marker=object();adapted=run.private(original,RecoveryJointGate=marker)
    assert adapted.__code__ is original.__code__ and original.__globals__['RecoveryJointGate'] is not marker
    assert observer.configured_waiter().__code__ is observer.waiter.watch.__code__


def test_observer_rejects_a_wrong_native_command():
    created=datetime(2026,10,25,tzinfo=timezone.utc)
    start=dict(native_pid=1,parent_pid=2,utc=created.isoformat())
    with pytest.raises(ValueError,match='collector'):
        observer.validate_identity(start,dict(ProcessId=1,ParentProcessId=2,CommandLine='wrong.py'),created)


def test_success_receipt_requires_actual_zero_and_complete_117_scope(monkeypatch):
    with pytest.raises(RuntimeError):observer.successful_receipt(None,None,7)
    monkeypatch.setattr(observer.io,'digest',lambda p:'sha')
    monkeypatch.setattr(observer.io,'read',lambda p:dict(actual_sumo_runs=116,blocks=[None]*9,protocol_sha256='sha'))
    with pytest.raises(AssertionError):observer.successful_receipt(run.OUT,dict(protocol_sha256='sha',native_start_sha256='sha'),0)


def test_non_timing_raw_control_comparison_still_retains_actual_physics():
    value=dict(age_adapter={'anything':1},velocity_projection={},observation_adapter_inputs=[],
        wall_ms=3.,episode_wall_seconds=2.,command=[1,2,3],x=[[1,2]]*3,failure=True)
    normalized=run.non_timing(value)
    assert normalized==dict(command=[1,2,3],x=[[1,2]]*3,failure=True)


@pytest.fixture(scope='module')
def recorded_native_raw_fixture():
    # Closed, previously audited native data; no new simulator or new variant's
    # outcome is generated. Only deterministic raw-wrapper receipts are added.
    folder=run.parent.PARENT/'00000'
    manifest=run.io.read(run.parent.PARENT/'manifest.json')
    assert run.io.digest(folder/'observations.sqlite')==manifest['files_sha256']['00000/observations.sqlite']
    meta=run.inventory(folder/'observations.sqlite')['metadata']
    plan,group=run.source_case(0)
    records=run.observations(folder/'observations.sqlite',meta,run.io.schedule(group))
    key,body,trace=next(records)
    assert key[-2:]==('baseline','none') and not body['row']['aborted']
    body,trace=deepcopy(body),deepcopy(trace)
    port=probe.TimestampPort();gate=probe.TimestampedGate(None,'lag_strong','baseline','projected_raw',port)
    previous=np.zeros(3)
    for entry in trace:
        port.publish(entry['perception'])
        command,info=gate.select(entry['observed_state'],previous)
        np.testing.assert_array_equal(command,entry['command'])
        entry.pop('velocity_projection',None)
        entry['age_adapter']=info['age_adapter'];previous=np.asarray(entry['command'])
    body['observation_adapter_inputs']=deepcopy(gate.receipts)
    return plan,group,body,trace


def test_full_new_auditor_accepts_raw_wrapper_on_recorded_native_physics(recorded_native_raw_fixture):
    plan,group,body,trace=recorded_native_raw_fixture
    _,auditor=run.native_functions('projected_raw')
    result=auditor(plan,group,dict(rule='baseline',model='none'),body,trace,None)
    assert result['public_preprocessing_replayed'] and result['received_frames']==len(trace)
    assert result['envelope_frames']==len(trace)


@pytest.mark.parametrize('kind',['processed_point','state_interval','source_timestamp'])
def test_forging_both_logged_receipt_copies_does_not_defeat_independent_replay(recorded_native_raw_fixture,kind):
    plan,group,body,trace=deepcopy(recorded_native_raw_fixture)
    copies=[body['observation_adapter_inputs'][0],trace[0]['age_adapter']]
    for item in copies:
        if kind=='processed_point':item['processed_state'][0][0]+=1
        elif kind=='state_interval':item['envelope']['lower'][0][0]-=1
        else:item['source_decision'][0]=1
    _,auditor=run.native_functions('projected_raw')
    with pytest.raises((ValueError,AssertionError)):
        auditor(plan,group,dict(rule='baseline',model='none'),body,trace,None)


def test_one_failed_condition_prevents_successful_subset_gain_claim():
    rows=[]
    for variant in probe.VARIANTS:
        for index in range(3):
            rows.append(dict(variant=variant,source_index=index,
                row=dict(controller='baseline',team_cost=10.,failure=False)))
            rows.append(dict(variant=variant,source_index=index,
                row=dict(controller='bank_performance_1802',team_cost=9.,failure=index==2)))
    result=verifier.controller_scopes(rows)
    assert all(r['all_scope_cost_gain_percent'] is None for r in result if r['controller']!='baseline')
    assert all(r['all_scope_cost_gain_percent']==0 for r in result if r['controller']=='baseline')
