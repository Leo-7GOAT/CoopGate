from copy import deepcopy
from pathlib import Path
import pickle
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import joint_recovery_20261028 as recovery
from coopgate.episode_store_v21 import EpisodeStore, inventory


def test_worker_watcher_and_inference_bodies_remain_original_and_pool_is_picklable():
    original=recovery.original
    namespace=recovery.routed_functions()
    for name in ('worker','watch','finalize','stage_terminal','validate_terminal'):
        assert namespace[name].__code__ is getattr(original,name).__code__
    assert namespace['OUT']==recovery.OUT
    assert original.OUT!=recovery.OUT
    assert namespace['collect'].audit_record is original.collect.audit_record
    assert namespace['certificate_from_complete_records'] is original.certificate_from_complete_records
    assert pickle.loads(pickle.dumps(recovery.replay.initialize)) is recovery.replay.initialize
    assert pickle.loads(pickle.dumps(recovery.replay.audit_block)) is recovery.replay.audit_block


def test_prefix_identity_cannot_drop_duplicate_or_change_cases():
    metadata={'study':'same'};expected=[['x','a'],['x','b'],['x','c'],['x','d']]
    assert recovery.prefix_state(dict(metadata=metadata,cases=expected[:3]),metadata,expected)==3
    assert recovery.prefix_state(dict(metadata=metadata,cases=expected),metadata,expected)==4
    for cases in (expected[1:3],[expected[0],expected[0]],expected+[['x','e']]):
        with pytest.raises(ValueError):recovery.prefix_state(dict(metadata=metadata,cases=cases),metadata,expected)
    with pytest.raises(ValueError):recovery.prefix_state(dict(metadata={},cases=[]),metadata,expected)


def test_complete_prefix_is_copied_exactly_without_calling_simulator(tmp_path,monkeypatch):
    prior=tmp_path/'prior';snap=tmp_path/'snap';snap.mkdir()
    source=prior/'calibration/blocks/00000';source.mkdir(parents=True)
    for name,value in [('observations.sqlite',b'durable bytes'),('complete.json',b'complete'),('observations.sqlite.seal.json',b'sealed')]:
        (source/name).write_bytes(value)
    entry=dict(status='complete',files={p.name:recovery.original.io.digest(p) for p in source.iterdir()})
    recovery.original.io.exclusive_json(snap/'inventory.json',dict(entries=[entry]))
    monkeypatch.setattr(recovery,'PRIOR',prior);monkeypatch.setattr(recovery,'SNAP',snap)
    monkeypatch.setattr(recovery,'inputs',lambda **kwargs:{})
    def forbidden(*args,**kwargs):raise AssertionError('Retained complete block must not execute SUMO')
    monkeypatch.setattr(recovery.original.collect,'collect_cluster',forbidden)
    target=tmp_path/'new'
    recovery.collect_cluster(dict(stage='calibration',index=0),target,{})
    assert {p.name:p.read_bytes() for p in target.iterdir()}=={p.name:p.read_bytes() for p in source.iterdir()}
    (source/'complete.json').write_bytes(b'changed')
    with pytest.raises(ValueError,match='bytes changed'):
        recovery.collect_cluster(dict(stage='calibration',index=0),tmp_path/'another',{})


def test_replays_cannot_be_reported_as_extra_independent_samples():
    proof=dict(entries=[dict(index=0,status='complete',records=4),dict(index=1,status='unsealed_boundary',records=3)])
    plan=dict(retained_complete_blocks=[0],retained_complete_runs=4,original_partial_engineering_runs=3,additional_sumo_runs=16380)
    recovery.validate_accounting(plan,proof)
    for key in plan:
        changed=deepcopy(plan)
        changed[key]=[] if key=='retained_complete_blocks' else changed[key]+1
        with pytest.raises(ValueError,match='accounting'):
            recovery.validate_accounting(changed,proof)


def boundary_fixture(tmp_path,change=None):
    case=dict(cluster_id='x',methods=['baseline','1802','1812','1822'])
    metadata={'same':'identity'}
    old=tmp_path/'old.sqlite';new=tmp_path/'new.sqlite'
    first=EpisodeStore(old,metadata);second=EpisodeStore(new,metadata)
    for i,method in enumerate(case['methods']):
        body=dict(cluster_id='x',method=method,row=dict(completed=False,collision=True,
            per_vehicle_cost=[1,2,3],controller_ms=[1.]),calls=[dict(wall_ms=1.)],episode_wall_seconds=1.)
        trace=[dict(command=[1,2,3],gate_ms=1.)]
        if i<3:first.append(['x',method],body,trace)
        new_body=deepcopy(body);new_trace=deepcopy(trace)
        new_body['row']['controller_ms']=[10.];new_body['calls'][0]['wall_ms']=10.
        new_body['episode_wall_seconds']=10.;new_trace[0]['gate_ms']=10.
        if i==0 and change=='cost':new_body['row']['per_vehicle_cost'][0]+=1
        if i==0 and change=='failure':new_body['row']['collision']=False
        if i==0 and change=='command':new_trace[0]['command'][0]+=1
        second.append(['x',method],new_body,new_trace)
    first.close();second.seal([['x',m] for m in case['methods']])
    return old,new,dict(inventory=inventory(old)),case


def test_boundary_preserves_real_failure_and_ignores_only_timing(tmp_path):
    assert recovery.verify_boundary(*boundary_fixture(tmp_path))==3


@pytest.mark.parametrize('change',['cost','failure','command'])
def test_boundary_rejects_every_non_timing_difference(tmp_path,change):
    with pytest.raises(ValueError,match='non-timing'):
        recovery.verify_boundary(*boundary_fixture(tmp_path,change))
