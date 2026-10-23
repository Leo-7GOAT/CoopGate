"""Continuation tests use synthetic records, never inspect calibration effects."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import recover_prospective_v24 as recovery
from coopgate.episode_store_v21 import EpisodeStore, inventory, observations, readonly


def fixture(tmp_path, monkeypatch, n):
    prior, output = tmp_path/'prior', tmp_path/'recovery'
    monkeypatch.setattr(recovery,'PRIOR',prior)
    monkeypatch.setattr(recovery,'OUTPUT',output)
    monkeypatch.setattr(recovery,'RECOVERY',output/'recovery')
    plan = dict(scope='prospective',family='phase05',stages=recovery.original.stages('phase05'))
    spec = plan['stages']['calibration']
    spec.update(n_initials=1,shards=1)
    for root in (prior,output):
        (root/'frozen').mkdir(parents=True)
        (root/'frozen/plan.json').write_text(json.dumps(plan),encoding='utf-8')
    (output/'recovery').mkdir()
    (output/'recovery/plan.json').write_text('{}',encoding='utf-8')
    directory = prior/'calibration/blocks/00000'
    directory.mkdir(parents=True)
    metadata = recovery.original.block_metadata(prior,'calibration',0,None)
    expected = recovery.original.schedule(spec,0)
    state = np.array([[-80.,12.],[-85.,13.],[-120.,11.]])
    def shadow():
        return dict(initial=state.tolist(),seed=19367,scenario='observed_state_only_nominal_forecast',
            disturbance='clean',per_vehicle_finish=[10.,11.,12.],per_vehicle_variation=[1.,2.,3.],
            completed=True,collision=False,aborted=False,contract_violations=0,setup_ms=3.,controller_ms=4.)
    def record(case):
        row = shadow()
        row['controller'] = case[2]
        if case[-1] != 'observed_shadow':
            row.update(initial_index=0,seed=spec['seed_base'],scenario=recovery.original.regime_for(spec,0),
                       disturbance=case[1])
        trace = [dict(x=state.tolist(),command=[0.,0.,0.],disturbance=[0.,0.,0.],gate_ms=.1)]
        return row,trace
    store = EpisodeStore(directory/'observations.sqlite',metadata)
    for case in expected[:n]:
        store.append(case,*record(case))
    store.close()
    if n >= 10:
        rows = {case[2]:record(case)[0] for case in expected[:10]}
        prediction = recovery.original.prediction_from_rows(state,'phase05',rows)
        prediction.update(full_prediction_wall_ms=123.456,actual_prediction_runs=10)
        recovery.original.exclusive_json(directory/'decision.json',dict(prediction=prediction,choices=None,owners=None,
            calibration_sha256=None,selection_wall_ms=.123,actual_main_records_before_selection=0,
            observed_shadow_records_before_selection=10))
    entry = dict(initial_index=0,records=n,complete=False,database_sha256=recovery.original.digest(directory/'observations.sqlite'),
        decision_sha256=recovery.original.digest(directory/'decision.json') if n>=10 else None)
    env = SimpleNamespace(x=state.copy(),reset=lambda *args:None)
    class ShadowEnv:
        def __init__(self,value):
            np.testing.assert_equal(value,state)
        def __enter__(self):
            return self
        def __exit__(self,*args):
            pass
    calls = []
    def execute(env,case,spec,factory):
        # Every realized common-grid run must follow a persisted full prediction.
        saved = recovery.original.read(output/'calibration/blocks/00000/decision.json')
        assert saved['observed_shadow_records_before_selection']==10
        calls.append(tuple(case)+('collection',))
        return record(tuple(case)+('collection',))
    def run_episode(env,seed,scenario,mode,factory,trace):
        assert (seed,scenario,mode,trace)==(1,'unused','clean',True)
        case = (0,'clean',factory,'observed_shadow')
        calls.append(case)
        row,steps = record(case)
        row.pop('controller')
        return row,steps,state.copy()
    monkeypatch.setattr(recovery,'ObservableInitialMergeEnv',ShadowEnv)
    monkeypatch.setattr(recovery,'run_episode',run_episode)
    monkeypatch.setattr(recovery.original,'execute',execute)
    verified = []
    monkeypatch.setattr(recovery.original,'verify_record',lambda row,trace,spec:verified.append(row))
    fs = {name:name for name in spec['methods']}
    return SimpleNamespace(prior=prior,output=output,directory=directory,plan=plan,spec=spec,entry=entry,
        metadata=metadata,expected=expected,env=env,fs=fs,calls=calls,verified=verified,record=record)


@pytest.mark.parametrize('n',[9,10,25,28,49])
def test_complete_only_exact_missing_suffix_and_preserve_original_blobs(tmp_path,monkeypatch,n):
    f = fixture(tmp_path,monkeypatch,n)
    originals = {p.name:p.read_bytes() for p in f.directory.iterdir()}
    with readonly(f.directory/'observations.sqlite') as connection:
        raw = connection.execute('SELECT * FROM episodes ORDER BY position').fetchall()
    recovery.complete_partial(f.entry,f.plan,f.env,f.fs)
    new = f.output/'calibration/blocks/00000'
    assert f.calls == f.expected[n:]
    assert len(f.verified)==50-max(n,10)
    assert originals == {p.name:p.read_bytes() for p in f.directory.iterdir()}
    with readonly(new/'observations.sqlite') as connection:
        assert connection.execute('SELECT * FROM episodes ORDER BY position').fetchall()[:n]==raw
    loaded = list(observations(new/'observations.sqlite',f.metadata,f.expected))
    assert len(loaded)==50
    for key,row,trace in loaded:
        assert (row,trace)==f.record(key)
    receipt = recovery.original.read(new/'complete.json')
    assert receipt['full_initial_wall_ms'] is None
    assert receipt['new_sumo_roles']==50-n
    assert receipt['retained_sumo_roles']==n
    assert receipt['resumed_segment_wall_ms']>0
    decision = recovery.original.read(new/'decision.json')
    if n<10:
        assert decision['prediction']['full_prediction_wall_ms'] is None
        assert decision['prediction']['resumed_prediction_wall_ms']>0
    else:
        assert (new/'decision.json').read_bytes()==originals['decision.json']
        assert decision['prediction']['full_prediction_wall_ms']==123.456
    recovery.original.load_block(f.output,f.plan,'calibration',0,None)


@pytest.mark.parametrize('change',['hole','duplicate','mode','metadata','checksum_count','empty','complete'])
def test_reject_malformed_or_unregistered_prefix(tmp_path,monkeypatch,change):
    f = fixture(tmp_path,monkeypatch,9)
    state = inventory(f.directory/'observations.sqlite')
    if change=='hole': state['cases'][3]=list(f.expected[4])
    elif change=='duplicate': state['cases'][3]=state['cases'][2]
    elif change=='mode': state['cases'][0][1]='pulse'
    elif change=='metadata': state['metadata']['plan_sha256']='different'
    elif change=='checksum_count': state['row_sha256'].pop()
    elif change=='empty': state.update(cases=[],row_sha256=[],trace_sha256=[])
    elif change=='complete': state.update(cases=list(map(list,f.expected)),row_sha256=['a']*50,trace_sha256=['b']*50)
    with pytest.raises(ValueError): recovery.prefix_length(state,f.metadata,f.expected)


def test_reject_source_byte_changes_and_never_reopen_original(tmp_path,monkeypatch):
    f = fixture(tmp_path,monkeypatch,9)
    path = f.directory/'observations.sqlite'
    with pytest.raises(ValueError):
        recovery.copy_prefix(path,path,f.metadata,f.expected,f.entry['database_sha256'])
    with pytest.raises(ValueError):
        recovery.copy_prefix(path,tmp_path/'new.sqlite',f.metadata,f.expected,'wrong')
    assert not (tmp_path/'new.sqlite').exists()


def test_detect_corrupt_compressed_blob_even_if_inventory_column_is_unchanged(tmp_path,monkeypatch):
    f = fixture(tmp_path,monkeypatch,9)
    path = f.directory/'observations.sqlite'
    import zlib
    with sqlite3.connect(path) as connection:
        connection.execute('UPDATE episodes SET row_blob=? WHERE position=2',(zlib.compress(b'{}'),))
    with pytest.raises(ValueError,match='checksum'):
        recovery.prefix_records(path,f.metadata,f.expected,recovery.original.digest(path))


def test_new_failure_is_saved_without_sealing_or_modifying_original(tmp_path,monkeypatch):
    f = fixture(tmp_path,monkeypatch,25)
    before = (f.directory/'observations.sqlite').read_bytes()
    def fail(row,trace,spec):
        raise ValueError('deliberate verifier failure')
    monkeypatch.setattr(recovery.original,'verify_record',fail)
    with pytest.raises(ValueError,match='deliberate'):
        recovery.complete_partial(f.entry,f.plan,f.env,f.fs)
    new = f.output/'calibration/blocks/00000'
    assert len(inventory(new/'observations.sqlite')['cases'])==26
    assert not (new/'complete.json').exists()
    assert not (new/'observations.sqlite.seal.json').exists()
    assert (f.directory/'observations.sqlite').read_bytes()==before
    with pytest.raises(FileExistsError):
        recovery.complete_partial(f.entry,f.plan,f.env,f.fs)


def test_boundary_excludes_only_timings_and_preserves_bad_replay_before_assertion(tmp_path,monkeypatch):
    f = fixture(tmp_path,monkeypatch,9)
    original_run = recovery.run_episode
    def different(*args,**kwargs):
        row,trace,state = original_run(*args,**kwargs)
        trace[0]['command'][0]=1.
        return row,trace,state
    monkeypatch.setattr(recovery,'run_episode',different)
    with pytest.raises(AssertionError):
        recovery.boundary(f.entry,f.plan,f.env,f.fs)
    directory = f.output/'calibration/boundary/00000'
    assert (directory/'observation.json').exists()
    assert not (directory/'verified.json').exists()
    assert not (f.output/'calibration/blocks').exists()


def test_boundary_phase_precedes_any_missing_data_and_final_keeps_original_budget():
    assert recovery.phases('calibration')==('boundary','collect','review')
    assert recovery.phases('confirmation')==('collect','review')
    specs=recovery.original.stages('phase05')
    assert specs['calibration']['n_initials']==4096
    assert specs['confirmation']['n_initials']==1024
    assert specs['calibration']['total_sumo_runs']==204800
    assert specs['confirmation']['total_sumo_runs']==92160
