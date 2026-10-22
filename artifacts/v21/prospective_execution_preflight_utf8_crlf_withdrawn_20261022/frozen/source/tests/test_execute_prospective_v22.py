from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import execute_prospective_v22 as execution
from coopgate.episode_store_v21 import inventory, observations
from coopgate.prospective_validation_v21 import MODELS, profile_names
from observable_profile_prediction_v22 import choose_profiles, prediction_from_rows


def test_windows_process_snapshot_keeps_chinese_paths_and_fails_closed(monkeypatch):
    import json
    from types import SimpleNamespace
    records=[dict(ProcessId=41,ParentProcessId=7,Name='python.exe',
                  CommandLine='python C:/Users/Leo/Desktop/论文整改/训练.py')]
    def run(command,**kwargs):
        assert '[Console]::OutputEncoding' in command[-1]
        assert '$OutputEncoding' in command[-1]
        assert kwargs['text'] is False and kwargs['check'] is True
        return SimpleNamespace(stdout=json.dumps(records,ensure_ascii=False).encode('utf-8'))
    monkeypatch.setattr(execution.subprocess,'run',run)
    assert execution.native_snapshot()==records
    records[0]['ProcessId']=None
    with pytest.raises(ValueError): execution.native_snapshot()
    monkeypatch.setattr(execution.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout=b'\xff'))
    with pytest.raises(UnicodeDecodeError): execution.native_snapshot()


def test_registered_data_and_compute_budgets_are_not_conflated():
    splits = execution.stages('phase05')
    assert [(splits[n]['n_initials'], splits[n]['main_runs'], splits[n]['total_sumo_runs'])
            for n in ('preflight','calibration','confirmation')] == [(1,40,90),(4096,163840,204800),(1024,40960,92160)]
    assert sum(splits[n]['total_sumo_runs'] for n in ('calibration','confirmation')) == 296960
    for stage, spec in splits.items():
        ids = [i for shard in range(spec['shards']) for i in execution.assigned(spec,shard)]
        assert sorted(ids) == list(range(spec['n_initials']))
        roles = execution.schedule(spec,0)
        assert len(roles) == len(set(roles)) == (90 if stage!='calibration' else 50)
        assert all(k[-1]=='observed_shadow' for k in roles[:10])
        assert len({(m,n) for _,m,n,r in roles if r=='collection'}) == 40
    first, final = splits['calibration'], splits['confirmation']
    assert first['seed_base']+first['n_initials'] <= final['seed_base']
    assert splits['preflight']['seed_base'] == 63100001


@pytest.mark.parametrize('scope,stage', [('preflight','calibration'),('preflight','confirmation'),('prospective','preflight')])
def test_preflight_cannot_authorize_new_data(scope,stage):
    with pytest.raises(ValueError):
        execution.stage_spec(dict(scope=scope,stages=execution.stages('phase05')),stage)


def prediction_fixture():
    state = [[-80.,12.],[-85.,13.],[-120.,11.]]
    rows = {}
    for name in ('baseline',*profile_names('phase05')):
        rows[name] = dict(controller=name, initial=state, seed=19367,
            scenario='observed_state_only_nominal_forecast', disturbance='clean',
            per_vehicle_finish=[10. if name=='baseline' else 9.]*3, per_vehicle_variation=[0.]*3,
            completed=True, collision=False, aborted=False, contract_violations=0)
    return state, rows


def fixed_choices():
    state,rows = prediction_fixture()
    return choose_profiles(prediction_from_rows(state,'phase05',rows),execution.synthetic_calibration('phase05'),'phase05')


def test_dispatch_keeps_all_models_and_whole_controllers_with_explicit_aliases():
    choices = fixed_choices()
    owners = execution.deployment_owners(choices,'phase05')
    assert sum(map(len,owners.values())) == 18
    assert len(owners) == 10
    assert ['ltt','1812'] in owners['baseline']
    state,_ = prediction_fixture()
    calls = []
    fs = {name:(lambda x,n=name:calls.append(n) or n) for name in owners}
    dispatch = execution.dispatch_factory('baseline',choices,'phase05',state,fs)
    assert dispatch(np.array(state)) == 'baseline'
    assert calls == ['baseline']
    with pytest.raises(AssertionError):
        dispatch(np.array(state)+.1)
    assert calls == ['baseline']
    choices['ltt']['1812'] = 'raw_1802'
    with pytest.raises(ValueError):
        execution.deployment_owners(choices,'phase05')


@pytest.mark.parametrize('change', ['missing_model','missing_adapter','changed_fixed'])
def test_deployment_rejects_cherry_picked_or_changed_comparisons(change):
    choices = fixed_choices()
    if change=='missing_model':
        del choices['ltt']['1822']
    elif change=='missing_adapter':
        del choices['mean_spi']
    else:
        choices['phase05']['1802']='baseline'
    with pytest.raises(ValueError):
        execution.deployment_owners(choices,'phase05')


def test_primary_requires_every_model_and_cannot_be_rescued_by_a_comparator():
    cal = dict(primary_certificate=dict(all_fixed_models_both_targets_certified=True))
    members = {str(s):dict(runs=1024,actual_random_mode=dict(adopted=256),mean_team_gain_percent=.8,
                         mean_team_gain_ci95=[.01,.1],completed=1024) for s in MODELS}
    result = dict(results={'phase05':dict(per_training_seed=members), 'conformal':dict(anything='better')})
    assert execution.final_acceptance(cal,result,'phase05')['complete_primary_success']
    cal['primary_certificate']['all_fixed_models_both_targets_certified']=False
    assert not execution.final_acceptance(cal,result,'phase05')['complete_primary_success']
    cal['primary_certificate']['all_fixed_models_both_targets_certified']=True
    members['1812']['mean_team_gain_ci95']=[-.001,.2]
    assert not execution.final_acceptance(cal,result,'phase05')['complete_primary_success']
    members['1812']['mean_team_gain_ci95']=[.01,.1]
    members['1822']['actual_random_mode']['adopted']=0
    assert not execution.final_acceptance(cal,result,'phase05')['complete_primary_success']
    del members['1822']
    with pytest.raises(ValueError):
        execution.final_acceptance(cal,result,'phase05')


def native_fixture(tmp_path):
    directory = tmp_path/'execution'
    directory.mkdir()
    put = lambda name,value:execution.exclusive_json(directory/name,value)
    put('watcher_launch.json',dict(plan_sha256='plan',stage='calibration',supervisor_pid=101))
    put('supervisor_exit.json',dict(plan_sha256='plan',stage='calibration',supervisor_pid=101,exit_code=0))
    for phase,pids in [('collect',[201,202,203]),('review',[301,302,303])]:
        put(phase+'_launch.json',dict(plan_sha256='plan',stage='calibration',supervisor_pid=101,worker_pids=pids))
        put(phase+'_exits.json',dict(workers=[dict(pid=p,exit_code=0) for p in pids]))
    put('completed.json',dict(plan_sha256='plan',stage='calibration'))
    return directory


@pytest.mark.parametrize('change', ['none','supervisor_failed','worker_running','review_failed','wrong_pid','wrong_plan','missing_worker','missing_receipt'])
def test_analysis_needs_real_complete_successful_process_receipts(tmp_path,change):
    directory = native_fixture(tmp_path)
    path = directory/'supervisor_exit.json'
    if change=='missing_receipt':
        path.unlink()
    elif change!='none':
        if change in ('worker_running','review_failed','missing_worker'):
            path = directory/('review_exits.json' if change=='review_failed' else 'collect_exits.json')
        value = execution.read(path)
        if change=='supervisor_failed': value['exit_code']=1
        if change=='wrong_pid': value['supervisor_pid']=999
        if change=='wrong_plan': value['plan_sha256']='different'
        if change=='worker_running': value['workers'][0]['exit_code']=None
        if change=='review_failed': value['workers'][1]['exit_code']=1
        if change=='missing_worker': value['workers'].pop()
        path.write_text(__import__('json').dumps(value))
    if change=='none':
        execution.require_native_completion(tmp_path,'plan','calibration',3)
    else:
        with pytest.raises((ValueError,FileNotFoundError)):
            execution.require_native_completion(tmp_path,'plan','calibration',3)


def test_shadow_identity_uses_actual_commands_and_states_not_timing():
    trace = [dict(command=[1.,0.,-1.],x=[[-80.,12.],[-85.,13.],[-120.,11.]],disturbance=[[0.,0.,0.]],gate_ms=1.)]
    other = deepcopy(trace)
    other[0]['gate_ms']=3000.
    execution.verify_shadow_trace(trace,other)
    other[0]['command'][0]=2.
    with pytest.raises(AssertionError): execution.verify_shadow_trace(trace,other)
    with pytest.raises(ValueError): execution.verify_shadow_trace(trace,[])


@pytest.mark.parametrize('interrupt', [False,True])
def test_actual_store_preserves_selection_order_and_interrupted_prefix(tmp_path,monkeypatch,interrupt):
    family='phase05'
    plan=dict(scope='preflight',family=family,stages=execution.stages(family),models={})
    (tmp_path/'frozen').mkdir()
    (tmp_path/'frozen/plan.json').write_text('{}')
    state,shadows=prediction_fixture()
    class Environment:
        x=np.array(state)
        def reset(self,*args): pass
    def forecast(initial,protocol,family,on_record):
        for name,row in shadows.items(): on_record(name,row,[])
        return dict(**prediction_from_rows(initial,family,shadows),full_prediction_wall_ms=123.,actual_prediction_runs=10)
    calls=[]
    def execute(env,case,spec,factory):
        decision=execution.read(tmp_path/'preflight/blocks/00000/decision.json')
        assert decision['choices']==fixed_choices()
        assert decision['actual_main_records_before_selection']==0
        if interrupt and len(calls)==1: raise RuntimeError('simulated infrastructure interruption')
        calls.append(case)
        index,mode,name=case
        row=dict(shadows[name],initial_index=index,seed=spec['seed_base']+index,
            disturbance=mode,scenario=execution.regime_for(spec,index),completed=False,aborted=True)
        # The intentionally poor realized outcome cannot influence prior choice.
        row['per_vehicle_finish']=[25.]*3
        factory(np.array(state))
        return row,[]
    monkeypatch.setattr(execution,'forecast',forecast)
    monkeypatch.setattr(execution,'execute',execute)
    monkeypatch.setattr(execution,'verify_record',lambda *args:None)
    fs={name:lambda x:object() for name in shadows}
    cal=execution.synthetic_calibration(family)
    cal_sha=execution.value_digest(cal)
    def collect(): execution.collect_initial(tmp_path,plan,'preflight',0,Environment(),fs,cal,cal_sha)
    path=tmp_path/'preflight/blocks/00000/observations.sqlite'
    if interrupt:
        with pytest.raises(RuntimeError): collect()
        assert len(inventory(path)['cases'])==11
        assert not Path(str(path)+'.seal.json').exists()
        with pytest.raises(FileNotFoundError): list(observations(path,{},[]))
    else:
        collect()
        _,records=execution.load_block(tmp_path,plan,'preflight',0,cal_sha)
        assert len(records)==90
        assert all(row['aborted'] for key,row,_ in records if key[-1]!='observed_shadow')
    before=execution.digest(path)
    with pytest.raises(FileExistsError): collect()
    assert execution.digest(path)==before
