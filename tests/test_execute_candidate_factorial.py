import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import execute_candidate_factorial as execute
import freeze_fair_comparison_v39 as design
from coopgate.candidate_ablation_v43 import VARIANTS
from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.disturbance_v9 import disturbance_tape
from coopgate.episode_store_v21 import inventory, observations
from coopgate.matched_shift_v21 import matched_plant_parameters


def parent_plan():
    return dict(cases=[dict(case_id=f'case-{i}',independent_cluster=f'cluster-{i}',initial_id=i,
        scenario='nominal',dynamics='nominal',disturbance='clean',policy_id='1802',
        policy_sha256='policy',initial_sha256='initial',dynamics_sha256='plant',
        disturbance_sha256='tape') for i in range(640)])


def test_schedule_preserves_every_parent_identity_and_balances_positions():
    parent = parent_plan()
    cases = execute.cases_from_parent(parent)
    counts = {v:[0]*5 for v in VARIANTS}
    for case,original in zip(cases,parent['cases']):
        assert case['cluster_id']==original['independent_cluster']
        assert case['seed']==original['initial_id']
        assert case['parent_case_id']==original['case_id']
        assert len(execute.schedule(case))==5
        for position,variant in enumerate(case['variants']):
            counts[variant][position]+=1
    assert all(v==[128]*5 for v in counts.values())


@pytest.mark.parametrize('mutation',['missing','duplicate','extra'])
def test_case_count_or_independent_identity_cannot_change(mutation):
    plan=parent_plan()
    if mutation=='missing': plan['cases'].pop()
    elif mutation=='extra': plan['cases'].append(plan['cases'][0])
    else: plan['cases'][-1]['independent_cluster']=plan['cases'][0]['independent_cluster']
    with pytest.raises(ValueError,match='640'):
        execute.cases_from_parent(plan)


def records():
    return [dict(cluster_id='fixture',variant=v,row=dict(per_vehicle_cost=[c]*3,
        per_vehicle_finish=[8.]*3,team_cost=c,learned_action_steps=int(v!='baseline'),
        completed=True,aborted=False,collision=False,contract_violations=0))
        for v,c in zip(VARIANTS,[10.,9.,8.,7.,6.])]


def test_valid_factorial_interaction_and_complete_means():
    result=execute.analysis(['fixture'],records())
    assert result['all_scope_contrasts']['interaction']['gain_percentage_points']==pytest.approx(0)
    assert result['all_scope_method_means']['projected_bank']['gain_percentage_points']==pytest.approx(40)
    assert not result['new_risk_certificate']


@pytest.mark.parametrize('failed_variant',['baseline','raw_nominal'])
def test_capped_failed_cost_cannot_be_presented_as_valid_gain(failed_variant):
    rows=records()
    row=next(r for r in rows if r['variant']==failed_variant)['row']
    row.update(completed=False,aborted=True,learned_action_steps=0)
    result=execute.analysis(['fixture'],rows)
    assert result['failed_pairs_removed']==0
    assert result['all_scope_contrasts']['interaction']['gain_percentage_points'] is None
    assert result['all_scope_method_means']['raw_nominal']['gain_percentage_points'] is None
    if failed_variant=='baseline':
        assert all(v['gain_percentage_points'] is None for v in result['all_scope_contrasts'].values())
    else:
        assert result['all_scope_method_means']['projected_bank']['gain_percentage_points']==pytest.approx(40)


@pytest.mark.parametrize('command',['watch','worker','summarize'])
def test_other_studies_prevent_execution_before_any_mutation(tmp_path,monkeypatch,command):
    monkeypatch.setattr(execute,'OUT',tmp_path/'forbidden')
    monkeypatch.setattr(execute.io,'native_snapshot',lambda:[])
    def refuse(*args): raise ValueError('Other project study processes')
    monkeypatch.setattr(execute.isolation,'require_quiet',refuse)
    with pytest.raises(ValueError,match='Other project'):
        getattr(execute,command)()
    assert not execute.OUT.exists()


def test_audit_counterexample_is_saved_before_stop_and_never_sealed(tmp_path,monkeypatch):
    case=dict(cluster_id='counterexample',parent_case_id='parent',block_index=0,plant='nominal',variants=list(VARIANTS))
    class Env:
        def __init__(self,**kwargs): pass
        def __enter__(self): return self
        def __exit__(self,*args): pass
    monkeypatch.setattr(execute.runtime,'MatchedShiftedMergeEnv',Env)
    monkeypatch.setattr(execute.runtime,'execute_episode',lambda c,v,*args:dict(cluster_id=c['cluster_id'],variant=v,row={},trace=[]))
    target=tmp_path/'block'
    def reject(*args):
        assert len(inventory(target/'observations.sqlite')['cases'])==1
        raise ValueError('physical certificate counterexample')
    monkeypatch.setattr(execute.runtime,'audit_episode',reject)
    with pytest.raises(ValueError,match='counterexample'):
        execute.collect(case,target,None,'development')
    assert not (target/'complete.json').exists()
    assert not (target/'observations.sqlite.seal.json').exists()
    with pytest.raises(FileExistsError): execute.collect(case,target,None,'development')


def test_freeze_refuses_parent_outcomes_before_creating_plan(tmp_path,monkeypatch):
    old=tmp_path/'parent';(old/'execution').mkdir(parents=True)
    monkeypatch.setattr(execute.parent,'OUT',old)
    monkeypatch.setattr(execute,'OUT',tmp_path/'new')
    with pytest.raises(ValueError,match='before parent formal outcomes'):
        execute.freeze()
    assert not execute.OUT.exists()


def test_full_replay_rejects_terminal_rewrite_before_any_effect_output(tmp_path,monkeypatch):
    monkeypatch.setattr(execute,'OUT',tmp_path)
    plan_path=tmp_path/'plan.json';plan_path.write_text('{}')
    monkeypatch.setattr(execute,'PLAN',plan_path)
    cases=execute.cases_from_parent(parent_plan())
    plan=dict(cases=cases,models={'1802':{'path':'unused'}})
    monkeypatch.setattr(execute,'inputs',lambda:plan)
    monkeypatch.setattr(execute,'terminal',lambda sha:{'verified':'unchanged fixture'})
    monkeypatch.setattr(execute,'load_coupled',lambda path:(None,None))
    monkeypatch.setattr(execute.io,'native_snapshot',lambda:[])
    monkeypatch.setattr(execute.isolation,'require_quiet',lambda *args:None)
    target=tmp_path/'execution';target.mkdir()
    for name in (*execute.TERMINAL_FILES,'exit.json'):
        (target/name).write_text('{}')
    digest,read=execute.io.digest,execute.io.read
    sha=digest(plan_path)
    def fake_digest(path):
        return 'database' if path.name in ('observations.sqlite','observations.sqlite.seal.json') else digest(path)
    def fake_read(path):
        if path.name!='complete.json': return read(path)
        case=cases[int(path.parent.name)]
        return dict(**execute.metadata(case,sha),actual_sumo_runs=5,database_sha256='database',
                    seal_sha256='database',reviews=[{'variant':v} for v in case['variants']])
    monkeypatch.setattr(execute.io,'digest',fake_digest)
    monkeypatch.setattr(execute.io,'read',fake_read)
    def observe(path,meta,schedule):
        for cluster,variant in schedule:
            yield [cluster,variant],dict(cluster_id=cluster,variant=variant),[]
    monkeypatch.setattr(execute,'observations',observe)
    def replay(*args):
        (target/'exit.json').write_text('{ }')
        return {}
    monkeypatch.setattr(execute.runtime,'audit_episode',replay)
    monkeypatch.setattr(execute,'analysis',lambda *args:pytest.fail('Must reject before reporting effects'))
    with pytest.raises(ValueError,match='Evidence changed'):
        execute.summarize()
    assert not (tmp_path/'analysis').exists()


def terminal_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(execute,'OUT',tmp_path)
    target=tmp_path/'execution';target.mkdir()
    identity=dict(launcher_pid=12,native_pid=13,parent_pid=12,native_creation_date='fixed',command_line=execute.SCRIPT+' worker')
    entries=[dict(block_index=i,files={}) for i in range(640)]
    values={name:{} for name in execute.TERMINAL_FILES}
    values['spawn.json']=dict(plan_sha256='plan',launcher_pid=12)
    values['launch.json']=identity
    values['blocks_manifest.json']=dict(plan_sha256='plan',actual_sumo_runs=3200,entries=entries)
    for name,value in values.items(): execute.io.exclusive_json(target/name,value)
    done=dict(plan_sha256='plan',native_pid=13,actual_sumo_runs=3200,blocks=640,
              blocks_manifest_sha256=execute.io.digest(target/'blocks_manifest.json'))
    (target/'collection_complete.json').write_text(json.dumps(done))
    receipt=dict(**identity,exit_code=0,plan_sha256='plan',hashes={name:execute.io.digest(target/name) for name in execute.TERMINAL_FILES})
    execute.io.exclusive_json(target/'exit.json',receipt)
    monkeypatch.setattr(execute.io,'check_acknowledgment',lambda *args:None)
    monkeypatch.setattr(execute.io,'native_snapshot',lambda:[])
    monkeypatch.setattr(execute,'block_hashes',lambda directory,index:entries[index])
    return target,receipt


def test_complete_terminal_accepts_and_changed_block_rejects(tmp_path,monkeypatch):
    terminal_fixture(tmp_path,monkeypatch)
    assert len(execute.terminal('plan')['entries'])==640
    monkeypatch.setattr(execute,'block_hashes',lambda directory,index:dict(block_index=index,files={'changed':'yes'}))
    with pytest.raises(ValueError,match='Changed completed'):
        execute.terminal('plan')


@pytest.mark.parametrize('mutation',['nonzero','hash','live'])
def test_terminal_rejects_nonzero_rewrite_and_live_native_worker(tmp_path,monkeypatch,mutation):
    target,receipt=terminal_fixture(tmp_path,monkeypatch)
    if mutation=='nonzero':
        receipt['exit_code']=7;(target/'exit.json').write_text(json.dumps(receipt))
    elif mutation=='hash': (target/'launch.json').write_text('{}')
    else: monkeypatch.setattr(execute.io,'native_snapshot',lambda:[dict(ProcessId=13,CreationDate='fixed')])
    with pytest.raises(ValueError): execute.terminal('plan')


def test_real_native_nonzero_exit_is_preserved_without_fake_completion(tmp_path,monkeypatch):
    plan=tmp_path/'plan.json';plan.write_text('{}')
    script=tmp_path/'failing_worker.py'
    script.write_text('import os,json,time,sys\nfrom pathlib import Path\n'
        f'p=Path({str(tmp_path/"execution")!r})\n'
        '(p/"worker.native.json").write_text(json.dumps(dict(native_pid=os.getpid(),parent_pid=os.getppid())))\n'
        'deadline=time.monotonic()+30\n'
        'while not (p/"worker.acknowledged.json").exists():\n'
        ' if time.monotonic()>deadline: sys.exit(9)\n'
        ' time.sleep(.05)\n'
        'sys.exit(7)\n')
    monkeypatch.setattr(execute,'OUT',tmp_path)
    monkeypatch.setattr(execute,'PLAN',plan)
    monkeypatch.setattr(execute,'SCRIPT',script.as_posix())
    monkeypatch.setattr(execute,'inputs',lambda:None)
    monkeypatch.setattr(execute.isolation,'require_quiet',lambda *args:None)
    with pytest.raises(SystemExit) as error: execute.watch()
    assert error.value.code==7
    receipt=execute.io.read(tmp_path/'execution/exit.json')
    assert receipt['exit_code']==7 and receipt['native_pid']>0
    assert not (tmp_path/'execution/collection_complete.json').exists()


@pytest.mark.skipif(os.environ.get('COOPGATE_RUN_SUMO_SMOKE')!='1',reason='Opt-in real SUMO')
@pytest.mark.parametrize('broken',[False,True])
def test_five_real_cells_persist_replay_and_preserve_pre_adoption_failure(tmp_path,broken):
    torch.set_num_threads(1)
    seed,scene,plant,mode=139_889_001+int(broken),'nominal','lag_strong','clean'
    models=design.load_models(ROOT)
    case=dict(block_index=0,cluster_id='factorial-development',parent_case_id='development',
        seed=seed,scenario=scene,plant=plant,disturbance=mode,policy_id='1802',variants=list(VARIANTS),
        initial_sha256=design.array_sha256(design.planned_initial(seed,scene),decimals=8),
        dynamics_sha256=design.canonical_sha256(matched_plant_parameters(seed,plant,.1)),
        disturbance_sha256=design.array_sha256(disturbance_tape(seed,mode,.1,250)))
    class Broken:
        def proposal(self,*args): raise RuntimeError('Injected factorial pre-adoption policy error')
    model=Broken() if broken else load_coupled(ROOT/models['1802']['path'])[0]
    directory=tmp_path/'block'
    receipt=execute.collect(case,directory,model,'development-not-formal')
    assert receipt['actual_sumo_runs']==5
    records=[]
    for key,body,trace in observations(directory/'observations.sqlite',execute.metadata(case,'development-not-formal'),execute.schedule(case)):
        review=execute.runtime.audit_episode(case,dict(**body,trace=trace),model)
        assert review=={k:v for k,v in receipt['reviews'][case['variants'].index(body['variant'])].items() if k!='variant'}
        records.append(body)
    report=execute.analysis([case['cluster_id']],records)
    assert report['failed_pairs_removed']==0
    if broken:
        assert all(m['failure_probability']==1 and m['adoption_probability']==0
                   and m['gain_percentage_points'] is None for m in report['all_scope_method_means'].values())
    else:
        assert all(m['gain_percentage_points'] is not None for m in report['all_scope_method_means'].values())
