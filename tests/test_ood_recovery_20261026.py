import json
from pathlib import Path

import pytest

from coopgate.episode_store_v21 import EpisodeStore, inventory, readonly
import recover_ood_20261026 as recovery


def fixture(tmp_path,monkeypatch):
    prior,out=tmp_path/'prior',tmp_path/'out'
    monkeypatch.setattr(recovery,'PRIOR',prior);monkeypatch.setattr(recovery,'OUT',out)
    group=dict(block_index=0,initial_id='fixed',seed=17,regime='fixed',plant='fixed',
               perception='fixed',method_order=recovery.io.design.method_keys())
    expected=recovery.io.schedule(group)
    path=prior/'blocks/00000/observations.sqlite';path.parent.mkdir(parents=True)
    meta=recovery.io.metadata(group,'p','b');store=EpisodeStore(path,meta)
    for index in range(2):store.append(expected[index],dict(index=index),[dict(value=index)])
    store.close()
    entry=dict(index=0,records=2,complete=False,database_sha256=recovery.io.digest(path),inventory=inventory(path))
    (out/'blocks').mkdir(parents=True)
    models={name:object() for name in ('1802','1812','1822')}
    return path,out,group,entry,models


def test_non_timing_comparison_preserves_failures_and_physics():
    a=dict(episode_wall_seconds=1.,calls=[dict(wall_ms=2,error='failure')],
           row=dict(cost=3,setup_ms=4),trace=[dict(command=[-5],gate_ms=6)])
    b=dict(episode_wall_seconds=7.,calls=[dict(wall_ms=8,error='failure')],
           row=dict(cost=3,setup_ms=9),trace=[dict(command=[-5],gate_ms=10)])
    assert recovery.non_timing(a)==recovery.non_timing(b)
    b['trace'][0]['command']=[2.5]
    assert recovery.non_timing(a)!=recovery.non_timing(b)
    assert recovery.non_timing(a)['calls'][0]['error']=='failure'


def test_partial_suffix_keeps_exact_blobs_and_runs_only_missing_methods(tmp_path,monkeypatch):
    path,out,group,entry,models=fixture(tmp_path,monkeypatch);ran=[]
    def run(plan,g,method,ms,env):
        ran.append(method);return dict(new=method),[dict(command=[0,0,0])]
    def audit(*args):return dict(verified=True)
    recovery.complete_partial(entry,{},group,models,None,'p','b',run,audit)
    assert ran==group['method_order'][2:]
    after=out/'blocks/00000/observations.sqlite'
    with readonly(path) as a,readonly(after) as b:
        assert a.execute('SELECT * FROM episodes ORDER BY position').fetchall()==b.execute(
            'SELECT * FROM episodes WHERE position<2 ORDER BY position').fetchall()
    done=json.loads((after.parent/'complete.json').read_text())
    assert done['actual_sumo_runs']==13 and len(done['reviews'])==13
    assert len(inventory(after)['cases'])==13
    assert recovery.io.digest(path)==entry['database_sha256']


def test_new_record_is_durable_before_failed_audit_and_source_is_immutable(tmp_path,monkeypatch):
    path,out,group,entry,models=fixture(tmp_path,monkeypatch)
    def run(*args):return dict(new=True),[]
    def audit(plan,g,method,record,trace,model):
        if record.get('new'):
            assert len(inventory(out/'blocks/00000/observations.sqlite')['cases'])==3
            raise ValueError('independent audit failed')
        return dict(verified=True)
    with pytest.raises(ValueError,match='independent audit failed'):
        recovery.complete_partial(entry,{},group,models,None,'p','b',run,audit)
    assert not (out/'blocks/00000/observations.sqlite.seal.json').exists()
    assert recovery.io.digest(path)==entry['database_sha256']


def test_changed_original_prefix_is_rejected_before_any_new_run(tmp_path,monkeypatch):
    _,_,group,entry,models=fixture(tmp_path,monkeypatch)
    entry['database_sha256']='changed'
    with pytest.raises(ValueError,match='changed'):
        recovery.complete_partial(entry,{},group,models,None,'p','b',
            lambda *args:pytest.fail('must not run'),lambda *args:{})


def test_recovery_keeps_original_scientific_plan_and_separate_output():
    assert recovery.OUT!=recovery.PRIOR
    assert recovery.io.PLAN==recovery.PRIOR/'plan.json'
    assert recovery.io.BINDING==recovery.PRIOR/'execution_binding/manifest.json'
    plan=recovery.io.read(recovery.io.PLAN)
    assert len(plan['groups'])==1728 and plan['budget']['actual_sumo_runs']==22464
    assert all(len(g['method_order'])==13 for g in plan['groups'])
