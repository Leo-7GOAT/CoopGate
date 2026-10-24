import ast
from pathlib import Path
from types import FunctionType

import numpy as np
import pytest

from coopgate import buffered_baseline_v25 as baseline
from coopgate import memoized_recovery_v37 as memo
from coopgate import zero_score_projection as fast


def functions(path):
    return {n.name:n for n in ast.parse(Path(path).read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef)}


def test_downstream_function_bodies_remain_identical_to_frozen_sources():
    actual=functions(fast.__file__)
    for module,names in [(baseline,('team_rule_tail','deduplicated_joint_rollout')),
                         (memo,('cached_proposal','recovery_command','forecast_kernel'))]:
        reference=functions(module.__file__)
        for name in names:
            assert ast.dump(ast.Module(body=actual[name].body,type_ignores=[]))==ast.dump(ast.Module(body=reference[name].body,type_ignores=[]))


@pytest.mark.parametrize('accept_at',[1,3,28])
@pytest.mark.parametrize('kind',['ordinary','equal','signed_zero','underflow','backup_underflow','nan','infinity'])
def test_shortcut_preserves_stable_vertex_and_certificate_visit_order(kind,accept_at):
    c=memo.MemoizedRecoveryGate(None,'nominal','baseline').packed.copy()
    previous=np.array([0.,1.25,-2.5]);proposed=np.array([2.5,0.,-1.25])
    if kind=='equal': previous[:]=proposed[:]=2.5
    if kind=='signed_zero': previous[:]=-0.;proposed[:]=0.
    if kind=='underflow': previous[:]=0.;proposed[:]=1e-200
    if kind=='backup_underflow': previous[:]=proposed[:]=0.;c[6]=1e-200
    if kind=='nan': proposed[0]=np.nan
    if kind=='infinity': proposed[0]=np.inf
    state=np.array([[-140.,12.],[-220.,10.],[-190.,12.]])
    runs=[]
    for original in (baseline.project_joint.py_func,fast.project_joint.py_func):
        visited=[]
        def certificate(x,p,u,*args):
            visited.append(u.tobytes())
            return len(visited)==accept_at
        function=FunctionType(original.__code__,{**original.__globals__,'robust_certificate':certificate})
        with np.errstate(invalid='ignore',over='ignore',under='ignore'):
            result=function(state,previous,proposed,c,.5)
        runs.append((result,visited))
    (a,left),(b,right)=runs
    assert a[0].tobytes()==b[0].tobytes() and a[1:]==b[1:] and left==right


def test_physical_projection_matches_on_fixed_continuous_random_inputs():
    rng=np.random.default_rng(804710)
    c=memo.MemoizedRecoveryGate(None,'nominal','baseline').packed
    for _ in range(256):
        x=np.column_stack((rng.uniform(-160,20,3),rng.uniform(0,20,3)))
        x[2,0]=x[0,0]-rng.uniform(7,50)
        previous=rng.uniform(-5,2.5,3);proposed=rng.uniform(-5,2.5,3)
        a=baseline.project_joint(x,previous,proposed,c,.5)
        b=fast.project_joint(x,previous,proposed,c,.5)
        assert a[0].tobytes()==b[0].tobytes() and a[1:]==b[1:]


@pytest.mark.parametrize('kind',['equal','signed_zero','underflow','backup_underflow'])
def test_compiled_projection_preserves_float_ties(kind):
    c=memo.MemoizedRecoveryGate(None,'nominal','baseline').packed.copy()
    previous=np.zeros(3);proposed=np.zeros(3)
    if kind=='equal': previous[:]=proposed[:]=2.5
    if kind=='signed_zero': previous[:]=-0.
    if kind=='underflow': proposed[:]=1e-200
    if kind=='backup_underflow': c[6]=1e-200
    state=np.array([[-140.,12.],[-220.,10.],[-190.,12.]])
    a=baseline.project_joint(state,previous,proposed,c,.5)
    b=fast.project_joint(state,previous,proposed,c,.5)
    assert a[0].tobytes()==b[0].tobytes() and a[1:]==b[1:]


def test_fixed_schedule_balances_each_setting_and_rejects_changed_evidence():
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
    import benchmark_zero_score_projection as study
    items=study.cases()
    assert len(items)==24
    assert len({(c['model'],c['profile'],c['rule']) for c in items})==24
    for item in items:
        keys=item['schedule']
        assert len(keys)==4 and len({tuple(k) for k in keys})==4
        assert [k[2] for k in keys[:2]]==[k[2] for k in keys[2:]][::-1]
    study.require_equal([{'proof':1}]*4)
    with pytest.raises(ValueError): study.require_equal([{'proof':1}]*3)
    with pytest.raises(ValueError): study.require_equal([{'proof':1}]*3+[{'proof':2}])


@pytest.mark.parametrize('profile',['nominal','gain_drag','lag_mild','lag_strong'])
@pytest.mark.parametrize('forced',[False,True])
@pytest.mark.parametrize('path',[0,32])
def test_full_forecasts_and_planning_statistics_are_bit_exact(profile,forced,path):
    gate=memo.MemoizedRecoveryGate(None,profile,'baseline')
    args=(np.array([[-140.,12.],[-220.,10.],[-190.,12.]]),np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3),
          0,gate.tapes[path],gate.gain,gate.drag,gate.retention,gate.bounds,gate.packed,np.zeros(3),forced)
    saved=[v.copy() if isinstance(v,np.ndarray) else v for v in args]
    ca,sa=memo.make_cache();cb,sb=memo.make_cache()
    a=memo.forecast_kernel(*args,ca,sa,memo.CAPACITY,True)
    b=fast.forecast_kernel(*args,cb,sb,memo.CAPACITY,True)
    for old,new in zip(a,b):
        if isinstance(old,np.ndarray): assert old.dtype==new.dtype and old.shape==new.shape and old.tobytes()==new.tobytes()
        else: assert old==new
    assert len(a)==len(b)==22
    np.testing.assert_array_equal(sa,sb)
    for old,new in zip(args,saved):
        if isinstance(old,np.ndarray): assert old.tobytes()==new.tobytes()


@pytest.mark.parametrize('invalid',['forced','recovery'])
def test_invalid_first_or_recovery_keeps_original_error(invalid):
    gate=memo.MemoizedRecoveryGate(None,'lag_strong','baseline')
    state=np.array([[-140.,12.],[-220.,10.],[-190.,12.]]) if invalid=='forced' else np.array([[-1.,12.],[-1.,12.],[-2.,12.]])
    args=(state,np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3),0,gate.tapes[0],gate.gain,gate.drag,
          gate.retention,gate.bounds,gate.packed,np.full(3,10.),invalid=='forced')
    ca,sa=memo.make_cache();cb,sb=memo.make_cache()
    a=memo.forecast_kernel(*args,ca,sa,0,True);b=fast.forecast_kernel(*args,cb,sb,0,True)
    assert a[3]==b[3]==(3 if invalid=='forced' else 2)
    for old,new in zip(a,b):
        if isinstance(old,np.ndarray): assert old.tobytes()==new.tobytes()
        else: assert old==new
    np.testing.assert_array_equal(sa,sb)
