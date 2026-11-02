"""Certificate equivalence protects the unchanged physical decision rule."""
import ast
from pathlib import Path
import numpy as np
import pytest
from coopgate import buffered_baseline_v25 as reference
from coopgate import memoized_recovery_v37 as memo
from coopgate import scalar_nominal_recovery as scalar
from coopgate.fast_v8 import parameters
from coopgate.merge_v4 import MergeConfig


def test_downstream_ast_is_unchanged():
    def funcs(module):
        return {n.name:n for n in ast.parse(Path(module.__file__).read_text()).body if isinstance(n,ast.FunctionDef)}
    actual=funcs(scalar)
    for module,names in ((reference,('team_rule_tail','deduplicated_joint_rollout')),
                        (memo,('cached_proposal','recovery_command','forecast_kernel'))):
        source=funcs(module)
        for name in names:assert ast.dump(actual[name])==ast.dump(source[name])


def test_continuous_and_boundary_certificate_equivalence():
    c=parameters(MergeConfig());rng=np.random.default_rng(830211)
    cases=[]
    for _ in range(4096):
        state=np.column_stack((rng.uniform(-220,80,3),rng.uniform(0,20,3)))
        state[2,0]=state[0,0]-rng.uniform(6,70)
        previous=rng.uniform(-5,2.5,3);command=rng.uniform(-5,2.5,3)
        cases.append((state,previous,command))
    for position in (np.nextafter(c[8],-np.inf),c[8],np.nextafter(c[8],np.inf),c[9],c[11]):
        for speed in (0.,np.nextafter(0.,1.),10.,20.):
            for action in (-5.,-2.5,-0.,0.,2.5):
                cases.append((np.array([[position,speed],[position-40,speed],[position-80,speed]]),np.full(3,action),np.full(3,action)))
    for state,previous,command in cases:
        copies=[x.copy() for x in (state,previous,command,c)]
        assert scalar.robust_certificate(state,previous,command,c,.5)==reference.robust_certificate(state,previous,command,c,.5)
        for a,b in zip((state,previous,command,c),copies):assert a.tobytes()==b.tobytes()


@pytest.mark.parametrize('state',[
    [[-140.,12.],[-220.,10.],[-190.,12.]],
    [[-77.,14.],[-75.,13.],[-113.,15.]],
    [[80.,20.],[70.,20.],[62.,20.]],
    [[-28.,1.],[-60.,0.],[-95.,0.]]])
def test_full_planner_returns_and_errors_are_identical(state):
    c=parameters(MergeConfig());x=np.array(state);previous=np.zeros(3);outputs=[]
    for function in (reference.deduplicated_joint_rollout,scalar.deduplicated_joint_rollout):
        try:
            value=function(x,previous,0,c)
            outputs.append((value[0].tobytes(),value[1:]))
        except RuntimeError as exc:outputs.append((type(exc).__name__,str(exc)))
    assert outputs[0]==outputs[1]


def test_full_forecast_all_outputs_and_cache_statistics_match():
    state=np.array([[-140.,12.],[-220.,10.],[-190.,12.]])
    gate=memo.MemoizedRecoveryGate(None,'lag_strong','baseline');outputs=[]
    for function in (memo.forecast_kernel,scalar.forecast_kernel):
        cache,stats=memo.make_cache()
        values=function(state,np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3),0,
            gate.tapes[0],gate.gain,gate.drag,gate.retention,gate.bounds,gate.packed,
            np.zeros(3),False,cache,stats,memo.CAPACITY,True)
        outputs.append((values,stats.copy()))
    for a,b in zip(outputs[0][0],outputs[1][0]):
        if isinstance(a,np.ndarray):assert a.tobytes()==b.tobytes()
        else:assert a==b
    np.testing.assert_array_equal(outputs[0][1],outputs[1][1])
