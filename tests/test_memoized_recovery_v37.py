import ast
import inspect

import numpy as np
import pytest

from coopgate import memoized_recovery_v37 as memo
from coopgate import recovery_forecast_v35 as original
from coopgate.actuator_prediction_roundoff_v31 import plant_arrays
from coopgate.actuator_recovery_v34 import class_bounds
from coopgate.disturbance_v9 import disturbance_tape
from coopgate.fast_v8 import parameters
from coopgate.marl_v12 import LEVELS
from coopgate.merge_v4 import MergeConfig
from coopgate.online_actuator_gate_v32 import reference_parameters
from coopgate.recovery_joint_gate_v35 import RecoveryJointGate, RULES

STATE = np.array([[-140.,12.],[-220.,10.],[-190.,12.]])
EXTRA = {'proposal_cache','proposal_cache_entries','proposal_cache_capacity','planning_counter_scope'}


def test_every_forecast_update_and_recovery_check_is_syntactically_unchanged():
    before = ast.parse(inspect.getsource(original.forecast_kernel.py_func))
    after = ast.parse(inspect.getsource(memo.forecast_kernel.py_func))
    fn = after.body[0]
    fn.args.args = [a for a in fn.args.args if a.arg not in ('cache','stats','cache_capacity')]
    for node in ast.walk(fn):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id == 'recovery_command':
            node.args = node.args[:-3]
    assert ast.dump(before,include_attributes=False) == ast.dump(after,include_attributes=False)
    assert memo.forecast_kernel.targetoptions['fastmath'] is False
    assert memo.MemoizedRecoveryGate.__init__ is RecoveryJointGate.__init__


def test_key_preserves_every_input_bit_global_clock_and_configuration():
    x,prev,c = STATE.copy(),np.zeros(3),parameters(MergeConfig())
    key = memo.proposal_key(x,prev,0,c)
    for arr in (x,prev,c):
        for index in range(arr.size):
            old = arr.flat[index]
            arr.flat[index] = np.nextafter(old,np.inf)
            assert memo.proposal_key(x,prev,0,c) != key
            arr.flat[index] = old
    assert memo.proposal_key(x,prev,1,c) != key
    prev[0] = -0.
    assert memo.proposal_key(x,prev,0,c) != key


def args(profile,mode,first,step=0):
    plant = reference_parameters(profile)
    gain,drag,rho = plant_arrays(plant,.1)
    return (STATE.copy(),np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3),step,
        disturbance_tape(37013,mode,.1,250),gain,drag,rho,class_bounds(profile),parameters(MergeConfig()),
        np.zeros(3),first)


def same(a,b):
    assert len(a) == len(b) == 22
    for left,right in zip(a,b,strict=True):
        np.testing.assert_equal(left,right)


@pytest.mark.parametrize('profile',('nominal','gain_drag','lag_mild','lag_strong'))
@pytest.mark.parametrize('mode',('clean','uniform','pulse'))
@pytest.mark.parametrize('first',(False,True))
def test_full_forecast_values_traces_errors_and_reference_work_are_exact(profile,mode,first):
    values = args(profile,mode,first)
    preserved = [v.copy() if isinstance(v,np.ndarray) else v for v in values]
    reference = original.forecast_kernel(*values,True)
    for capacity in (0,2,memo.CAPACITY):
        cache,stats = memo.make_cache()
        for _ in range(2):
            same(reference,memo.forecast_kernel(*values,cache,stats,capacity,True))
        assert stats[0] == stats[1]+stats[2]
        assert stats[2] == stats[3]+stats[4]
        assert stats[5] == 2*reference[5] and stats[6] <= stats[5]
        assert len(cache) <= capacity
        if capacity == 0:
            assert stats[1] == 0 and stats[7] > 0
        else:
            assert stats[1] > 0
    for actual,old in zip(values,preserved):
        np.testing.assert_equal(actual,old)


def test_no_trace_allocation_does_not_disable_the_planning_cache():
    values = args('lag_strong','clean',False)
    cache,stats = memo.make_cache()
    first = memo.forecast_kernel(*values,cache,stats,memo.CAPACITY,False)
    count = len(cache)
    assert count > 0
    second = memo.forecast_kernel(*values,cache,stats,memo.CAPACITY,False)
    same(first,second)
    assert len(cache) == count and stats[1] > 0 and stats[6] < stats[5]


@pytest.mark.parametrize('boundary',('unsafe','exhausted','completed','invalid_first'))
def test_boundary_errors_and_full_work_flags_match(boundary):
    values = list(args('lag_strong','uniform',False))
    if boundary == 'unsafe': values[0] = np.array([[0.,10.],[0.,10.],[-1.,10.]])
    if boundary == 'exhausted': values[5] = 50
    if boundary == 'completed': values[0] = np.array([[80.,0.],[90.,0.],[100.,0.]])
    if boundary == 'invalid_first': values[-2],values[-1] = np.full(3,-5.),True
    cache,stats = memo.make_cache()
    expected = original.forecast_kernel(*values,True)
    for _ in range(2): same(expected,memo.forecast_kernel(*values,cache,stats,memo.CAPACITY,True))
    if boundary == 'unsafe':
        assert stats[4] > 0 and len(cache) == 0 and stats[1] == 0


def test_actuator_interval_checks_are_recomputed_even_for_identical_cached_plans():
    c = parameters(MergeConfig())
    bounds = class_bounds('lag_strong')
    cache,stats = memo.make_cache()
    for mlo,mhi in ((np.zeros(3),np.zeros(3)),(np.full(3,-.5),np.full(3,.5)),
                    (np.full(3,-5.),np.full(3,2.5))):
        inputs = (STATE,np.zeros(3),0,mlo,mhi,bounds,c)
        actual = memo.recovery_command(*inputs,cache,stats,memo.CAPACITY)
        expected = original.recovery_command(*inputs)
        for a,b in zip(actual,expected,strict=True): np.testing.assert_equal(a,b)
    assert stats[1] == 2


class CoastingActor:
    def proposal(self,obs,mask):
        scores = np.broadcast_to(-np.abs(LEVELS),mask.shape).copy()
        scores[~mask] = -np.inf
        return scores.argmax(-1)


@pytest.mark.parametrize('rule',RULES)
def test_gate_keeps_all_decision_evidence_and_clears_cache_each_decision(rule):
    model = None if rule == 'baseline' else CoastingActor()
    baseline = RecoveryJointGate(model,'lag_strong',rule)
    accelerated = memo.MemoizedRecoveryGate(model,'lag_strong',rule)
    previous = np.zeros(3)
    prior_cache = None
    for _ in range(2):
        a,ia = baseline.select(STATE,previous)
        b,ib = accelerated.select(STATE,previous)
        np.testing.assert_array_equal(a,b)
        assert {k:v for k,v in ia.items() if k != 'gate_ms'} == {
            k:v for k,v in ib.items() if k != 'gate_ms' and k not in EXTRA}
        assert ib['proposal_cache']['computed_successful_transitions'] <= ib['completed_planning_model_transitions']
        if rule in ('nominal_performance','bank_performance'):
            assert prior_cache is not accelerated._plan_cache
            prior_cache = accelerated._plan_cache
        previous = a
