from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_fast_repair_v20 import admitted_state, configured
from coopgate.arrival_phase_v20 import arrival_phase_suffix
from coopgate.cached_portfolio_v19 import new_command_cache
from coopgate.phase_repair_v21 import PhaseRepairGate, cached_arrival_phase
from coopgate.repair_portfolio_v20 import RepairPortfolioGate


@pytest.mark.parametrize('seed', [1802,1812,1822])
def test_disabled_and_half_period_reference_preserve_real_admitted_command(seed):
    state, previous, step, item = admitted_state(seed)
    reference = configured(RepairPortfolioGate,step,item)
    expected, info = reference.select(state,previous)
    for tolerance in (None,.5):
        gate = configured(PhaseRepairGate,step,item,phase_tolerance=tolerance)
        actual, checked = gate.select(state,previous)
        np.testing.assert_array_equal(actual,expected)
        for key in ('composition','repair_weight','candidate_count','original_candidate_count',
                    'nominal_candidate_count','selected_bank_differences','forecast_errors','generation_errors'):
            assert checked[key] == info[key]
        if tolerance is not None:
            phases = np.asarray(checked['selected_phase_differences'])
            assert phases.shape == (33,3) and np.max(phases)<=tolerance+1e-10


@pytest.mark.parametrize('tolerance', [.05,.1])
def test_every_accepted_tight_action_passes_original_and_extra_conditions(tolerance):
    for seed in (1802,1812,1822):
        state, previous, step, item = admitted_state(seed)
        gate = configured(PhaseRepairGate,step,item,phase_tolerance=tolerance)
        actual, info = gate.select(state,previous)
        if info['portfolio_joint_action_admitted']:
            original=np.asarray(info['selected_bank_differences'])
            phase=np.asarray(info['selected_phase_differences'])
            assert original.shape == (33,3,2) and np.max(original)<=1e-10
            assert phase.shape == (33,3) and np.max(phase)<=tolerance+1e-10
        else:
            np.testing.assert_array_equal(actual,info['baseline_command'])


def test_cached_event_times_match_independent_forecaster_with_and_without_reuse():
    state, previous, step, item = admitted_state(1802)
    gate=configured(PhaseRepairGate,step,item)
    command=np.asarray(item['command'])
    cache=new_command_cache()
    for path in (0,17,25,32):
        for use_first in (False,True):
            expected=arrival_phase_suffix(state,previous,step,gate.tapes[path],gate.packed,command,use_first)[1]
            actual,_,_=cached_arrival_phase(state,previous,step,gate.tapes[path],gate.packed,command,use_first,cache)
            np.testing.assert_array_equal(actual,expected)
            repeat,_,work=cached_arrival_phase(state,previous,step,gate.tapes[path],gate.packed,command,use_first,cache)
            np.testing.assert_array_equal(repeat,expected)
            assert work==0


def test_unavailable_phase_forecast_rejects_learning_instead_of_bypassing_condition(monkeypatch):
    state,previous,step,item=admitted_state(1802)
    gate=configured(PhaseRepairGate,step,item)
    def unavailable(*args):
        raise RuntimeError('Phase unavailable')
    monkeypatch.setattr(gate,'_phase',unavailable)
    actual,info=gate.select(state,previous)
    assert info['forecast_errors'] and not info['portfolio_joint_action_admitted']
    np.testing.assert_array_equal(actual,info['baseline_command'])


def test_one_candidate_failure_does_not_poison_the_common_baseline_cache(monkeypatch):
    state,previous,step,item=admitted_state(1802)
    gate=configured(PhaseRepairGate,step,item)
    failed=[]
    def controlled_future(x,prev,command,path,base,cache):
        if path==1 and not failed:
            failed.append(command.copy())
            raise RuntimeError('Candidate-only failure')
        value=base.copy()
        value[:,0]-=.1
        return value,0,0,False,0
    monkeypatch.setattr(gate,'_future',controlled_future)
    monkeypatch.setattr(gate,'_phase',lambda *args:(np.zeros(3),0,0))
    command,info=gate.select(state,previous)
    assert info['candidate_count']>1 and len(failed)==1
    assert info['portfolio_joint_action_admitted']
    assert not np.array_equal(command,failed[0])
    assert len(info['selected_bank_differences'])==len(info['selected_phase_differences'])==33
    assert info['forecast_errors'][0]['stage']=='candidate_cost'


@pytest.mark.parametrize('bad',[-.1,.5001,np.nan,np.inf,True])
def test_invalid_time_tolerance_is_rejected(bad):
    with pytest.raises(ValueError,match='tolerance'):
        PhaseRepairGate(None,bad)
