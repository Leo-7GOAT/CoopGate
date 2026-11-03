from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import run_history_holdout_ablation as runner
import mappo_completed_attempt5 as completed
from coopgate.compact_coupled_gate import CompactCoupledGate
from coopgate.strict_phase_only_gate import StrictPhaseOnlyGate


@pytest.mark.parametrize('variant,expected',[('compact',CompactCoupledGate),('phase',StrictPhaseOnlyGate)])
def test_prespecified_factory_keeps_distinct_comparator_mechanisms(monkeypatch,variant,expected):
    monkeypatch.setattr(runner,'VARIANT',variant)
    gate=runner.old_factory('4704',object(),dict(plant='lag_strong'))
    assert type(gate) is expected and gate.gain_floor==.0025
    with pytest.raises(ValueError):runner.old_factory('unknown',None,dict(plant='lag_strong'))


@pytest.mark.parametrize('code',[None,1])
def test_live_or_failed_primary_is_refused_before_reading_any_effect(monkeypatch,tmp_path,code):
    reads=[]
    def read(path):
        reads.append(Path(path).name)
        assert Path(path).name=='orchestrator_native_receipt.json'
        return dict(actual_native_exit_code=code)
    monkeypatch.setattr(completed.e,'read',read)
    with pytest.raises(ValueError,match='owner'):
        completed.complete_evidence(tmp_path)
    assert reads==['orchestrator_native_receipt.json']


def test_wrong_comparator_plan_is_rejected_before_primary_access(monkeypatch,tmp_path):
    monkeypatch.setattr(runner,'OUT',tmp_path);monkeypatch.setattr(runner,'VARIANT','phase')
    monkeypatch.setattr(runner.e,'read',lambda path:dict(variant='compact'))
    monkeypatch.setattr(runner.e,'bound',lambda *a:pytest.fail('Must reject variant before primary access'))
    with pytest.raises(ValueError,match='variant'):runner.guard()
