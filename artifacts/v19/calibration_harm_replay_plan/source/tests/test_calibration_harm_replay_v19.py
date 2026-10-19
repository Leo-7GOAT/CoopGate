import copy
import importlib.util
from pathlib import Path
import sys

import pytest

TOOLS=Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0,str(TOOLS))
spec=importlib.util.spec_from_file_location('diagnose_calibration_harm_v19',TOOLS/'diagnose_calibration_harm_v19.py')
diagnosis=importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnosis)


def row(index=0,method='baseline',mode='uniform'):
    return dict(initial_index=index,controller=method,disturbance=mode,
                per_vehicle_cost=[10.,10.,10.],per_vehicle_finish=[9.,9.,9.],
                completed=True,collision=False,aborted=False,contract_violations=0,learned_action_steps=0)


def test_every_harmful_model_is_kept_and_shared_baseline_is_replayed_once():
    baseline=row()
    a,b=row(method='portfolio0_1802'),row(method='portfolio0_1822')
    a['per_vehicle_cost'][0]+=.125
    b['per_vehicle_finish'][1]+=.5
    cases=diagnosis.select_cases([b,baseline,a])
    assert [c['controller'] for c in cases]==['baseline','portfolio0_1802','portfolio0_1822']


def test_zero_adoption_failure_is_not_censored_and_safe_improvement_is_not_selected():
    baseline=row()
    failed=row(method='portfolio0_1812')
    failed['completed']=False
    good=row(method='portfolio0_1822')
    good['per_vehicle_cost']=[9.,9.,9.]
    assert not failed['learned_action_steps']
    cases=diagnosis.select_cases([baseline,failed,good])
    assert [c['controller'] for c in cases]==['baseline','portfolio0_1812']


def test_improvement_of_team_average_cannot_hide_individual_cost_or_time_loss():
    base=row()
    cost=row(method='portfolio0_1802')
    cost['per_vehicle_cost']=[8.,10.,10.125]
    time=copy.deepcopy(cost)
    time['per_vehicle_cost']=[8.,9.,9.]
    time['per_vehicle_finish']=[8.,9.5,8.]
    assert diagnosis.harmful(cost,base) and diagnosis.harmful(time,base)


def test_missing_or_duplicate_pairing_is_rejected():
    with pytest.raises(KeyError):
        diagnosis.select_cases([row(method='portfolio0_1802')])
    with pytest.raises(ValueError,match='Duplicate'):
        diagnosis.select_cases([row(),row()])


def test_counterfactual_forecast_failure_remains_an_explicit_unavailable_diagnosis(monkeypatch):
    reference=row(mode='clean')
    reference.update(seed=7,initial=[[-80.,12.],[-85.,13.],[-120.,11.]])
    trace=[dict(command=[0.,0.,0.],x=[[-74.,12.],[-78.5,13.],[-114.5,11.]],disturbance=[[0.,0.,0.]]*5)]

    def fail_forecast(*_):
        raise RuntimeError('Counterfactual outside forecast contract')

    monkeypatch.setattr(diagnosis,'disturbed_suffix',fail_forecast)
    result=diagnosis.hindsight(reference,trace,reference)
    assert not result['available'] and result['step']==0
    assert result['reason']=='counterfactual_baseline_forecast_exception'
