"""Candidate-search regressions: retain valid alternatives after a veto."""
import numpy as np
from pathlib import Path
from experiments.joint_admissibility import gate as g

BANK=Path(__file__).resolve().parents[1]/'artifacts/mappo_ctde_20261101/development/compact_coupled_bank_20261102.json'

class Actor:
    def proposal(self, observations, mask):
        return np.array([np.flatnonzero(row)[0] for row in mask],dtype=int)

def prediction(command):
    result=[0]*22
    result[0]=np.column_stack((np.full(3,100. if command is None else 99.),np.full(3,10.)))
    result[1]=np.full(3,10.)
    result[2]=True
    result[21]=True
    return result

def controller(monkeypatch):
    obj=g.JointAdmissibilityGate(Actor(),'nominal',BANK)
    monkeypatch.setattr(obj,'predict',lambda state,previous,mlo,mhi,command,path:prediction(command))
    return obj

def test_history_veto_continues_to_another_physically_certified_candidate(monkeypatch):
    obj=controller(monkeypatch);seen=[]
    def history(x,prev,mlo,mhi,command):
        seen.append(command.copy())
        passed=len(seen)>1
        return passed,dict(passed=passed,test_history_predicate=True)
    monkeypatch.setattr(obj,'_history_admissible',history)
    chosen,info=obj.select(np.array([[-140.,12.],[-220.,10.],[-190.,12.]]),np.zeros(3))
    assert len(seen)==2
    assert not np.array_equal(seen[0],seen[1])
    assert np.array_equal(chosen,seen[1]) and info['actual_learning_used']
    assert info['candidate_evidence'][0]['history_rejected']
    assert info['candidate_evidence'][1]['passed']
    assert info['certified'] and info['selected_gain_confidence_score']>=.0025
    assert all(np.all(np.asarray(p)<=1e-10) for p in info['selected_phase_differences'])

def test_all_history_vetoes_return_the_exact_common_baseline(monkeypatch):
    obj=controller(monkeypatch)
    monkeypatch.setattr(obj,'_history_admissible',lambda *args:(False,dict(passed=False)))
    chosen,info=obj.select(np.array([[-140.,12.],[-220.,10.],[-190.,12.]]),np.zeros(3))
    assert len(info['history_candidate_evidence'])>1
    assert np.array_equal(chosen,info['baseline_command'])
    assert not info['actual_learning_used'] and not obj.learning_used_ever
    assert info['certified']

def test_first_fully_admissible_candidate_is_preserved(monkeypatch):
    obj=controller(monkeypatch);seen=[]
    def history(*args):
        seen.append(args[-1].copy())
        return True,dict(passed=True)
    monkeypatch.setattr(obj,'_history_admissible',history)
    chosen,info=obj.select(np.array([[-140.,12.],[-220.,10.],[-190.,12.]]),np.zeros(3))
    assert len(seen)==1 and np.array_equal(chosen,seen[0])
    assert info['actual_learning_used'] and info['certified']

def test_strict_phase_veto_also_searches_remaining_candidates(monkeypatch):
    obj=controller(monkeypatch);blocked=[];history_seen=[]
    def predict(state,previous,mlo,mhi,command,path):
        result=prediction(command)
        if command is not None:
            if not blocked:blocked.append(command.copy())
            if np.array_equal(command,blocked[0]):result[1]+=0.001
        return result
    def history(*args):
        history_seen.append(args[-1].copy())
        return True,dict(passed=True)
    monkeypatch.setattr(obj,'predict',predict)
    monkeypatch.setattr(obj,'_history_admissible',history)
    chosen,info=obj.select(np.array([[-140.,12.],[-220.,10.],[-190.,12.]]),np.zeros(3))
    assert info['actual_learning_used'] and not np.array_equal(chosen,blocked[0])
    assert len(history_seen)==1 and np.array_equal(chosen,history_seen[0])
    assert info['phase_tolerance_seconds']==0.
    assert all(np.all(np.asarray(p)<=1e-10) for p in info['selected_phase_differences'])
