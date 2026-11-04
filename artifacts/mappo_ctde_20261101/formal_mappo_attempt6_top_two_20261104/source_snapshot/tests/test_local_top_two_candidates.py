"""Local ranked alternatives retain the existing physical action contract."""
from pathlib import Path

import numpy as np
import pytest
import torch
from coopgate.actuator_recovery_v34 import stopping_certificate
from experiments.top_two_candidates.gate import local_top_two,TopTwoHistoryGate,TopTwoReserveGate

BANK=Path(__file__).resolve().parents[1]/'artifacts/mappo_ctde_20261101/development/compact_coupled_bank_20261102.json'


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(()))
        logits=torch.zeros((3,13));logits[:,12]=3.;logits[:,11]=2.
        self.register_buffer('logits',logits)
    def features(self,observations):return observations
    def actor(self,features):return self.logits
    def proposal(self,observations,mask):
        return self.logits.masked_fill(~torch.as_tensor(mask),-1e9).argmax(-1).numpy()
    def value(self,*args):raise AssertionError('Deployment must not call the central critic')


def test_top_two_respects_support_and_ascending_tie_order():
    model=Actor();model.logits.zero_();mask=np.zeros((3,13),bool)
    mask[0,[4,7]]=True;mask[1,[5]]=True;mask[2,[6,8,12]]=True
    indices,_=local_top_two(model,np.zeros((3,18)),mask)
    np.testing.assert_array_equal(indices,[[4,7],[5,5],[6,8]])


def test_empty_local_support_is_rejected():
    with pytest.raises(ValueError):local_top_two(Actor(),np.zeros((3,18)),np.zeros((3,13),bool))


@pytest.mark.parametrize('cls',(TopTwoHistoryGate,TopTwoReserveGate))
def test_four_extra_candidates_are_distinct_and_physically_certified(cls):
    obj=cls(Actor(),'nominal',BANK);x=np.array([[-140.,12.],[-220.,10.],[-190.,12.]])
    previous=np.zeros(3);baseline=np.full(3,2.5);zero=np.zeros(3)
    candidates,count=obj._candidate_union(x,previous,baseline,baseline,zero,zero,obj.bounds,obj.packed)
    assert len(candidates)==count==4
    assert [c[0] for c in candidates]==[100,101,102,103]
    assert len({c[2].tobytes() for c in candidates})==4
    for _,_,command,_ in candidates:
        assert stopping_certificate(x,previous,command,zero,zero,obj.bounds,obj.packed)[0]
        assert not np.array_equal(command,baseline)


def test_cleared_vehicles_keep_exact_baseline_commands():
    obj=TopTwoHistoryGate(Actor(),'nominal',BANK)
    x=np.array([[100.,12.],[-220.,10.],[-190.,12.]])
    previous=np.zeros(3);baseline=np.full(3,2.5);zero=np.zeros(3)
    candidates,_=obj._candidate_union(x,previous,baseline,baseline,zero,zero,obj.bounds,obj.packed)
    assert len(candidates)<=3
    assert all(command[0]==baseline[0] for _,_,command,_ in candidates)


@pytest.mark.parametrize('cls',(TopTwoHistoryGate,TopTwoReserveGate))
def test_real_select_advances_baseline_once_and_keeps_same_gain_floor(cls):
    obj=cls(Actor(),'nominal',BANK)
    command,info=obj.select(np.array([[-140.,12.],[-220.,10.],[-190.,12.]]),np.zeros(3))
    assert obj.step==obj.baseline.step==1
    assert info['certified'] and info['gain_confidence_floor']==.0025
    assert len(info['top_two_candidates']['extra_candidates'])<=4
    assert info['deployed_critic_used'] is False
    assert info['logits_are_feasibility_probabilities'] is False


@pytest.mark.parametrize('mutation',('mask','code','rank'))
def test_ranked_evidence_rejects_corrupt_proposal_provenance(mutation):
    from experiments.top_two_candidates.audit import validate
    obj=TopTwoHistoryGate(Actor(),'nominal',BANK)
    command,info=obj.select(np.array([[-140.,12.],[-220.,10.],[-190.,12.]]),np.zeros(3))
    info['command']=command.tolist();validate(info)
    if mutation=='mask':info['top_two_candidates']['indices'][0][1]=0
    elif mutation=='code':info['top_two_candidates']['extra_candidates'][0]['composition']=404
    else:info['top_two_candidates']['raw_logits'][0]=[2.,3.]
    with pytest.raises(ValueError):validate(info)
