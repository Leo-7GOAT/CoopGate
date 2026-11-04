"""Actual SUMO failure-prefix tests;no formal samples or alpha used."""
from copy import deepcopy
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import mappo_top_two_formal_runtime as runtime
from mappo_top_two_formal_design import validation_cases
from mappo_top_two_joint_data import paired_cluster,episode_outcome
from coopgate.mappo_deployment import load_deployment_actor
from coopgate.memoized_recovery_v37 import MemoizedRecoveryGate


@pytest.fixture(scope='module')
def model():
    torch.set_num_threads(1)
    root=Path(__file__).resolve().parents[1]
    return load_deployment_actor(root/'artifacts/mappo_ctde_20261101/development/joint_4703/policy.pt')[0]


class FaultAt:
    def __init__(self,inner,when,error):self.inner=inner;self.when=when;self.error=error;self.calls=0
    def select(self,*args):
        if self.calls==self.when:raise self.error('Injected controller failure for validation only')
        self.calls+=1;return self.inner.select(*args)


@pytest.mark.parametrize('when,error',((0,RuntimeError),(2,TypeError)))
def test_actual_controller_exception_prefix_is_preserved_and_independently_replayed(model,when,error):
    case=validation_cases()[0]
    def factory(method,m,c):return FaultAt(runtime.make_controller(method,m,c),when,error)
    with runtime.MatchedShiftedMergeEnv(plant='nominal') as env:
        record=runtime.execute_episode(case,'4703',model,env,controller_factory=factory)
    row=record['row'];assert row['aborted'] and not row['completed'] and len(record['trace'])==when
    assert len(record['calls'])==when+1 and record['calls'][-1]['error']['type']==error.__name__
    proof=runtime.audit_record(case,record,model,reference_factory=factory)
    assert proof['recorded_failure'] and proof['genuine_failures_retained']
    assert episode_outcome(row)['failed']


def test_sumoreported_collision_is_retained_instead_of_dropped():
    case=validation_cases()[0]
    with runtime.MatchedShiftedMergeEnv(plant='nominal') as env:
        record=runtime.execute_episode(case,'baseline',None,env)
    # Valid external collision flag,with completed flag updated consistently.
    # The independent geometric trace may be safe;the SUMO failure still counts.
    record['row']['collision']=True;record['row']['completed']=False
    assert runtime.audit_record(case,record,None)['recorded_failure']




def test_rejected_independent_coin_never_queries_actor():
    class Never:
        def proposal(self,*args):raise AssertionError('Rejected branch queried actor')
    case=validation_cases()[0];case['controller_coins']['4703']=False
    controller=runtime.make_controller('thinned_4703',Never(),case)
    _,info=controller.select(np.array([[-140.,12.],[-220.,10.],[-190.,12.]]),np.zeros(3))
    assert info['rule']=='baseline' and not info['actual_learning_used'] and not info['coin_is_actual_adoption']
