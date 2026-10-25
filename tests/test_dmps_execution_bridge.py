from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from coopgate.dmps_execution_bridge import ModelCommitPlanner
from coopgate.dmps_nominal_model_v2 import NominalModelEnvV2

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import load_original_dmps as author


class Execution:
    def __init__(self):self.commits=0;self.generation=0
    def reset(self):
        self.generation+=1
        self.state=np.array([[-34.,8.],[-32.,8.],[-70.,8.]])
        self.previous=np.zeros(3);self.step=0
        return self.obs()
    def obs(self):return NominalModelEnvV2(self.state,self.previous,self.step).obs()
    def observables(self):return self.state.copy(),self.previous.copy(),self.step
    def execute(self,action,context):
        self.commits+=1
        # Deliberately disagree with the nominal prediction to expose leakage.
        self.state[:,0]-=.125;self.step+=1
        return self.obs(),-7.25,False,dict(actual=True)
    def get_state(self):raise AssertionError('Physical state cloning is forbidden')


class ForecastPlanner:
    def __init__(self,env,*args,**kwargs):self.env=env
    def take_step(self,action):
        root=self.env.get_state()
        self.env.step([-1,-1,-1]);self.env.set_state(root)
        return [self.env.step([-1,-1,-1])],([np.full(3,-1.)],1)


def test_only_real_transition_returned_after_search_and_latest_actual_state_resynchronized():
    real=Execution();planner=ModelCommitPlanner(real,ForecastPlanner,None)
    initial=planner.reset()
    for expected in (1,2):
        data,(actions,code)=planner.take_step(np.ones(3))
        assert real.commits==expected and code==1 and len(actions)==1
        assert data[0][1:]==(-7.25,False,dict(actual=True))
        np.testing.assert_array_equal(data[0][0],real.obs())
        np.testing.assert_array_equal(planner.forecaster.obs,real.obs())
        assert not np.array_equal(planner.model_env.obs(),real.obs())
    assert not np.array_equal(initial,planner.obs)


def test_reset_resynchronizes_actual_episode_and_forecaster_observation():
    real=Execution();planner=ModelCommitPlanner(real,ForecastPlanner,None)
    expected=planner.reset();planner.take_step(np.ones(3));actual=planner.reset()
    assert real.generation==2
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(planner.forecaster.obs,expected)


def test_original_author_search_rejects_risky_action_without_real_branch_execution():
    source=Path(__file__).resolve().parents[2]/'reference_sources/dmps-f37df2d'
    if not source.exists():pytest.skip('Pinned external author checkout not available')
    module,_=author.load(source)
    def critic(obs,acts):return torch.zeros((len(obs),1),device=obs.device)
    real=Execution();planner=ModelCommitPlanner(real,module.MCTSPlanner,critic)
    planner.reset();np.random.seed(805100)
    data,(_,code)=planner.take_step(np.ones(3))
    assert code==1 and real.commits==1
    assert planner.model_env.model_decisions>1
    np.testing.assert_array_equal(data[0][0],real.obs())
