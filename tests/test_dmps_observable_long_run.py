from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from train_dmps_observable_long_run import GaussianExploration,STEPS,evaluate_all,evaluation_cases,training_cases


def test_longer_training_schedule_has_no_shared_evaluation_initials():
    train=training_cases();test=evaluation_cases()
    assert len(train)==STEPS+1 and len({c['seed'] for c in train})==STEPS+1
    assert {c['seed'] for c in train}.isdisjoint(c['seed'] for c in test)
    assert sum(16 for t in range(STEPS) if t>2048 and t%16==0)==30704


def test_exploration_stream_not_consumed_by_search_global_rng():
    first=GaussianExploration(805202);second=GaussianExploration(805202)
    for _ in range(3):
        np.random.rand(15)
        np.testing.assert_array_equal(first.gen(),second.gen())


def test_failed_condition_does_not_remove_later_scheduled_cases(tmp_path):
    cases=evaluation_cases()[:3]
    class Env:
        def __init__(self):self.calls=[];self.episodes=[];self.raw=None;self.cursor=0
        def finish_record(self):
            self.episodes.append(dict(case=self.case,failed=self.failure is not None))
    env=Env()
    class Planner:
        def reset(self):
            env.case=cases[env.cursor];env.cursor+=1;env.failure=None
            return np.zeros(10)
        def take_step(self,action):
            if env.cursor==2:raise RuntimeError('deliberate nominal-domain failure')
            env.calls.append(1)
            return [(np.zeros(10),-1.,True,{})],([action],0)
    result=evaluate_all(env,Planner(),lambda obs:torch.zeros(3),cases,tmp_path)
    assert [r['case'] for r in result]==cases and env.cursor==3
    assert [r['actual_decisions'] for r in result]==[1,0,1]
    assert [r['failure'] is not None for r in result]==[False,True,False]
    assert (tmp_path/'case_failure_001.json').exists()
