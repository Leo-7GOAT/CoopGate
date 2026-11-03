"""Re-run pinned nominal MCTS and loaded actor on sealed physical observations.

This executes no SUMO and clones no physical process. A selected command must
match before the recorded real transition is returned to the planner.
"""
import numpy as np
import torch

from .dmps_execution_bridge import ModelCommitPlanner
from .dmps_nominal_model_v2 import NominalModelEnvV2


class ReplayExecution:
    def __init__(self,record,calls):
        self.record=record;self.calls=calls;self.index=0
        assert len(record['decisions'])==len(calls)

    def reset(self):
        self.index=0
        return (np.asarray(self.calls[0]['observation'],dtype=np.float32) if self.calls else
            NominalModelEnvV2(np.asarray(self.record['initial'])).obs())

    def observables(self):
        if self.index<len(self.calls):
            decision=self.record['decisions'][self.index]
            return np.asarray(decision['observation_state']),np.asarray(decision['previous_command']),self.index
        if not self.calls:return np.asarray(self.record['initial']),np.zeros(3),0
        last=self.record['decisions'][-1]['trace']
        return np.asarray(last['x']),np.asarray(last['command']),self.index

    def execute(self,action,context):
        assert self.index<len(self.calls),'Replay attempted an unrecorded physical command'
        call=self.calls[self.index];decision=self.record['decisions'][self.index]
        np.testing.assert_array_equal(context['proposal'],call['proposed'])
        np.testing.assert_array_equal(action,call['executed'])
        assert context['intervention']==call['intervention']
        assert context['nominal_model_decisions']==decision['planner']['nominal_model_decisions']
        np.testing.assert_allclose(context['predicted_state'],decision['planner']['predicted_state'],atol=1e-12,rtol=0)
        self.index+=1
        return np.asarray(call['next_observation'],dtype=np.float32),call['reward'],call['done'],{}


def replay_episode(record,calls,model,planner_class,search_seed,expected_failure):
    saved=np.random.get_state();execution=ReplayExecution(record,calls)
    planner=ModelCommitPlanner(execution,planner_class,model.critic1)
    attempted=0;failure=None
    try:
        np.random.seed(search_seed)
        try:
            obs=planner.reset()
            for _ in range(50):
                with torch.no_grad():proposal=model.actor(torch.as_tensor(obs,dtype=torch.float32)).numpy()
                attempted+=1
                data,_=planner.take_step(proposal);obs,_,done,_=data[0]
                if done:break
        except (RuntimeError,ValueError) as error:
            failure=dict(type=type(error).__name__,message=str(error))
        assert execution.index==len(calls)
        expected=None if expected_failure is None else {k:expected_failure[k] for k in ('type','message')}
        assert failure==expected,'Loaded policy/planner failure differs from sealed evaluation'
        return dict(replayed_physical_commands=execution.index,attempted_actor_decisions=attempted,
            policy_proposals_and_nominal_mcts_choices_match=True,failure_matches=True,new_sumo_executions=0)
    finally:
        # Restore only outside the whole case; never rewind MCTS search RNG at
        # individual state restores inside the original search.
        np.random.set_state(saved)
