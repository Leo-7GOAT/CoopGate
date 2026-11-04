"""Original DMPS search in a nominal model, with exactly one real commit.

The original MCTSPlanner owns only NominalModelEnvV2. The boundary adapter sends
its selected action to an execution object and returns the REAL transition to
the unchanged author trainer. Search observations never enter its replay.
"""
import time

import numpy as np
import torch

from .dmps_nominal_model_v2 import NominalModelEnvV2


class ModelCommitPlanner:
    def __init__(self,execution,planner_class,critic,gamma=.99,num_iters=4,branch_factor=4,path_len=2):
        self.execution=execution
        self.model_env=NominalModelEnvV2(np.array([[-140.,12.],[-220.,10.],[-190.,12.]]))
        self.forecaster=planner_class(self.model_env,gamma,num_iters,branch_factor,path_len,
            torch.device('cpu'),critic,single_action=True)
        self.obs=None

    def reset(self):
        self.obs=self.execution.reset()
        x,previous,step=self.execution.observables()
        self.model_env.sync(x,previous,step)
        np.testing.assert_array_equal(self.obs,self.model_env.obs())
        self.forecaster.obs=self.obs.copy()
        return self.obs.copy()

    def take_step(self,proposal):
        x,previous,step=self.execution.observables()
        self.model_env.sync(x,previous,step)
        if not self.model_env.safe():raise RuntimeError('Observed geometry outside nominal DMPS domain')
        np.testing.assert_array_equal(self.obs,self.model_env.obs())
        self.forecaster.obs=self.obs.copy();work=self.model_env.model_decisions
        start=time.perf_counter()
        predicted,(actions,code)=self.forecaster.take_step(proposal)
        elapsed=(time.perf_counter()-start)*1000
        if len(predicted)!=1 or len(actions)!=1 or self.model_env.index!=step+1:
            raise RuntimeError('Exactly one original nominal choice required before real commit')
        if not self.model_env.safe():raise RuntimeError('Original choice is outside nominal geometric safety')
        # Re-read only observable copies to prove search did not advance the
        # execution object. No hidden plant state or forecast tape is requested.
        after_x,after_previous,after_step=self.execution.observables()
        np.testing.assert_array_equal(x,after_x);np.testing.assert_array_equal(previous,after_previous)
        if step!=after_step:raise RuntimeError('Search advanced the physical decision index')
        context=dict(proposal=np.asarray(proposal).tolist(),selected_action=np.asarray(actions[0]).tolist(),
            intervention=int(code),predicted_state=self.model_env.x.tolist(),
            nominal_model_decisions=self.model_env.model_decisions-work,wall_ms=elapsed,
            prediction_contract='Nominal model recovery only; actual transition committed after search')
        actual=self.execution.execute(np.asarray(actions[0]).copy(),context)
        self.obs=np.asarray(actual[0]).copy();self.forecaster.obs=self.obs.copy()
        return [actual],([np.asarray(actions[0]).copy()],code)
