"""Correct the nominal terminal set's post-goal command-memory condition.

The immutable first deployment adapter is retained with its original evidence.
This additive revision changes only terminal-set recognition, not dynamics,
action decoding, the author planner, policy weights, or physical evaluation.
"""
import numpy as np
import torch

from .dmps_nominal_model import NominalModelEnv, ObservableDMPS


class NominalModelEnvV2(NominalModelEnv):
    def stable(self):
        active=self.x[:,0]<self.c.goal
        # Exact clamped speeds avoid treating a tiny positive closing speed as
        # an infinite-horizon invariant. The clock is only a finite-search cap.
        return bool(self.safe() and np.all(self.x[active,1]==0.)
            and np.all(self.previous[active]<=0.)
            and np.all(self.x[~active,1]==self.c.vmax)
            and np.all(self.previous[~active]>=-self.c.jerk*self.c.decision_dt))


class ObservableDMPSV2(ObservableDMPS):
    def __init__(self,model,planner_class):
        self.model=model
        self.env=NominalModelEnvV2(np.array([[-140.,12.],[-220.,10.],[-190.,12.]]))
        self.planner=planner_class(self.env,.99,4,4,2,torch.device('cpu'),model.critic1,single_action=True)
