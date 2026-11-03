"""Stepwise joint action admission against the SAME full baseline continuation.

Exact deterministic nominal policy improvement holds component-wise by
telescoping. It is NOT a probabilistic or bounded-disturbance performance
certificate. The physical action certificate is still robust and separate.
This is an established rollout/policy-improvement construction, not new theory.
"""
import time
import numpy as np
from numba import njit
from .joint_planner_v15 import joint_rollout
from .cooperative_v15 import JointPolicyGate
from .fast_v8 import parameters, advance_one
from .merge_v4 import MergeConfig
from .robust_v9 import validate, robust_certificate


@njit(cache=True, fastmath=False)
def baseline_suffix(state, previous, step, packed):
    x=state.copy();prev=previous.copy();values=np.zeros((3,2))
    first=previous.copy();transitions=0;internal=0
    for k in range(step,round(packed[2]/packed[1])):
        active=x[:,0]<packed[11]
        if not np.any(active):break
        command,_,_,count=joint_rollout(x,prev,k,packed);internal+=count
        if k==step:first=command.copy()
        for i in range(3):
            if active[i]:
                values[i,0]+=packed[1]+.05*abs(command[i]-prev[i])
                values[i,1]+=packed[1]
        if not advance_one(x,command,packed):raise RuntimeError('Unsafe nominal baseline suffix')
        prev=command;transitions+=1
    return values,first,transitions,internal


@njit(cache=True, fastmath=False)
def action_then_baseline(state, previous, step, command, packed):
    if not robust_certificate(state,previous,command,packed,.5):
        raise ValueError('Uncertified first command')
    active=state[:,0]<packed[11]
    following=state.copy()
    if not advance_one(following,command,packed):raise RuntimeError('Unsafe nominal first transition')
    values,_,transitions,internal=baseline_suffix(following,command,step+1,packed)
    for i in range(3):
        if active[i]:
            values[i,0]+=packed[1]+.05*abs(command[i]-previous[i])
            values[i,1]+=packed[1]
    return values,transitions+1,internal


class JointAdvantageGate:
    def __init__(self, model):
        self.c=MergeConfig();self.packed=parameters(self.c)
        self.primary=JointPolicyGate(model);self.step=0

    def select(self,state,previous,unused=None):
        started=time.perf_counter();x,prev,_=validate(state,previous,self.c,.5)
        if self.step>=round(self.c.horizon/self.c.decision_dt):
            raise ValueError('Controller finite horizon exhausted')
        base,baseline_command,transitions,internal=baseline_suffix(x,prev,self.step,self.packed)
        self.primary.step=self.step
        proposed,_=self.primary.select(x,prev)
        difference=np.zeros((3,2));accepted=False
        if not np.array_equal(proposed,baseline_command):
            candidate,count,work=action_then_baseline(x,prev,self.step,proposed,self.packed)
            transitions+=count;internal+=work;difference=candidate-base
            accepted=bool(np.all(difference<=1e-10) and difference[:,0].mean() < -1e-9)
        command=proposed if accepted else baseline_command
        self.step+=1
        if not robust_certificate(x,prev,command,self.packed,.5):
            raise RuntimeError('Chosen command failed physical certificate')
        return command,dict(in_set=True,certified=True,intervention=float(not accepted),
            nominal_joint_action_admitted=accepted,nominal_difference=difference.tolist(),
            gate_ms=(time.perf_counter()-started)*1000.,
            performance_forecast_transitions=transitions,performance_model_transitions=internal,
            performance_claim='Exact nominal full-baseline potential, not a disturbance risk bound')
