"""Learning-free cooperative rollout baseline for the future MARL comparison.

Two deterministic coordinate sweeps optimize current joint proposals against a
full-horizon rule continuation. This is approximate discrete planning, not a
global optimum or new MPC algorithm. Every method keeps the V14 physical filter.
"""
import time
import numpy as np
from numba import njit
from .marl_v12 import LEVELS,project_joint
from .single_v12 import constant_proposal
from .fast_v8 import advance_one,parameters
from .merge_v4 import MergeConfig
from .robust_v9 import validate,robust_certificate


@njit(cache=True,fastmath=False)
def team_rule_tail(state,previous,step,c,cutoff=1e12):
    x=state.copy();prev=previous.copy();value=0.;transitions=0
    for k in range(step,round(c[2]/c[1])):
        active=x[:,0]<c[11]
        if not np.any(active):break
        command,_,ok=project_joint(x,prev,constant_proposal(prev,c),c,.5)
        if not ok:raise RuntimeError('Rule continuation outside physical contract')
        for i in range(3):
            if active[i]:value+=(c[1]+.05*abs(command[i]-prev[i]))/3.
        if value>=cutoff:return value,transitions
        if not advance_one(x,command,c):raise RuntimeError('Unsafe rule continuation')
        transitions+=1;prev=command
    return value,transitions


@njit(cache=True,fastmath=False)
def joint_rollout(state,previous,step,c,sweeps=2):
    chosen,_,ok=project_joint(state,previous,constant_proposal(previous,c),c,.5)
    if not ok:raise RuntimeError('No certified baseline command')
    best,transitions=team_rule_tail(state,previous,step,c);base_value=best
    if np.all(state[:,0]>=c[11]):return chosen,best,base_value,transitions
    # Ramp first is a fixed ordering, not chosen from test outcomes.
    for _ in range(sweeps):
        improved=False
        for i in (1,0,2):
            if state[i,0]>=c[11]:continue
            center=chosen.copy()
            for level in LEVELS:
                if abs(level-previous[i])>c[6]*c[1]+1e-8:continue
                proposal=center.copy();proposal[i]=level
                for j in range(3):
                    if state[j,0]>=c[11]:proposal[j]=min(c[5],previous[j]+c[6]*c[1])
                command,_,valid=project_joint(state,previous,proposal,c,.5)
                if not valid:raise RuntimeError('No certificate for candidate')
                if np.array_equal(command,chosen):continue
                stage=0.
                for j in range(3):
                    if state[j,0]<c[11]:stage+=(c[1]+.05*abs(command[j]-previous[j]))/3.
                if stage>=best:continue
                following=state.copy()
                if not advance_one(following,command,c):raise RuntimeError('Unsafe planning transition')
                tail,count=team_rule_tail(following,command,step+1,c,best-stage)
                transitions+=1+count
                if stage+tail<best-1e-10:
                    chosen=command.copy();best=stage+tail;improved=True
        if not improved:break
    return chosen,best,base_value,transitions


class JointRolloutGate:
    def __init__(self,config=None):
        self.c=config or MergeConfig();self.packed=parameters(self.c);self.step=0

    def select(self,state,previous,unused=None):
        start=time.perf_counter();x,prev,_=validate(state,previous,self.c,.5)
        backup=np.maximum(prev-self.c.jerk*self.c.decision_dt,-self.c.brake)
        if not robust_certificate(x,prev,backup,self.packed,.5):raise RuntimeError('Joint planner outside backup set')
        command,value,base,count=joint_rollout(x,prev,self.step,self.packed)
        self.step+=1
        return command,dict(in_set=True,certified=True,
            intervention=float(np.mean(np.abs(command-constant_proposal(prev,self.packed))>1e-8)),
            gate_ms=(time.perf_counter()-start)*1000.,planning_value=value,rule_value=base,
            performance_model_transitions=count)
