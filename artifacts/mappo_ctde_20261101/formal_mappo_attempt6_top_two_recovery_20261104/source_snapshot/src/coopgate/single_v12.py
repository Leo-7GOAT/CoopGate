"""Single trainable ramp vehicle and a strong matched rollout baseline."""
import time
import numpy as np
from numba import njit
from .merge_v4 import MergeConfig
from .fast_v8 import parameters,advance_one
from .marl_v12 import LEVELS,project_joint


@njit(cache=True,fastmath=False)
def constant_proposal(previous,c):
    return np.minimum(c[5],previous+c[6]*c[1])


@njit(cache=True,fastmath=False)
def constant_tail(state,previous,step,c,radius,weight,cutoff=1e12):
    x=state.copy();prev=previous.copy();value=0.;ticks=0;variation=0.
    for k in range(step,round(c[2]/c[1])):
        active=np.sum(x[:,0]<c[11])
        if active==0:break
        command,_,valid=project_joint(x,prev,constant_proposal(prev,c),c,radius)
        if not valid:raise RuntimeError('Baseline outside contract')
        delta=np.abs(command-prev).sum();ticks+=active;variation+=delta
        value+=active*c[1]/3.+weight*delta
        if value>=cutoff:return value,ticks,variation
        if not advance_one(x,command,c):raise RuntimeError('Nominal baseline transition unsafe')
        prev=command
    return value,ticks,variation


@njit(cache=True,fastmath=False)
def single_rollout(state,previous,step,c,radius,weight):
    proposal=constant_proposal(previous,c)
    base,_,valid=project_joint(state,previous,proposal,c,radius)
    if not valid:raise RuntimeError('No baseline continuation')
    best,_,_=constant_tail(state,previous,step,c,radius,weight)
    chosen=base.copy();selected=proposal[1];base_value=best
    if state[1,0]>=c[11]:return chosen,selected,best,base_value
    for level in LEVELS:
        if abs(level-previous[1])>c[6]*c[1]+1e-8:continue
        p=proposal.copy();p[1]=level
        command,_,ok=project_joint(state,previous,p,c,radius)
        if not ok:continue
        if np.array_equal(command,base):continue
        stage=np.sum(state[:,0]<c[11])*c[1]/3.+weight*np.abs(command-previous).sum()
        x=state.copy()
        if not advance_one(x,command,c):raise RuntimeError('Candidate transition unsafe')
        tail,_,_=constant_tail(x,command,step+1,c,radius,weight,best-stage)
        cost=stage+tail
        if cost<best-1e-10:chosen,selected,best=command.copy(),level,cost
    return chosen,selected,best,base_value


class SingleRolloutGate:
    def __init__(self,config=None,radius=.5,variation_weight=.05):
        self.c=config or MergeConfig();self.radius=radius;self.weight=variation_weight
        self.packed=parameters(self.c);self.step=0

    def select(self,state,previous,unused=None):
        start=time.perf_counter()
        command,selected,value,base=single_rollout(state,previous,self.step,self.packed,self.radius,self.weight)
        self.step+=1
        return command,{'in_set':True,'certified':True,'intervention':float(np.mean(np.abs(command-constant_proposal(previous,self.packed))>1e-8)),
            'gate_ms':(time.perf_counter()-start)*1000.,'ego_proposal':float(selected),
            'rollout_value':float(value),'baseline_value':float(base)}
