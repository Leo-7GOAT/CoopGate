"""Development-only headroom diagnostic; this is a planner, not RL."""
import json
import numpy as np
from numba import njit
from coopgate.ego_v13 import ego_tail,ego_rollout
from coopgate.marl_v12 import LEVELS,project_joint,batch_step
from coopgate.single_v12 import constant_proposal
from coopgate.fast_v8 import advance_one,parameters
from coopgate.merge_v4 import MergeConfig
from coopgate.disturbance_v9 import disturbance_tape
from develop_v12 import initial


@njit(cache=True)
def two_stage(x,prev,step,c,radius=.5,weight=.05):
    chosen,proposal,value,base=ego_rollout(x,prev,step,c,radius,weight)
    if x[1,0]>=c[11]:return chosen
    best=1e12
    for action in LEVELS:
        if abs(action-prev[1])>2.5+1e-8:continue
        proposed=constant_proposal(prev,c);proposed[1]=action
        command,_,ok=project_joint(x,prev,proposed,c,radius)
        if not ok:continue
        following=x.copy()
        if not advance_one(following,command,c):raise RuntimeError('Unsafe branch')
        tail=0.
        if following[1,0]<c[11]:_,_,tail,_=ego_rollout(following,command,step+1,c,radius,weight)
        value=.5+weight*abs(command[1]-prev[1])+tail
        if value<best-1e-9:chosen=command.copy();best=value
    return chosen


if __name__=='__main__':
    c=MergeConfig();packed=parameters(c);n=96
    states=np.array([initial(25000001+i,'nominal' if i<n//2 else 'dense') for i in range(n)])
    prev=np.zeros((n,3));finished=np.zeros((n,3),bool);times=np.zeros((n,3));tv=np.zeros((n,3))
    tapes=np.array([disturbance_tape(25000001+i,['clean','uniform','bias','pulse'][i%4],.5,250) for i in range(n)])
    for step in range(50):
        proposals=np.array([two_stage(x,p,step,packed) for x,p in zip(states,prev)])
        following,commands,valid=batch_step(states,prev,proposals,tapes[:,5*step:5*(step+1)],packed,.5)
        if not valid.all():raise RuntimeError('Unsafe diagnostic')
        times+=~finished*.5;tv+=(~finished)*np.abs(commands-prev)
        finished|=following[:,:,0]>=c.goal;states=following;prev=commands
        if finished.all():break
    result={'ego_time':times[:,1].mean(),'ego_variation':tv[:,1].mean(),'ego_cost':(times[:,1]+.05*tv[:,1]).mean(),
        'other_time':times[:,[0,2]].mean(),'completed':int(finished.all(-1).sum())}
    print(json.dumps(result))
