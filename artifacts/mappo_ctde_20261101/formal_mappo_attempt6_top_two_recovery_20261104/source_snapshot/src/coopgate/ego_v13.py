"""Ego-specific single-agent objective; never substitute a team mean for ego gain."""
import time
import numpy as np
import torch
from numba import njit
from .merge_v4 import MergeConfig
from .fast_v8 import parameters,advance_one
from .marl_v12 import LEVELS,ActorCritic,project_joint,batch_step,action_mask,observations,central_state
from .robust_v9 import robust_certificate,validate
from .single_v12 import constant_proposal
from .training_v12 import SingleVectorMerge


def ego_observations(x,previous,step,config=None):
    """CAV local state includes visible neighbors' last executed commands.

    The joint shield depends on command memory. Dropping that memory aliases
    states with different transitions; the old distance-to-zone feature was
    redundant with own position plus relative distance, so we replace it.
    """
    x=np.asarray(x);previous=np.asarray(previous);single=x.ndim==2
    if single:x=x[None];previous=previous[None]
    result=observations(x,previous,step,config,sensor_range=150.)
    for i in range(3):
        neighbors=[j for j in range(3) if j!=i]
        distance=x[:,neighbors,0]-x[:,i,None,0];order=np.argsort(np.abs(distance),axis=1,kind='stable')
        for rank in range(2):
            j=np.array(neighbors)[order[:,rank]];visible=np.abs(x[np.arange(len(x)),j,0]-x[:,i,0])<=150.
            result[:,i,10+6*rank]=previous[np.arange(len(x)),j]/5.*visible
    return result[0] if single else result


@njit(cache=True,fastmath=False)
def ego_tail(state,previous,step,c,radius,weight,cutoff=1e12):
    x=state.copy();prev=previous.copy();value=0.;elapsed=0.;variation=0.
    for k in range(step,round(c[2]/c[1])):
        if x[1,0]>=c[11]:break
        command,_,valid=project_joint(x,prev,constant_proposal(prev,c),c,radius)
        if not valid:raise RuntimeError('Baseline outside contract')
        delta=abs(command[1]-prev[1]);elapsed+=c[1];variation+=delta
        value+=c[1]+weight*delta
        if value>=cutoff:return value,elapsed,variation
        if not advance_one(x,command,c):raise RuntimeError('Baseline unsafe')
        prev=command
    return value,elapsed,variation


@njit(cache=True,fastmath=False)
def ego_rollout(state,previous,step,c,radius,weight):
    proposal=constant_proposal(previous,c);base,_,valid=project_joint(state,previous,proposal,c,radius)
    if not valid:raise RuntimeError('No baseline continuation')
    best,_,_=ego_tail(state,previous,step,c,radius,weight);base_value=best
    chosen=base.copy();selected=proposal[1]
    if state[1,0]>=c[11]:return chosen,selected,best,base_value
    for level in LEVELS:
        if abs(level-previous[1])>c[6]*c[1]+1e-8:continue
        p=proposal.copy();p[1]=level;command,_,ok=project_joint(state,previous,p,c,radius)
        if not ok or np.array_equal(command,base):continue
        stage=c[1]+weight*abs(command[1]-previous[1]);x=state.copy()
        if not advance_one(x,command,c):raise RuntimeError('Trial unsafe')
        tail,_,_=ego_tail(x,command,step+1,c,radius,weight,best-stage)
        if stage+tail<best-1e-10:chosen,selected,best=command.copy(),level,stage+tail
    return chosen,selected,best,base_value


class EgoRolloutGate:
    def __init__(self,config=None,radius=.5,weight=.05):
        self.c=config or MergeConfig();self.radius=radius;self.weight=weight;self.packed=parameters(self.c);self.step=0

    def select(self,x,previous,unused=None):
        start=time.perf_counter();command,proposal,value,base=ego_rollout(x,previous,self.step,self.packed,self.radius,self.weight)
        self.step+=1
        return command,{'in_set':True,'certified':True,'intervention':float(np.mean(np.abs(command-constant_proposal(previous,self.packed))>1e-8)),
            'gate_ms':(time.perf_counter()-start)*1000.,'ego_proposal':float(proposal),'rollout_value':float(value),'baseline_value':float(base)}


class EgoVector(SingleVectorMerge):
    def inputs(self):
        return (ego_observations(self.x,self.previous,self.steps,self.c)[:,1:2],
                central_state(self.x,self.previous,self.steps,self.c),action_mask(self.previous,self.c)[:,1:2])

    def step(self,actions):
        mask=action_mask(self.previous,self.c);joint=np.where(mask,np.arange(len(LEVELS)),-1).max(-1)
        joint[:,1]=np.asarray(actions).reshape(self.n)
        nxt,commands,valid=batch_step(self.x,self.previous,LEVELS[joint],self.rng.uniform(-.5,.5,(self.n,5,3)),self.packed,.5)
        if not valid.all():raise RuntimeError('Ego training safety failed')
        delta=np.abs(commands[:,1]-self.previous[:,1]);stage=np.full(self.n,self.c.decision_dt)
        potential=-np.maximum(self.c.goal-self.x[:,1,0],0)/self.c.vmax
        future_potential=-np.maximum(self.c.goal-nxt[:,1,0],0)/self.c.vmax
        self.steps+=1;self.x=nxt;self.previous=commands
        terminal=self.x[:,1,0]>=self.c.goal;boundary=terminal|(self.steps>=50)
        future_potential[boundary]=0
        reward=-stage-self.variation_weight*delta+.995*future_potential-potential
        self.returns+=reward;self.time_cost+=stage;self.variation+=delta
        next_inputs=self.inputs()
        finished=[{'return':float(self.returns[i]),'cost':float(self.time_cost[i]),'variation':float(self.variation[i]),
            'complete':bool(terminal[i])} for i in np.flatnonzero(boundary)]
        self.episodes+=int(boundary.sum());self.reset(np.flatnonzero(boundary))
        return next_inputs,reward.astype(np.float32),terminal,boundary,finished


def load_ego(path):
    torch.set_num_threads(1);checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint['algorithm']=='double_dqn_ego_v13':
        from .ego_models_v13 import QPolicy
        model=QPolicy()
    elif checkpoint['algorithm'] in ('single_ego_ppo_v12','single_ego_ppo_v13'):model=ActorCritic()
    else:raise ValueError('Wrong policy type')
    model.load_state_dict(checkpoint['model']);model.eval();return model,checkpoint


class EgoPolicyGate:
    def __init__(self,policy,config=None,radius=.5):
        self.policy=policy;self.c=config or MergeConfig();self.radius=radius;self.packed=parameters(self.c);self.step=0

    def select(self,state,previous,unused=None):
        start=time.perf_counter();x,prev,_=validate(state,previous,self.c,self.radius)
        mask=action_mask(prev,self.c);indices=np.where(mask,np.arange(len(LEVELS)),-1).max(-1)
        indices[1]=self.policy.proposal(ego_observations(x,prev,self.step,self.c)[1:2],mask[1:2])[0]
        proposed=LEVELS[indices].copy();cleared=x[:,0]>=self.c.goal
        proposed[cleared]=np.minimum(self.c.accel,prev[cleared]+self.c.jerk*self.c.decision_dt)
        backup=np.maximum(prev-self.c.jerk*self.c.decision_dt,-self.c.brake)
        in_set=bool(robust_certificate(x,prev,backup,self.packed,self.radius))
        if not in_set:raise RuntimeError('Ego policy outside backup set')
        command,vertex,valid=project_joint(x,prev,proposed,self.packed,self.radius)
        if not valid:raise RuntimeError('No safe ego policy candidate')
        self.step+=1
        return command,{'in_set':True,'certified':True,'intervention':float(np.mean(np.abs(command-proposed)>1e-8)),
            'gate_ms':(time.perf_counter()-start)*1000.,'proposal':proposed.tolist(),'agent_actions':indices.tolist(),
            'joint_proposal_accepted':bool(np.array_equal(command,proposed))}
