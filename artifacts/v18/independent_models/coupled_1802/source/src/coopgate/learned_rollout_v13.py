"""Learning-augmented ego rollout, with the learned tail exposed for ablation.

The neural policy is sequential PPO; online rollout is explicit and must not
be presented as an improvement of the standalone actor.
"""
import time
import numpy as np
import torch
from numba import njit
from .ego_v13 import load_ego,ego_rollout,ego_tail
from .single_v12 import constant_proposal
from .marl_v12 import LEVELS,project_joint
from .fast_v8 import advance_one,parameters
from .merge_v4 import MergeConfig


@njit(cache=True,fastmath=False)
def actor_action(x,previous,step,weights,markov,kind=0):
    obs=np.zeros(18,np.float32);obs[0]=x[1,0]/150.;obs[1]=x[1,1]/20.;obs[2]=previous[1]/5.
    obs[3]=1.;obs[4]=1. if x[1,0]>=60. else 0.;obs[5]=step/50.
    first=0 if abs(x[0,0]-x[1,0])<=abs(x[2,0]-x[1,0]) else 2
    radius=150. if markov else 70.
    for rank in range(2):
        j=first if rank==0 else 2-first;delta=x[j,0]-x[1,0];start=6+rank*6
        if abs(delta)>radius:continue
        obs[start]=1.;obs[start+1]=delta/radius;obs[start+2]=(x[j,1]-x[1,1])/20.;obs[start+3]=0.
        obs[start+4]=previous[j]/5. if markov else (x[j,0]+25.)/150.
        obs[start+5]=1. if x[j,0]>=60. else 0.
    # The deployed actor consumes float32 observations even in the model.
    for k in range(18):obs[k]=np.float32(obs[k])
    w0,b0,w1,b1,w2,b2=weights
    if kind==0:logits=w2@np.tanh(w1@np.tanh(w0@obs+b0)+b1)+b2
    else:
        if kind==2:
            features=np.zeros(26,np.float32);features[:18]=obs;p=obs[0]*150.;v=obs[1]*20.
            own_eta=max(0.,-25.-p)/max(v,1.)
            for rank in range(2):
                start=6+rank*6;present=obs[start];distance=obs[start+1]*150.
                neighbor_v=max(v+obs[start+2]*20.,1.);eta=max(0.,-25.-p-distance)/neighbor_v
                features[18+4*rank]=present*min(10.,max(-10.,eta-own_eta))/10.
                features[19+4*rank]=present*distance/(max(v,1.)*10.)
                features[20+4*rank]=present*(1. if abs(distance)<35. else 0.)
                features[21+4*rank]=present*(1. if eta<own_eta else 0.)
        else:features=obs
        logits=w2@np.maximum(np.float32(0.),w1@np.maximum(np.float32(0.),w0@features+b0)+b1)+b2
    best=-1e20;chosen=0.
    for i in range(len(LEVELS)):
        if abs(LEVELS[i]-previous[1])<=2.5+1e-8 and logits[i]>best:best=logits[i];chosen=LEVELS[i]
    return chosen


@njit(cache=True,fastmath=False)
def learned_tail(state,previous,step,c,radius,weight,weights,markov,kind=0,cutoff=1e12):
    x=state.copy();prev=previous.copy();value=0.
    for k in range(step,50):
        if x[1,0]>=c[11]:break
        proposal=constant_proposal(prev,c);proposal[1]=actor_action(x,prev,k,weights,markov,kind)
        command,_,ok=project_joint(x,prev,proposal,c,radius)
        if not ok:raise RuntimeError('Learned rollout outside backup contract')
        value+=.5+weight*abs(command[1]-prev[1])
        if value>=cutoff:return value
        if not advance_one(x,command,c):raise RuntimeError('Learned tail unsafe')
        prev=command
    return value


@njit(cache=True,fastmath=False)
def augmented_rollout(x,prev,step,c,radius,weight,weights,markov,kind=0):
    chosen,proposal,best,rule_value=ego_rollout(x,prev,step,c,radius,weight)
    if x[1,0]>=c[11]:return chosen,False,best
    # Keep the traditional rollout's incumbent; ties retain it.
    used=False
    for level in LEVELS[::-1]:
        if abs(level-prev[1])>2.5+1e-8:continue
        p=constant_proposal(prev,c);p[1]=level;command,_,ok=project_joint(x,prev,p,c,radius)
        if not ok:continue
        stage=.5+weight*abs(command[1]-prev[1]);following=x.copy()
        if not advance_one(following,command,c):raise RuntimeError('Candidate unsafe')
        tail=learned_tail(following,command,step+1,c,radius,weight,weights,markov,kind,best-stage)
        if stage+tail<best-1e-10:chosen=command.copy();best=stage+tail;used=True
    return chosen,used,best


class LearnedRolloutGate:
    def __init__(self,policy_path,config=None,radius=.5,weight=.05):
        self.c=config or MergeConfig();self.packed=parameters(self.c);self.radius=radius;self.weight=weight;self.step=0
        meta=torch.load(policy_path,map_location='cpu',weights_only=True);params=meta['model'];self.kind=0
        self.markov=meta.get('observation_version')=='markov_joint_memory_v1'
        if meta['algorithm'] in ('double_dqn_ego_v13','complete_action_double_dqn_ego_v13'):
            self.kind=1 if meta['algorithm']=='double_dqn_ego_v13' else 2
            keys=['encoder.0.weight','encoder.0.bias','encoder.2.weight','encoder.2.bias','advantage.weight','advantage.bias']
        else:keys=['actor.0.weight','actor.0.bias','actor.2.weight','actor.2.bias','actor.4.weight','actor.4.bias']
        self.weights=tuple(np.asarray(params[k].numpy(),dtype=np.float32) for k in keys)

    def select(self,x,previous,unused=None):
        start=time.perf_counter();command,used,value=augmented_rollout(x,previous,self.step,self.packed,self.radius,self.weight,self.weights,self.markov,self.kind)
        self.step+=1
        return command,{'in_set':True,'certified':True,'intervention':float(np.mean(np.abs(command-constant_proposal(previous,self.packed))>1e-8)),
            'gate_ms':(time.perf_counter()-start)*1000,'learned_tail_improved':bool(used),'rollout_value':float(value)}
