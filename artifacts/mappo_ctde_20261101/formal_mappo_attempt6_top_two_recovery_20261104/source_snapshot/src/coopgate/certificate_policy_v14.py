"""Sequential learned action values with explicit safety-filter consequences.

The actor scores the current admissible ego commands after the SAME safety
projection used by the baseline. There is no future performance rollout at
deployment. Full current three-vehicle state is assumed available, exactly as
for the planning baseline; this version does not claim partial observability.
"""
import time
import numpy as np
import torch
from torch import nn
from numba import njit
from .marl_v12 import LEVELS, project_joint
from .single_v12 import constant_proposal
from .ego_v13 import ego_tail
from .fast_v8 import advance_one, parameters
from .merge_v4 import MergeConfig
from .robust_v9 import robust_certificate

FEATURES = 28


@njit(cache=True, fastmath=False)
def candidates(x, previous, step, packed):
    """Current-state features and certified commands, without cost lookahead."""
    base=np.empty(22,np.float32)
    for i in range(3):
        p,v=x[i]
        base[5*i]=p/150.
        base[5*i+1]=v/20.
        base[5*i+2]=previous[i]/5.
        base[5*i+3]=min(15.,max(-25.-p,0.)/max(v,1.))/10.
        # An approximate braking-margin FEATURE, never a safety certificate.
        base[5*i+4]=(-25.-p-v*v/9.-v*max(previous[i]+5.,0.)/5.)/150.
    for k in range(3):
        i,j=(0,1) if k==0 else ((1,2) if k==1 else (0,2))
        base[15+2*k]=(x[i,0]-x[j,0])/150.
        base[16+2*k]=(x[i,1]-x[j,1])/20.
    base[21]=step/50.
    features=np.zeros((len(LEVELS),FEATURES),np.float32)
    commands=np.zeros((len(LEVELS),3))
    valid=np.zeros(len(LEVELS),np.bool_)
    for a in range(len(LEVELS)):
        if abs(LEVELS[a]-previous[1])>2.5+1e-8:continue
        if x[1,0]>=packed[11] and abs(LEVELS[a]-min(packed[5],previous[1]+2.5))>1e-8:continue
        proposal=constant_proposal(previous,packed);proposal[1]=LEVELS[a]
        command,_,ok=project_joint(x,previous,proposal,packed,.5)
        if not ok:raise RuntimeError('No certificate for ego proposal')
        valid[a]=True;commands[a]=command;features[a,:22]=base
        for i in range(3):
            features[a,22+i]=command[i]/5.
            features[a,25+i]=(command[i]-previous[i])/5.
    return features,commands,valid


@njit(cache=True,fastmath=False)
def batch_candidates(states,previous,steps,packed):
    features=np.empty((len(states),len(LEVELS),FEATURES),np.float32)
    commands=np.empty((len(states),len(LEVELS),3))
    valid=np.empty((len(states),len(LEVELS)),np.bool_)
    for i in range(len(states)):
        features[i],commands[i],valid[i]=candidates(states[i],previous[i],steps[i],packed)
    return features,commands,valid


class CertificatePolicy(nn.Module):
    def __init__(self,width=64):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(FEATURES,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU(),nn.Linear(width,1))

    def forward(self,features):
        return self.network(features).squeeze(-1)

    @torch.no_grad()
    def choose(self,features,mask):
        score=self(torch.as_tensor(features,dtype=torch.float32)).masked_fill(~torch.as_tensor(mask,dtype=torch.bool),-1e9)
        # Equal feature vectors have equal returns; use largest proposal on ties.
        return (len(LEVELS)-1-score.flip(-1).argmax(-1)).numpy()

    def compiled_weights(self):
        return tuple(value.detach().cpu().numpy().astype(np.float32) for value in self.state_dict().values())


def load_policy(path):
    meta=torch.load(path,map_location='cpu',weights_only=True)
    if meta['algorithm']!='certificate_feature_policy_iteration_v14':raise ValueError('Unexpected actor')
    model=CertificatePolicy(meta['width']);model.load_state_dict(meta['model']);model.eval()
    return model,meta


@njit(cache=True,fastmath=False)
def choose_compiled(x,previous,step,packed,weights):
    features,commands,valid=candidates(x,previous,step,packed)
    w0,b0,w1,b1,w2,b2=weights
    z=np.maximum(np.float32(0.),features@w0.T+b0)
    z=np.maximum(np.float32(0.),z@w1.T+b1)
    scores=(z@w2.T+b2)[:,0]
    best=-1e30;selected=-1
    for a in range(len(LEVELS)):
        if valid[a] and scores[a]>=best:best=scores[a];selected=a
    return commands[selected].copy(),selected


@njit(cache=True,fastmath=False)
def returns(x,previous,step,packed,weights,use_learned):
    features,commands,valid=candidates(x,previous,step,packed)
    values=np.full(len(LEVELS),np.inf);transitions=0
    for a in range(len(LEVELS)):
        if not valid[a]:continue
        # Many proposals map to the same joint command; cache their exact cost.
        duplicate=-1
        for b in range(a):
            if valid[b] and np.array_equal(commands[a],commands[b]):duplicate=b;break
        if duplicate>=0:values[a]=values[duplicate];continue
        nxt=x.copy();command=commands[a]
        if not advance_one(nxt,command,packed):raise RuntimeError('Unsafe branch')
        transitions+=1
        tail,elapsed,_=ego_tail(nxt,command,step+1,packed,.5,.05)
        transitions+=round(elapsed/.5)
        if use_learned:
            state=nxt.copy();memory=command.copy();learned=0.
            for k in range(step+1,50):
                if state[1,0]>=packed[11]:break
                action,_=choose_compiled(state,memory,k,packed,weights)
                learned+=.5+.05*abs(action[1]-memory[1])
                if not advance_one(state,action,packed):raise RuntimeError('Unsafe actor tail')
                memory=action;transitions+=1
            tail=min(tail,learned)
        values[a]=.5+.05*abs(command[1]-previous[1])+tail
    return features,values,transitions


@njit(cache=True,fastmath=False)
def batch_returns(states,previous,steps,packed,weights,use_learned):
    features=np.empty((len(states),len(LEVELS),FEATURES),np.float32)
    values=np.empty((len(states),len(LEVELS)))
    transitions=0
    for i in range(len(states)):
        features[i],values[i],count=returns(states[i],previous[i],steps[i],packed,weights,use_learned)
        transitions+=count
    return features,values,transitions


class CertificatePolicyGate:
    def __init__(self,model,config=None):
        self.model=model;self.c=config or MergeConfig();self.packed=parameters(self.c);self.step=0

    def select(self,x,previous,unused=None):
        started=time.perf_counter()
        features,commands,mask=candidates(x,previous,self.step,self.packed)
        index=int(self.model.choose(features[None],mask[None])[0])
        command=commands[index]
        if not robust_certificate(x,previous,command,self.packed,.5):raise RuntimeError('Uncertified actor command')
        self.step+=1
        proposal=constant_proposal(previous,self.packed);proposal[1]=LEVELS[index]
        return command,dict(in_set=True,certified=True,proposal=proposal.tolist(),ego_action=index,
            intervention=float(np.mean(np.abs(command-proposal)>1e-8)),
            gate_ms=(time.perf_counter()-started)*1000.,performance_rollouts=0)
