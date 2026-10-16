"""Certificate-conditioned policy with analytic current-command arrival features.

Constant-command arrival estimates are input features, not optimized future
trajectories. Learned scores still decide the current command; the physical
certificate is unchanged. No future disturbance is supplied to the actor.
"""
import numpy as np
import torch
from torch import nn
from numba import njit
from .certificate_policy_v14 import candidates as physical_candidates
from .certificate_policy_v14 import CertificatePolicy
from .ego_v13 import ego_tail
from .fast_v8 import advance_one
from .marl_v12 import LEVELS

FEATURES=40


@njit(cache=True,fastmath=False)
def arrival_feature(distance,speed,command):
    """Continuous-time constant-command ETA, with 20 m/s cap and 25 s sentinel.

    This is deliberately not a substitute for the discrete robust certificate
    or for the measured arrival time used by the reward and final tests.
    """
    if distance<=0:return 0.
    if abs(command)<1e-10:return min(25.,distance/max(speed,1e-6))
    if command>0:
        cap_time=max(0.,20.-speed)/command
        cap_distance=speed*cap_time+.5*command*cap_time*cap_time
        if distance>=cap_distance:return min(25.,cap_time+(distance-cap_distance)/20.)
    discriminant=speed*speed+2.*command*distance
    if discriminant<=0:return 25.
    return min(25.,2.*distance/max(speed+np.sqrt(discriminant),1e-8))


@njit(cache=True,fastmath=False)
def candidates(x,previous,step,packed):
    base,commands,mask=physical_candidates(x,previous,step,packed)
    features=np.zeros((len(LEVELS),FEATURES),np.float32);features[:,:28]=base
    for a in range(len(LEVELS)):
        if not mask[a]:continue
        entry=np.zeros(3);leave=np.zeros(3)
        for i in range(3):
            entry[i]=arrival_feature(packed[8]-x[i,0],x[i,1],commands[a,i])
            leave[i]=arrival_feature(packed[9]-x[i,0],x[i,1],commands[a,i])
            goal=arrival_feature(packed[11]-x[i,0],x[i,1],commands[a,i])
            features[a,28+3*i]=entry[i]/10.
            features[a,29+3*i]=leave[i]/10.
            features[a,30+3*i]=goal/10.
        features[a,37]=(entry[1]-leave[0])/10.
        features[a,38]=(entry[1]-leave[2])/10.
        features[a,39]=1. if leave[1]<min(entry[0],entry[2]) else 0.
    return features,commands,mask


@njit(cache=True,fastmath=False)
def batch_candidates(x,previous,steps,packed):
    features=np.empty((len(x),len(LEVELS),FEATURES),np.float32)
    commands=np.empty((len(x),len(LEVELS),3));mask=np.empty((len(x),len(LEVELS)),np.bool_)
    for i in range(len(x)):features[i],commands[i],mask[i]=candidates(x[i],previous[i],steps[i],packed)
    return features,commands,mask


class KinematicPolicy(CertificatePolicy):
    def __init__(self,width=64):
        nn.Module.__init__(self)
        self.network=nn.Sequential(nn.Linear(FEATURES,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU(),nn.Linear(width,1))


def load_policy(path):
    meta=torch.load(path,map_location='cpu',weights_only=True)
    if meta['algorithm']!='kinematic_feature_policy_iteration_v14':raise ValueError('Unexpected checkpoint')
    model=KinematicPolicy(meta['width']);model.load_state_dict(meta['model']);model.eval()
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
def batch_returns(states,previous,steps,packed,weights,use_learned):
    features=np.empty((len(states),len(LEVELS),FEATURES),np.float32)
    values=np.full((len(states),len(LEVELS)),np.inf);transitions=0
    for e in range(len(states)):
        x,prev,step=states[e],previous[e],steps[e]
        fe,commands,valid=candidates(x,prev,step,packed);features[e]=fe
        for a in range(len(LEVELS)):
            if not valid[a]:continue
            duplicate=-1
            for b in range(a):
                if valid[b] and np.array_equal(commands[a],commands[b]):duplicate=b;break
            if duplicate>=0:values[e,a]=values[e,duplicate];continue
            nxt=x.copy();command=commands[a]
            if not advance_one(nxt,command,packed):raise RuntimeError('Unsafe branch')
            transitions+=1
            tail,elapsed,_=ego_tail(nxt,command,step+1,packed,.5,.05)
            transitions+=round(elapsed/.5)
            if use_learned:
                state=nxt.copy();memory=command.copy();learned=0.
                for k in range(step+1,50):
                    if state[1,0]>=packed[11]:break
                    u,_=choose_compiled(state,memory,k,packed,weights)
                    learned+=.5+.05*abs(u[1]-memory[1])
                    if not advance_one(state,u,packed):raise RuntimeError('Unsafe learned tail')
                    memory=u;transitions+=1
                tail=min(tail,learned)
            values[e,a]=.5+.05*abs(command[1]-prev[1])+tail
    return features,values,transitions
