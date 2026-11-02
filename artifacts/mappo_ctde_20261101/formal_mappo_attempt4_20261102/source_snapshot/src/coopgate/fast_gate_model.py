"""Supervised gate surrogate; genuine MAPPO proposal actors stay separate."""
import numpy as np
import torch
from torch import nn
from .ego_v13 import ego_observations
from .mappo_ctde import local_kinematic_features
from .marl_v12 import central_state


def online_features(state,previous,step,proposal,baseline,lower,upper,memory,bounds,config):
    normalized=np.array(bounds,copy=True);normalized[:,2:4]*=1000.;normalized[:,6]*=10.
    derived=local_kinematic_features(torch.as_tensor(ego_observations(state,previous,step,config)))[:,18:].numpy().ravel()
    return np.concatenate((central_state(state,previous,step,config),np.asarray(proposal)/5,np.asarray(baseline)/5,
        np.asarray(lower)/5,np.asarray(upper)/5,np.asarray(memory)/5,normalized.ravel(),derived)).astype(np.float32)


class FastGateModel(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(73,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,29))
        self.register_buffer('mean',torch.zeros(73));self.register_buffer('scale',torch.ones(73))
        self.register_buffer('classes',torch.ones(29,dtype=torch.bool))

    def forward(self,x):return self.net((x-self.mean)/self.scale).masked_fill(~self.classes,-1e9)


def ensemble_choice(models,features,threshold):
    if not models or not 0<=threshold<=1:raise ValueError('Nonempty fixed ensemble and score threshold required')
    with torch.no_grad():prob=torch.stack([m(torch.as_tensor(features)).softmax(-1) for m in models])
    mean=prob.mean(0);label=int(mean.argmax());score=float(mean[label])
    agreement=bool((prob.argmax(-1)==label).all())
    return (label if label!=0 and agreement and score>=threshold else 0),dict(label=label,score=score,unanimous=agreement,score_is_safety_probability=False)
