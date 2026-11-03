"""Deployable dueling Q network; action selection uses only ego observations."""
import torch
from torch import nn
from .marl_v12 import OBS_DIM,LEVELS


class QPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(OBS_DIM,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
        self.value=nn.Linear(128,1);self.advantage=nn.Linear(128,len(LEVELS))

    def forward(self,obs):
        features=self.encoder(obs);a=self.advantage(features)
        return self.value(features)+a-a.mean(-1,keepdim=True)

    @torch.no_grad()
    def proposal(self,obs,mask):
        return self(torch.as_tensor(obs,dtype=torch.float32)).masked_fill(~torch.as_tensor(mask,dtype=torch.bool),-1e9).argmax(-1).numpy()
