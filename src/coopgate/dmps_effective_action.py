"""Observable execution-command inputs for the original four DMPS critics.

This is an explicit task-adapted variant, not the unmodified original method.
Actor, optimizer and author function bodies are unchanged. Hooks must be
reinstalled on every loaded model; a state_dict alone does not encode them.
"""
import torch


def effective_action(observation,action):
    if observation.shape[-1]!=10 or action.shape[-1]!=3 or observation.shape[:-1]!=action.shape[:-1]:
        raise ValueError('Matching (...,10) observation and (...,3) action tensors required')
    position=observation[..., (0,3,6)]
    previous=5*observation[..., (2,5,8)]
    lower=torch.maximum(previous-2.5,torch.full_like(previous,-5.))
    upper=torch.minimum(previous+2.5,torch.full_like(previous,2.5))
    desired=-1.25+3.75*torch.clamp(action,-1.,1.)
    command=torch.minimum(torch.maximum(desired,lower),upper)
    cruise=torch.minimum(torch.maximum(torch.full_like(previous,2.5),lower),upper)
    # Strict > avoids classifying a just-before-goal physical position rounded
    # to float32 0.4 as complete. At the boundary band the regular map remains;
    # no claim of exact post-goal quotienting is made inside that band.
    command=torch.where(position>.4,cruise,command)
    return (command+1.25)/3.75


class EffectiveCriticInputs:
    """Parameter-preserving input hooks for both online and target critics."""
    names=('critic1','critic2','critic1_targ','critic2_targ')

    def __init__(self,model):
        self.calls={name:0 for name in self.names};self.handles=[]
        for name in self.names:
            critic=getattr(model,name)
            if getattr(critic,'_coopgate_effective_input',False):
                raise ValueError('Effective-command hook already installed')
        for name in self.names:
            critic=getattr(model,name)
            critic._coopgate_effective_input=True
            def transform(module,inputs,role=name):
                self.calls[role]+=1
                obs,action=inputs
                return obs,effective_action(obs,action)
            self.handles.append(critic.register_forward_pre_hook(transform))
