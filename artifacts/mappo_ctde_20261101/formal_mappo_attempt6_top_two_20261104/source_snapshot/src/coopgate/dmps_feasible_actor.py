"""State-dependent feasible-request outputs for original actor/target modules.

The physical action support is unchanged. This addresses the execution-map
plateau, not Tanh saturation, critic errors, geometric safety, or convergence.
"""
import torch


def feasible_request(observation,latent):
    if observation.shape[-1]!=10 or latent.shape[-1]!=3 or observation.shape[:-1]!=latent.shape[:-1]:
        raise ValueError('Matching (...,10) observation and (...,3) latent tensors required')
    if not torch.isfinite(observation).all() or not torch.isfinite(latent).all() or torch.any(torch.abs(latent)>1+1e-7):
        raise ValueError('Finite observations and bounded Tanh outputs required')
    previous=5*observation[..., (2,5,8)]
    if torch.any(previous < -5.-1e-6) or torch.any(previous > 2.5+1e-6):
        raise ValueError('Previous command outside declared execution domain')
    lower=torch.maximum(previous-2.5,torch.full_like(previous,-5.))
    upper=torch.minimum(previous+2.5,torch.full_like(previous,2.5))
    command=lower+(torch.clamp(latent,-1.,1.)+1)*(upper-lower)/2
    # Same conservative observable completion convention as the critic map.
    command=torch.where(observation[..., (0,3,6)]>.4,upper,command)
    # Preserve the original normalized action API even at float32 endpoints.
    return torch.clamp((command+1.25)/3.75,-1.,1.)


class FeasibleActorOutputs:
    names=('actor','actor_targ')

    def __init__(self,model):
        self.calls={name:0 for name in self.names};self.handles=[]
        for name in self.names:
            if getattr(getattr(model,name),'_coopgate_feasible_output',False):
                raise ValueError('Feasible actor output hook already installed')
        for name in self.names:
            actor=getattr(model,name);actor._coopgate_feasible_output=True
            def transform(module,inputs,output,role=name):
                self.calls[role]+=1
                return feasible_request(inputs[0],output)
            self.handles.append(actor.register_forward_hook(transform))
