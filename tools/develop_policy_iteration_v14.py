"""Development-only model evaluation, including all intermediate checkpoints."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from coopgate.ego_v13 import ego_observations
from coopgate.marl_v12 import LEVELS, action_mask, batch_step
from coopgate.disturbance_v9 import disturbance_tape
from coopgate.fast_v8 import parameters
from coopgate.merge_v4 import MergeConfig
from coopgate.policy_iteration_v14 import load_policy
from develop_v12 import initial
from develop_ego_v13 import evaluate as old_evaluate


def evaluate(path, n=96):
    if path in ('rule','rollout'):
        return old_evaluate(path,n)
    torch.set_num_threads(1)
    model,meta=load_policy(path)
    c=MergeConfig();packed=parameters(c)
    x=np.array([initial(25000001+i,'nominal' if i<n//2 else 'dense') for i in range(n)])
    previous=np.zeros((n,3));finish=np.zeros((n,3));variation=np.zeros((n,3));complete=np.zeros((n,3),bool)
    tapes=np.array([disturbance_tape(25000001+i,['clean','uniform','bias','pulse'][i%4],.5,250) for i in range(n)])
    for step in range(50):
        mask=action_mask(previous);indices=np.where(mask,np.arange(len(LEVELS)),-1).max(-1)
        indices[:,1]=model.proposal(ego_observations(x,previous,step)[:,1],mask[:,1])
        nxt,command,valid=batch_step(x,previous,LEVELS[indices],tapes[:,5*step:5*(step+1)],packed,.5)
        if not valid.all():raise RuntimeError('Development safety contract failed')
        finish+=~complete*.5;variation+=(~complete)*np.abs(command-previous)
        complete|=nxt[:,:,0]>=c.goal;x=nxt;previous=command
        if complete.all():break
    return dict(path=str(path),n=n,ego_time=float(finish[:,1].mean()),ego_variation=float(variation[:,1].mean()),
                ego_cost=float((finish[:,1]+.05*variation[:,1]).mean()),other_time=float(finish[:,[0,2]].mean()),
                completed=int(complete.all(-1).sum()),rows=[dict(seed=25000001+i,finish=finish[i].tolist(),
                    variation=variation[i].tolist(),completed=bool(complete[i].all())) for i in range(n)])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('paths',nargs='+');parser.add_argument('--out');parser.add_argument('--n',type=int,default=96)
    args=parser.parse_args();results=[]
    for path in args.paths:
        result=evaluate(path,args.n);results.append(result)
        print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    if args.out:Path(args.out).write_text(json.dumps(results,indent=2)+'\n')
