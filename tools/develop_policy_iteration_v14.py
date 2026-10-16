"""Development-only model evaluation, including all intermediate checkpoints."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from coopgate.ego_v13 import ego_observations,EgoRolloutGate
from coopgate.marl_v12 import LEVELS, action_mask, batch_step
from coopgate.disturbance_v9 import disturbance_tape
from coopgate.fast_v8 import parameters
from coopgate.merge_v4 import MergeConfig
from coopgate.policy_iteration_v14 import load_policy
from develop_v12 import initial
from develop_ego_v13 import evaluate as old_evaluate


def evaluate(path, n=96, seed_base=25000001):
    if path in ('rule','rollout') and seed_base==25000001:
        return old_evaluate(path,n)
    torch.set_num_threads(1)
    model=None if path in ('rule','rollout') else load_policy(path)[0]
    c=MergeConfig();packed=parameters(c)
    x=np.array([initial(seed_base+i,'nominal' if i<n//2 else 'dense') for i in range(n)])
    gates=[EgoRolloutGate() for _ in range(n)] if path=='rollout' else None
    previous=np.zeros((n,3));finish=np.zeros((n,3));variation=np.zeros((n,3));complete=np.zeros((n,3),bool)
    tapes=np.array([disturbance_tape(seed_base+i,['clean','uniform','bias','pulse'][i%4],.5,250) for i in range(n)])
    for step in range(50):
        mask=action_mask(previous);indices=np.where(mask,np.arange(len(LEVELS)),-1).max(-1)
        if model is not None:indices[:,1]=model.proposal(ego_observations(x,previous,step)[:,1],mask[:,1])
        proposed=LEVELS[indices]
        if gates is not None:proposed=np.array([gate.select(s,p)[0] for gate,s,p in zip(gates,x,previous)])
        nxt,command,valid=batch_step(x,previous,proposed,tapes[:,5*step:5*(step+1)],packed,.5)
        if not valid.all():raise RuntimeError('Development safety contract failed')
        finish+=~complete*.5;variation+=(~complete)*np.abs(command-previous)
        complete|=nxt[:,:,0]>=c.goal;x=nxt;previous=command
        if complete.all():break
    return dict(path=str(path),n=n,seed_base=seed_base,backend='exact model development only',ego_time=float(finish[:,1].mean()),ego_variation=float(variation[:,1].mean()),
                ego_cost=float((finish[:,1]+.05*variation[:,1]).mean()),other_time=float(finish[:,[0,2]].mean()),
                completed=int(complete.all(-1).sum()),rows=[dict(seed=seed_base+i,finish=finish[i].tolist(),
                    variation=variation[i].tolist(),completed=bool(complete[i].all())) for i in range(n)])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('paths',nargs='+');parser.add_argument('--out');parser.add_argument('--n',type=int,default=96)
    parser.add_argument('--seed-base',type=int,default=25000001)
    args=parser.parse_args();results=[]
    for path in args.paths:
        result=evaluate(path,args.n,args.seed_base);results.append(result)
        print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    if args.out:Path(args.out).write_text(json.dumps(results,indent=2)+'\n')
