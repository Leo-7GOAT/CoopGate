"""Development evaluation of neural action scoring, with no online cost search."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from coopgate.certificate_policy_v14 import load_policy,batch_candidates
from coopgate.disturbance_v9 import disturbance_tape
from coopgate.fast_v8 import parameters
from coopgate.merge_v4 import MergeConfig
from coopgate.marl_v12 import batch_step,LEVELS
from develop_v12 import initial


def evaluate(path,n=96):
    torch.set_num_threads(1);model,_=load_policy(path);c=MergeConfig();packed=parameters(c)
    x=np.array([initial(25000001+i,'nominal' if i<n//2 else 'dense') for i in range(n)])
    previous=np.zeros((n,3));finish=np.zeros((n,3));variation=np.zeros((n,3));complete=np.zeros((n,3),bool)
    tapes=np.array([disturbance_tape(25000001+i,['clean','uniform','bias','pulse'][i%4],.5,250) for i in range(n)])
    for step in range(50):
        features,commands,mask=batch_candidates(x,previous,np.full(n,step),packed)
        selected=model.choose(features,mask);chosen=commands[np.arange(n),selected]
        nxt,command,valid=batch_step(x,previous,chosen,tapes[:,5*step:5*(step+1)],packed,.5)
        if not valid.all() or not np.array_equal(command,chosen):raise RuntimeError('Training/deployment safety projection mismatch')
        finish+=~complete*.5;variation+=(~complete)*np.abs(command-previous)
        complete|=nxt[:,:,0]>=c.goal;x=nxt;previous=command
        if complete.all():break
    return dict(path=str(path),n=n,ego_time=float(finish[:,1].mean()),ego_variation=float(variation[:,1].mean()),
                ego_cost=float((finish[:,1]+.05*variation[:,1]).mean()),other_time=float(finish[:,[0,2]].mean()),
                completed=int(complete.all(-1).sum()),rows=[dict(seed=25000001+i,finish=finish[i].tolist(),
                    variation=variation[i].tolist(),completed=bool(complete[i].all())) for i in range(n)])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('paths',nargs='+');p.add_argument('--out');p.add_argument('--n',type=int,default=96)
    a=p.parse_args();results=[]
    for path in a.paths:
        r=evaluate(path,a.n);results.append(r);print(json.dumps({k:v for k,v in r.items() if k!='rows'}),flush=True)
    if a.out:Path(a.out).write_text(json.dumps(results,indent=2)+'\n')
