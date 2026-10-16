"""Fast development-only evaluation; this is NOT the final SUMO test set."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from coopgate.marl_v12 import load_policy,observations,action_mask,batch_step,LEVELS
from coopgate.merge_v4 import MergeConfig
from coopgate.fast_v8 import parameters
from coopgate.disturbance_v9 import disturbance_tape


def initial(seed,regime):
    rng=np.random.default_rng(seed)
    v=rng.uniform(8,14,3) if regime=='nominal' else rng.uniform(12,17,3)
    p=-rng.uniform(70,110,3) if regime=='nominal' else -rng.uniform(70,85,3)
    p[2]=p[0]-rng.uniform(30,45)
    return np.column_stack((p,v))


def evaluate(path,n=64):
    torch.set_num_threads(1);c=MergeConfig();packed=parameters(c)
    model=None if path=='constant' else load_policy(path)[0]
    single=False if model is None else load_policy(path)[1]['algorithm']=='single_ego_ppo_v12'
    states=np.array([initial(18000001+i,'nominal' if i<n//2 else 'dense') for i in range(n)])
    prev=np.zeros((n,3));costs=np.zeros(n);tv=np.zeros(n);finished=np.zeros(n,bool)
    accepted=np.zeros(n);cycles=np.zeros(n);proposals=[]
    tapes=np.array([disturbance_tape(18000001+i,('clean','uniform','bias','pulse')[i%4],.5,250) for i in range(n)])
    for step in range(50):
        masks=action_mask(prev,c)
        a=np.where(masks,np.arange(len(LEVELS)),-1).max(-1)
        if model is not None:
            ob=observations(states,prev,step,c)
            if single: a[:,1]=model.proposal(ob[:,1],masks[:,1])
            else: a=model.proposal(ob,masks)
        proposed=LEVELS[a]
        nxt,commands,valid=batch_step(states,prev,proposed,tapes[:,step*5:(step+1)*5],packed,.5)
        if not valid.all():raise RuntimeError('Failed certificate during development')
        active=~finished
        costs+=active*(states[:,:,0]<c.goal).sum(-1)*c.decision_dt/3
        tv+=active*np.abs(commands-prev).sum(-1)
        accepted+=active*np.all(np.abs(commands-proposed)<1e-8,axis=1);cycles+=active
        proposals.extend(a[active].reshape(-1).tolist())
        finished|=np.all(nxt[:,:,0]>=c.goal,axis=1);states=nxt;prev=commands
        if finished.all():break
    return {'path':str(path),'n':n,'cost':float(costs.mean()),'variation':float(tv.mean()),'completed':int(finished.sum()),
            'joint_acceptance':float((accepted/cycles).mean()),'action_counts':np.bincount(proposals,minlength=len(LEVELS)).tolist(),
            'rows':[{'seed':18000001+i,'cost':float(costs[i]),'variation':float(tv[i]),'completed':bool(finished[i])} for i in range(n)]}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('paths',nargs='+');p.add_argument('--n',type=int,default=64);p.add_argument('--out')
    a=p.parse_args();results=[]
    for path in a.paths:
        result=evaluate(path,a.n);results.append(result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    if a.out:Path(a.out).write_text(json.dumps(results,indent=2)+'\n')
