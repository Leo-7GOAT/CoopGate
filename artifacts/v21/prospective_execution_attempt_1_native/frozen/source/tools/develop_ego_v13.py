"""Ego-specific development set, disjoint from all V12 scenes."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from coopgate.ego_v13 import load_ego,EgoRolloutGate,ego_observations
from coopgate.marl_v12 import observations,action_mask,LEVELS,batch_step
from coopgate.disturbance_v9 import disturbance_tape
from coopgate.merge_v4 import MergeConfig
from coopgate.fast_v8 import parameters
from develop_v12 import initial


def evaluate(path,n=96):
    c=MergeConfig();packed=parameters(c);model=None;markov=False
    augmented=path.startswith('augmented:')
    if path not in ('rule','rollout') and not augmented:
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        markov=checkpoint.get('observation_version')=='markov_joint_memory_v1'
        if checkpoint['algorithm']=='complete_action_double_dqn_ego_v13':
            from train_dyna_ego_v13 import DenseQPolicy
            torch.set_num_threads(1);model=DenseQPolicy();model.load_state_dict(checkpoint['model']);model.eval()
        elif checkpoint['algorithm']=='double_dqn_ego_v13':
            from train_dqn_ego_v13 import QPolicy
            torch.set_num_threads(1);model=QPolicy();model.load_state_dict(checkpoint['model']);model.eval()
        else:model=load_ego(path)[0]
    x=np.array([initial(25000001+i,'nominal' if i<n//2 else 'dense') for i in range(n)]);prev=np.zeros((n,3))
    finish=np.zeros((n,3));variation=np.zeros((n,3));complete=np.zeros((n,3),bool)
    tapes=np.array([disturbance_tape(25000001+i,['clean','uniform','bias','pulse'][i%4],.5,250) for i in range(n)])
    gates=[EgoRolloutGate() for _ in range(n)] if path=='rollout' else None
    if augmented:
        from coopgate.learned_rollout_v13 import LearnedRolloutGate
        gates=[LearnedRolloutGate(path.split(':',1)[1]) for _ in range(n)]
    for step in range(50):
        mask=action_mask(prev);a=np.where(mask,np.arange(len(LEVELS)),-1).max(-1)
        if model is not None:a[:,1]=model.proposal((ego_observations if markov else observations)(x,prev,step)[:,1],mask[:,1])
        proposed=LEVELS[a]
        if gates is not None:proposed=np.array([gate.select(s,p)[0] for gate,s,p in zip(gates,x,prev)])
        nxt,commands,valid=batch_step(x,prev,proposed,tapes[:,step*5:(step+1)*5],packed,.5)
        if not valid.all():raise RuntimeError('Invalid development transition')
        finish+=~complete*.5;variation+=(~complete)*np.abs(commands-prev)
        complete|=nxt[:,:,0]>=c.goal;x=nxt;prev=commands
        if complete.all():break
    return {'path':str(path),'n':n,'ego_time':float(finish[:,1].mean()),'ego_variation':float(variation[:,1].mean()),
        'ego_cost':float((finish[:,1]+.05*variation[:,1]).mean()),'other_time':float(finish[:,[0,2]].mean()),
        'team_time':float(finish.mean()),'completed':int(complete.all(-1).sum()),
        'rows':[{'seed':25000001+i,'mode':['clean','uniform','bias','pulse'][i%4],
            'finish':finish[i].tolist(),'variation':variation[i].tolist(),'completed':bool(complete[i].all())} for i in range(n)]}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('paths',nargs='+');p.add_argument('--out');p.add_argument('--n',type=int,default=96);a=p.parse_args()
    results=[]
    for path in a.paths:r=evaluate(path,a.n);results.append(r);print(json.dumps({k:v for k,v in r.items() if k!='rows'}),flush=True)
    if a.out:Path(a.out).write_text(json.dumps(results,indent=2)+'\n')
