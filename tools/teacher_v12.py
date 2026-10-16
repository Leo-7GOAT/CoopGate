"""Training-only teacher trajectories for an explicit imitation warm start."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from coopgate.merge_v4 import MergeConfig
from coopgate.experiment_v11 import prepare
from coopgate.learning_v6 import load_models
from coopgate.marl_v12 import observations,action_mask,LEVELS,ActorCritic,batch_step,central_state
from coopgate.fast_v8 import parameters
from develop_v12 import initial


def collect(out,n=256):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    c=MergeConfig();packed=parameters(c);models=load_models(Path('artifacts/v6'),[201,211,223])
    obs=[];states=[];masks=[];labels=[];rows=[];episode=[]
    for index in range(n):
        seed=20000001+index;regime='nominal' if index<n//2 else 'dense'
        x=initial(seed,regime);prev=np.zeros(3);gate,_=prepare('smooth_learned',models,x,prev,c,.5,seed)
        rng=np.random.default_rng(seed+1201);cost=0.;tv=0.
        for step in range(50):
            command,info=gate.select(x,prev,None)
            obs.append(observations(x,prev,step,c));states.append(central_state(x,prev,step,c));masks.append(action_mask(prev,c))
            labels.append(np.abs(LEVELS-command[:,None]).argmin(-1));episode.append(index)
            nxt,checked,valid=batch_step(x[None],prev[None],command[None],rng.uniform(-.5,.5,(1,5,3)),packed,.5)
            assert valid.all() and np.max(np.abs(checked[0]-command))<1e-10
            cost+=(x[:,0]<c.goal).sum()*c.decision_dt/3.;tv+=np.abs(command-prev).sum();x=nxt[0];prev=command
            if np.all(x[:,0]>=c.goal):break
        rows.append({'seed':seed,'regime':regime,'cost':cost,'variation':tv,'completed':bool(np.all(x[:,0]>=c.goal))})
        if (index+1)%16==0:print(json.dumps({'teacher_scenes':index+1,'steps':len(labels)}),flush=True)
    np.savez_compressed(out/'teacher.npz',obs=np.array(obs),state=np.array(states),mask=np.array(masks),actions=np.array(labels),episode=np.array(episode))
    (out/'metadata.json').write_text(json.dumps({'backend':'exact model, not SUMO','teacher':'frozen V11','n':n,'steps':len(labels),'rows':rows},indent=2)+'\n')


def warmstart(data,out,seed,epochs=35):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(seed)
    d=np.load(Path(data)/'teacher.npz');obs=torch.tensor(d['obs']);mask=torch.tensor(d['mask']);actions=torch.tensor(d['actions'])
    model=ActorCritic();opt=torch.optim.Adam(model.actor.parameters(),lr=1e-3);history=[]
    for epoch in range(epochs):
        losses=[]
        for ids in torch.randperm(len(obs)).split(256):
            dist=model.distribution(obs[ids],mask[ids]);loss=-dist.log_prob(actions[ids]).mean()
            opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
        history.append({'epoch':epoch+1,'cross_entropy':float(np.mean(losses))})
    algorithm='single_ego_ppo_v12' if obs.shape[1]==1 else 'sequential_mappo_v12'
    torch.save({'algorithm':algorithm,'seed':seed,'steps':0,'imitation_epochs':epochs,'model':model.state_dict()},out/'bc.pt')
    (out/'imitation.json').write_text(json.dumps({'history':history,'training_transitions':len(obs),'data':str(data)},indent=2)+'\n')
    print(json.dumps({'seed':seed,**history[-1]}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--n',type=int,default=256)
    p.add_argument('--data');p.add_argument('--seed',type=int,default=301);p.add_argument('--epochs',type=int,default=35)
    a=p.parse_args();warmstart(a.data,a.out,a.seed,a.epochs) if a.data else collect(a.out,a.n)
