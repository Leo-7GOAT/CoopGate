"""Separate V4 policies; shared actor / centralized critic, shield in training."""
from pathlib import Path
import csv,json,random
import numpy as np
import torch
from torch import nn
from .learning_v3 import PlatoonPolicy
from .mappo import compute_gae
from .merge_v4 import MergeConfig,MergeEnv,ACTIONS


def load(path):
    torch.set_num_threads(1);meta=torch.load(path,map_location='cpu',weights_only=True)
    if meta.get('benchmark')!='merge_v4':raise ValueError('Not a V4 checkpoint')
    model=PlatoonPolicy();model.load_state_dict(meta['model']);model.eval();return model,meta


def train(out,seed,steps=6000,bc_episodes=20):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'policy.pt').exists():raise RuntimeError('Refusing to overwrite')
    torch.set_num_threads(1);torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    rng=np.random.default_rng(seed);model=PlatoonPolicy();cfg=MergeConfig();data=[];labels=[]
    meta={'benchmark':'merge_v4','seed':seed,'steps':0,'bc_episodes':bc_episodes,'config':cfg.dict()}
    with MergeEnv(cfg,'constant') as env:
        for episode in range(bc_episodes):
            obs=env.reset(3000000+seed*1000+episode,'nominal')
            while True:
                old=obs;obs,_,t,u,_=env.step([4]*3)
                data.extend(old);labels.extend(np.abs(np.array(env.trace[-1]['command'])[:,None]-ACTIONS).argmin(axis=1))
                if t or u:break
    x=torch.tensor(np.array(data));y=torch.tensor(labels)
    opt=torch.optim.Adam(model.actor.parameters(),lr=1e-3)
    for _ in range(15):
        for ids in torch.randperm(len(y)).split(256):
            loss=nn.functional.cross_entropy(model.actor(x[ids]),y[ids]);opt.zero_grad();loss.backward();opt.step()
    torch.save({**meta,'model':model.state_dict()},out/'bc.pt')
    print(json.dumps({'seed':seed,'bc_samples':len(y),'bc_loss':float(loss.detach())}),flush=True)
    opt=torch.optim.Adam(model.parameters(),lr=3e-4);total=0;episodes=0;returns=[];episode_return=0;rows=[]
    with MergeEnv(cfg,'joint_rl') as env:
        obs=env.reset(4000000+seed*1000)
        while total<steps:
            batch={k:[] for k in ('obs','a','lp','r','v','nv','terminal','boundary')}
            for _ in range(min(256,steps-total)):
                a,lp,v=model.act(obs);nxt,r,t,u,m=env.step(a)
                if m['outside_steps'] or m['uncertified_steps'] or m['contract_violations'] or m['collision'] or m['max_prediction_error']>1e-6:
                    raise RuntimeError('Training contract failed: '+str(m))
                with torch.no_grad():nv=float(model.value(torch.as_tensor(nxt)))
                for key,value in zip(batch,(obs,a,lp,r,v,nv,t,t or u)):batch[key].append(value)
                total+=1;obs=nxt;episode_return+=r
                if t or u:
                    episodes+=1;returns.append(episode_return);episode_return=0;obs=env.reset(4000000+seed*1000+episodes)
            b={k:np.asarray(v) for k,v in batch.items()}
            adv,target=compute_gae(b['r'],b['v'],b['nv'],b['terminal'],b['boundary']);adv=(adv-adv.mean())/(adv.std()+1e-8)
            obs_t=torch.tensor(b['obs']);actions=torch.tensor(b['a']);old_lp=torch.tensor(b['lp']);adv=torch.tensor(adv);target=torch.tensor(target,dtype=torch.float32)
            for _ in range(4):
                for ids in torch.randperm(len(adv)).split(64):
                    dist=model.distribution(obs_t[ids]);ratio=(dist.log_prob(actions[ids])-old_lp[ids]).exp()
                    actor=-torch.minimum(ratio*adv[ids,None],ratio.clamp(.8,1.2)*adv[ids,None]).mean()
                    critic=(model.value(obs_t[ids])-target[ids]).square().mean();loss=actor+.5*critic-.01*dist.entropy().mean()
                    opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),.5);opt.step()
            rows.append({'steps':total,'episodes':episodes,'return_last20':float(np.mean(returns[-20:])) if returns else 0.,'actor_loss':float(actor.detach()),'critic_loss':float(critic.detach())})
            torch.save({**meta,'steps':total,'model':model.state_dict()},out/'policy.pt')
            with (out/'training.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=rows[-1]);w.writeheader();w.writerows(rows)
            if len(rows)%4==0 or total==steps:print(json.dumps({'seed':seed,**rows[-1]}),flush=True)
    (out/'config.json').write_text(json.dumps({**meta,'steps':steps},indent=2),newline='\n')
