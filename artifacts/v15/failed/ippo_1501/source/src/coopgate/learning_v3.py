"""Shield-aware shared-actor PPO for the scoped longitudinal benchmark."""
from pathlib import Path
import csv,json,random
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from .mappo import network,compute_gae
from .platoon import PlatoonEnv,PLATOON_OBS,ACCEL_ACTIONS
from .recovery import RecoveryConfig


class PlatoonPolicy(nn.Module):
    def __init__(self):
        super().__init__();self.actor=network(PLATOON_OBS,len(ACCEL_ACTIONS));self.critic=network(3*PLATOON_OBS,1)
    def distribution(self,obs):return Categorical(logits=self.actor(obs))
    def value(self,obs):return self.critic(obs.flatten(start_dim=-2)).squeeze(-1)
    @torch.no_grad()
    def act(self,obs,deterministic=False):
        obs=torch.as_tensor(obs,dtype=torch.float32);d=self.distribution(obs)
        a=d.probs.argmax(-1) if deterministic else d.sample()
        return a.numpy(),d.log_prob(a).numpy(),float(self.value(obs))


def load(path):
    torch.set_num_threads(1)
    meta=torch.load(path,map_location='cpu',weights_only=True)
    model=PlatoonPolicy();model.load_state_dict(meta['model']);model.eval();return model,meta


def train(out,seed,steps=12000,bc_episodes=30):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'policy.pt').exists():raise RuntimeError('Use a new output directory')
    torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    rng=np.random.default_rng(seed);cfg=RecoveryConfig();model=PlatoonPolicy()
    data=[];labels=[]
    with PlatoonEnv(cfg,'joint_mpc') as env:
        for k in range(bc_episodes):
            obs=env.reset(1000000+seed*1000+k,('cruise','moderate','hard_brake')[k%3])
            while True:
                old=obs;obs,_,t,u,_=env.step([2,2,2]);a=np.array(env.trace[-1]['commands'])
                data.extend(old);labels.extend(np.argmin(np.abs(a[:,None]-ACCEL_ACTIONS),axis=1))
                if t or u:break
    x=torch.tensor(np.asarray(data),dtype=torch.float32);y=torch.tensor(labels,dtype=torch.long)
    optimizer=torch.optim.Adam(model.actor.parameters(),lr=1e-3)
    for _ in range(15):
        for ids in torch.randperm(len(y)).split(256):
            loss=nn.functional.cross_entropy(model.actor(x[ids]),y[ids]);optimizer.zero_grad();loss.backward();optimizer.step()
    torch.save({'model':model.state_dict(),'seed':seed,'steps':0,'config':cfg.dict()},out/'bc.pt')
    print(json.dumps({'seed':seed,'bc_samples':len(y),'bc_loss':float(loss.detach())}),flush=True)
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    rows=[];returns=[];total=0;episodes=0;episode_return=0
    with PlatoonEnv(cfg,'joint_rl') as env:
        obs=env.reset(2000000+seed*1000,'cruise')
        while total<steps:
            batch={k:[] for k in ('obs','a','lp','r','v','nv','terminal','boundary')}
            for _ in range(min(256,steps-total)):
                a,lp,v=model.act(obs);nxt,r,t,u,info=env.step(a)
                if info['outside_set_steps'] or info['uncertified_action_steps'] or info['max_prediction_error']>1e-5:
                    raise RuntimeError('Training contract violated: '+str(info))
                with torch.no_grad():nv=float(model.value(torch.as_tensor(nxt)))
                for key,value in zip(batch,(obs,a,lp,r,v,nv,t,t or u)):batch[key].append(value)
                obs=nxt;episode_return+=r;total+=1
                if t or u:
                    episodes+=1;returns.append(episode_return);episode_return=0
                    obs=env.reset(2000000+seed*1000+episodes,str(rng.choice(['cruise','moderate','hard_brake'])))
            b={k:np.asarray(v) for k,v in batch.items()}
            advantage,target=compute_gae(b['r'],b['v'],b['nv'],b['terminal'],b['boundary'])
            advantage=(advantage-advantage.mean())/(advantage.std()+1e-8)
            obs_t=torch.tensor(b['obs']);actions=torch.tensor(b['a']);old_lp=torch.tensor(b['lp'])
            adv=torch.tensor(advantage);target=torch.tensor(target,dtype=torch.float32)
            for _ in range(4):
                for ids in torch.randperm(len(adv)).split(64):
                    dist=model.distribution(obs_t[ids]);ratio=(dist.log_prob(actions[ids])-old_lp[ids]).exp()
                    actor=-torch.minimum(ratio*adv[ids,None],ratio.clamp(.8,1.2)*adv[ids,None]).mean()
                    critic=(model.value(obs_t[ids])-target[ids]).square().mean()
                    loss=actor+.5*critic-.01*dist.entropy().mean()
                    optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step()
            rows.append({'steps':total,'episodes':episodes,'return_last20':float(np.mean(returns[-20:])) if returns else 0,
                         'actor_loss':float(actor.detach()),'critic_loss':float(critic.detach())})
            if len(rows)%5==0 or total==steps:print(json.dumps({'seed':seed,**rows[-1]}),flush=True)
            torch.save({'model':model.state_dict(),'seed':seed,'steps':total,'bc_episodes':bc_episodes,'config':cfg.dict(),
                        'training_controller':'joint_rl','intervention_penalty':.1},out/'policy.pt')
            with (out/'training.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=rows[-1]);writer.writeheader();writer.writerows(rows)
    (out/'config.json').write_text(json.dumps({'seed':seed,'steps':steps,'bc_episodes':bc_episodes,'config':cfg.dict()},indent=2),newline='\n')
