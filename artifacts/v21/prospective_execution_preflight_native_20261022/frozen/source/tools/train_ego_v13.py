"""PPO for the individual ego arrival/variation objective."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
import torch
from coopgate.marl_v12 import ActorCritic
from coopgate.ego_v13 import EgoVector,load_ego


def train(out,seed,steps=1048576,warmstart=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed)
    model=ActorCritic() if warmstart is None else load_ego(warmstart)[0]
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4,eps=1e-5);n=32
    env=EgoVector(n,seed+26000000,.05);history=[];episodes=[];started=time.perf_counter();total=0
    meta={'algorithm':'single_ego_ppo_v13','seed':seed,'steps':0,'learned_vehicles':[1],
        'objective':'ego arrival time + 0.05 ego variation until ego arrival','n_envs':32,'rollout':64,
        'gamma':.995,'gae_lambda':.95,'epochs':4,'learning_rate':3e-4,'clip':.2,'variation_weight':.05,
        'training_backend':'exact longitudinal model','warmstart':str(warmstart),'ego_goal_is_terminal':True,
        'observation_version':'markov_joint_memory_v1','sensor_range':150.}
    torch.save({**meta,'model':model.state_dict()},out/'initial.pt')
    while total<steps:
        records=[]
        for k in range(min(64,(steps-total+n-1)//n)):
            obs,state,mask=env.inputs()
            with torch.no_grad():
                dist=model.distribution(torch.tensor(obs),torch.tensor(mask));actions=dist.sample();lp=dist.log_prob(actions);v=model.value(torch.tensor(state))
            nxt,r,terminal,boundary,finished=env.step(actions.numpy());episodes.extend(finished)
            with torch.no_grad():nv=model.value(torch.tensor(nxt[1]))
            records.append((obs,state,mask,actions.numpy(),lp.numpy(),v.numpy(),r,nv.numpy(),boundary));total+=n
        obs,state,mask,actions,oldlp,values,rewards,nextvalues,boundaries=map(np.asarray,zip(*records))
        adv=np.zeros_like(rewards);carry=np.zeros(n)
        for k in range(len(records)-1,-1,-1):
            delta=rewards[k]+.995*(1-boundaries[k])*nextvalues[k]-values[k]
            carry=delta+.995*.95*(1-boundaries[k])*carry;adv[k]=carry
        returns=adv+values;adv=(adv-adv.mean())/(adv.std()+1e-8);b=len(records)*n
        ob,st,ma,ac,ol,ad,re=[torch.as_tensor(z.reshape((b,)+z.shape[2:])) for z in (obs,state,mask,actions,oldlp,adv,returns)]
        for epoch in range(4):
            for ids in torch.randperm(b).split(256):
                dist=model.distribution(ob[ids],ma[ids]);ratio=(dist.log_prob(ac[ids])-ol[ids]).exp()
                actor=-torch.minimum(ratio*ad[ids,None],ratio.clamp(.8,1.2)*ad[ids,None]).mean()
                critic=(model.value(st[ids])-re[ids]).square().mean();entropy=dist.entropy().mean()
                loss=actor+.5*critic-.02*entropy;optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step()
        recent=episodes[-100:]
        row={'steps':total,'episodes':env.episodes,'ego_time_last100':float(np.mean([r['cost'] for r in recent])) if recent else 25.,
            'ego_variation_last100':float(np.mean([r['variation'] for r in recent])) if recent else 0.,
            'actor_loss':float(actor.detach()),'critic_loss':float(critic.detach()),'entropy':float(entropy.detach()),'elapsed_s':time.perf_counter()-started}
        history.append(row)
        if len(history)%32==0 or total>=steps:
            print(json.dumps({'seed':seed,**row}),flush=True);torch.save({**meta,'steps':total,'model':model.state_dict()},out/f'step_{total}.pt')
        torch.save({**meta,'steps':total,'model':model.state_dict()},out/'policy.pt')
        with (out/'training.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=row,lineterminator='\n');w.writeheader();w.writerows(history)
    (out/'metadata.json').write_text(json.dumps({**meta,'steps':total,'episodes':env.episodes},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=601)
    p.add_argument('--steps',type=int,default=1048576);p.add_argument('--warmstart');a=p.parse_args();train(a.out,a.seed,a.steps,a.warmstart)
