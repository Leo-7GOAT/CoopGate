"""Sequential MAPPO; one policy update uses multi-step joint environment rollouts."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
import torch
from coopgate.marl_v12 import ActorCritic,load_policy
from coopgate.training_v12 import VectorMerge,SingleVectorMerge


def train(out,seed,steps=262144,n=32,rollout=64,variation_weight=.015,warmstart=None,single=False):
    out = Path(out); out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1); torch.manual_seed(seed); np.random.seed(seed)
    model = ActorCritic() if warmstart is None else load_policy(warmstart)[0]
    optimizer = torch.optim.Adam(model.parameters(),lr=3e-4,eps=1e-5)
    env = (SingleVectorMerge if single else VectorMerge)(n,seed+12000000,variation_weight); history=[]; episodes=[]; start=time.perf_counter()
    metadata={'algorithm':'single_ego_ppo_v12' if single else 'sequential_mappo_v12','seed':seed,'steps':0,'n_envs':n,'rollout':rollout,
        'gamma':.995,'gae_lambda':.95,'clip':.2,'epochs':4,'learning_rate':3e-4,
        'variation_weight':variation_weight,'training_backend':'vectorized exact longitudinal model',
        'warmstart':str(warmstart) if warmstart else None,'centralized_shield':True,'decentralized_actor':True,
        'learned_vehicles':[1] if single else [0,1,2],'finite_horizon_terminal':True,'potential_gamma':.995}
    torch.save({**metadata,'model':model.state_dict()},out/'initial.pt')
    total=0
    while total < steps:
        records=[]
        for k in range(min(rollout,(steps-total+n-1)//n)):
            obs,state,mask=env.inputs(); active=env.x[:,:,0]<env.c.goal
            if single: active=active[:,1:2]
            with torch.no_grad():
                dist=model.distribution(torch.tensor(obs),torch.tensor(mask))
                a=dist.sample(); lp=dist.log_prob(a); v=model.value(torch.tensor(state))
            nxt,r,terminal,boundary,finished=env.step(a.numpy()); episodes.extend(finished)
            with torch.no_grad(): nv=model.value(torch.tensor(nxt[1]))
            records.append((obs,state,mask,a.numpy(),lp.numpy(),v.numpy(),r,nv.numpy(),terminal,boundary,active))
            total+=n
        obs,state,mask,actions,old_lp,values,rewards,next_values,terminals,boundaries,active=map(np.asarray,zip(*records))
        advantages=np.zeros_like(rewards); carry=np.zeros(n)
        for k in range(len(records)-1,-1,-1):
            delta=rewards[k]+.995*(1-boundaries[k])*next_values[k]-values[k]
            carry=delta+.995*.95*(1-boundaries[k])*carry; advantages[k]=carry
        returns=advantages+values
        advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
        b=len(records)*n
        tensors=[torch.as_tensor(v.reshape((b,)+v.shape[2:])) for v in
                 (obs,state,mask,actions,old_lp,advantages,returns,active)]
        ob,st,ma,ac,ol,ad,re,act=tensors
        for epoch in range(4):
            for ids in torch.randperm(b).split(256):
                dist=model.distribution(ob[ids],ma[ids]); lp=dist.log_prob(ac[ids])
                ratio=(lp-ol[ids]).exp(); adv=ad[ids,None]
                weight=act[ids].float(); denominator=weight.sum().clamp_min(1)
                actor=-(torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv)*weight).sum()/denominator
                entropy=(dist.entropy()*weight).sum()/denominator
                critic=(model.value(st[ids])-re[ids]).square().mean()
                loss=actor+.5*critic-.02*entropy
                optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),.5); optimizer.step()
        recent=episodes[-100:]
        row={'steps':total,'episodes':env.episodes,'return_last100':float(np.mean([z['return'] for z in recent])) if recent else 0.,
            'cost_last100':float(np.mean([z['cost'] for z in recent])) if recent else 25.,
            'variation_last100':float(np.mean([z['variation'] for z in recent])) if recent else 0.,
            'completion_last100':float(np.mean([z['complete'] for z in recent])) if recent else 0.,
            'actor_loss':float(actor.detach()),'critic_loss':float(critic.detach()),'entropy':float(entropy.detach()),
            'elapsed_s':time.perf_counter()-start}
        history.append(row)
        if len(history)%16==0 or total>=steps:
            print(json.dumps({'seed':seed,**row}),flush=True)
            torch.save({**metadata,'steps':total,'model':model.state_dict()},out/f'step_{total}.pt')
        torch.save({**metadata,'steps':total,'model':model.state_dict()},out/'policy.pt')
        with (out/'training.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=row,lineterminator='\n');writer.writeheader();writer.writerows(history)
    (out/'metadata.json').write_text(json.dumps({**metadata,'steps':total,'episodes':env.episodes,'elapsed_s':time.perf_counter()-start},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=301)
    p.add_argument('--steps',type=int,default=262144);p.add_argument('--n',type=int,default=32)
    p.add_argument('--variation-weight',type=float,default=.015);p.add_argument('--warmstart')
    p.add_argument('--single',action='store_true')
    a=p.parse_args();train(a.out,a.seed,a.steps,a.n,variation_weight=a.variation_weight,warmstart=a.warmstart,single=a.single)
