"""Only start true MARL after the frozen single-vehicle gate has passed.

MAPPO uses one centralized state critic. Matched IPPO uses a local-observation
critic. Both use a shared decentralized actor, identical rewards, demonstrations,
seeds, interaction budgets, action masks and physical safety filter.
"""
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from coopgate.marl_v12 import ActorCritic,mlp,OBS_DIM,load_policy
from coopgate.training_v12 import VectorMerge
from teacher_v12 import warmstart
from train_v12 import train

ROOT=Path(__file__).resolve().parents[1]


def train_ippo(out,seed,warm,steps=2097152):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed)
    model=load_policy(warm)[0];model.critic=mlp(OBS_DIM,1)
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4,eps=1e-5)
    n=32;env=VectorMerge(n,seed+12000000,.05);history=[];episodes=[];total=0;started=time.perf_counter()
    meta={'algorithm':'sequential_ippo_v13','seed':seed,'steps':0,'n_envs':n,'rollout':64,
        'gamma':.995,'gae_lambda':.95,'clip':.2,'epochs':4,'learning_rate':3e-4,'variation_weight':.05,
        'training_backend':'vectorized exact longitudinal model','learned_vehicles':[0,1,2],
        'warmstart':str(warm),'critic_inputs':'local per-vehicle observation only','actor_parameter_sharing':True}
    torch.save({**meta,'model':model.state_dict()},out/'initial.pt')
    while total<steps:
        records=[]
        for k in range(min(64,(steps-total+n-1)//n)):
            obs,state,mask=env.inputs();active=env.x[:,:,0]<env.c.goal
            with torch.no_grad():
                dist=model.distribution(torch.tensor(obs),torch.tensor(mask));a=dist.sample()
                lp=dist.log_prob(a);v=model.value(torch.tensor(obs))
            nxt,r,terminal,boundary,finished=env.step(a.numpy());episodes.extend(finished)
            with torch.no_grad():nv=model.value(torch.tensor(nxt[0]))
            records.append((obs,mask,a.numpy(),lp.numpy(),v.numpy(),r,nv.numpy(),boundary,active));total+=n
        obs,mask,actions,old_lp,values,rewards,next_values,boundaries,active=map(np.asarray,zip(*records))
        advantages=np.zeros_like(values);carry=np.zeros((n,3))
        for k in range(len(records)-1,-1,-1):
            delta=rewards[k,:,None]+.995*(1-boundaries[k,:,None])*next_values[k]-values[k]
            carry=delta+.995*.95*(1-boundaries[k,:,None])*carry;advantages[k]=carry
        returns=advantages+values;advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
        b=len(records)*n
        ob,ma,ac,ol,ad,re,act=[torch.as_tensor(v.reshape((b,)+v.shape[2:])) for v in
            (obs,mask,actions,old_lp,advantages,returns,active)]
        for epoch in range(4):
            for ids in torch.randperm(b).split(256):
                dist=model.distribution(ob[ids],ma[ids]);ratio=(dist.log_prob(ac[ids])-ol[ids]).exp()
                weight=act[ids].float();denom=weight.sum().clamp_min(1)
                actor=-(torch.minimum(ratio*ad[ids],ratio.clamp(.8,1.2)*ad[ids])*weight).sum()/denom
                entropy=(dist.entropy()*weight).sum()/denom
                critic=(model.value(ob[ids])-re[ids]).square().mean();loss=actor+.5*critic-.02*entropy
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step()
        recent=episodes[-100:]
        row={'steps':total,'episodes':env.episodes,'return_last100':float(np.mean([r['return'] for r in recent])) if recent else 0.,
            'cost_last100':float(np.mean([r['cost'] for r in recent])) if recent else 25.,
            'variation_last100':float(np.mean([r['variation'] for r in recent])) if recent else 0.,
            'actor_loss':float(actor.detach()),'critic_loss':float(critic.detach()),'entropy':float(entropy.detach()),
            'elapsed_s':time.perf_counter()-started};history.append(row)
        if len(history)%64==0 or total>=steps:print(json.dumps({'algorithm':'IPPO','seed':seed,**row}),flush=True)
        torch.save({**meta,'steps':total,'model':model.state_dict()},out/'policy.pt')
        with (out/'training.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=row,lineterminator='\n');writer.writeheader();writer.writerows(history)
    (out/'metadata.json').write_text(json.dumps({**meta,'steps':total,'episodes':env.episodes},indent=2)+'\n')


def run(out,data,steps=2097152):
    single=json.loads((ROOT/'artifacts/v12/single/summary.json').read_text())
    if not single['acceptance']['all_passed'] or not single.get('individual_ego_verified',False):
        raise RuntimeError('Individual ego superiority has not passed; team averages are insufficient')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    protocol={'seeds':[511,521,531],'steps_per_seed_per_algorithm':steps,'bc_epochs':80,'variation_weight':.05,
        'algorithms':['MAPPO','IPPO'],'selection':'fixed final checkpoint, every seed retained',
        'single_agent_prerequisite_sha256':hashlib.sha256((ROOT/'artifacts/v12/single/summary.json').read_bytes()).hexdigest(),
        'teacher_sha256':hashlib.sha256((Path(data)/'teacher.npz').read_bytes()).hexdigest(),
        'source_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__).resolve(),ROOT/'tools/train_v12.py',ROOT/'tools/teacher_v12.py',ROOT/'src/coopgate/marl_v12.py',ROOT/'src/coopgate/training_v12.py']}}
    (out/'training_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for seed in protocol['seeds']:
        warmstart(data,out/f'bc_{seed}',seed,80)
        train(out/f'mappo_{seed}',seed,steps,variation_weight=.05,warmstart=out/f'bc_{seed}/bc.pt',single=False)
        train_ippo(out/f'ippo_{seed}',seed,out/f'bc_{seed}/bc.pt',steps)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--data',default='runs/v12_teacher')
    p.add_argument('--steps',type=int,default=2097152);a=p.parse_args();run(a.out,a.data,a.steps)
