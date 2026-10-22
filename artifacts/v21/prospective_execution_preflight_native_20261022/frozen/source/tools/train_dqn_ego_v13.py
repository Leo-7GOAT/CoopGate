"""Off-policy Double DQN ego control with masks and simulator transitions."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from coopgate.ego_v13 import EgoVector
from coopgate.marl_v12 import OBS_DIM,LEVELS
from coopgate.ego_models_v13 import QPolicy


def train(out,seed=601,steps=2097152):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed)
    rng=np.random.default_rng(seed);online=QPolicy();target=QPolicy();target.load_state_dict(online.state_dict())
    opt=torch.optim.Adam(online.parameters(),lr=3e-4);n=32;env=EgoVector(n,seed+28000000,.05)
    capacity=262144;storage={k:np.empty((capacity,)+shape,dtype=dtype) for k,shape,dtype in
        [('obs',(OBS_DIM,),np.float32),('next',(OBS_DIM,),np.float32),('action',(),np.int64),('reward',(),np.float32),
         ('terminal',(),np.bool_),('next_mask',(len(LEVELS),),np.bool_)]}
    position=size=total=updates=0;history=[];episodes=[];started=time.perf_counter()
    meta={'algorithm':'double_dqn_ego_v13','seed':seed,'steps':0,'gamma':.995,'batch_size':256,'replay_capacity':capacity,
        'target_interval_updates':250,'learned_vehicles':[1],'objective':'ego arrival + 0.05 ego variation until arrival',
        'epsilon_start':1.,'epsilon_end':.03,'epsilon_decay_steps':600000,'training_backend':'exact longitudinal model',
        'observation_version':'markov_joint_memory_v1','sensor_range':150.}
    torch.save({**meta,'model':online.state_dict()},out/'initial.pt')
    while total<steps:
        obs,state,mask=env.inputs();obs=obs[:,0];mask=mask[:,0]
        epsilon=max(.03,1.-.97*total/600000)
        actions=online.proposal(obs,mask)
        random_actions=(rng.random(mask.shape)-np.where(mask,0.,2.)).argmax(-1)
        explore=rng.random(n)<epsilon;actions=np.where(explore,random_actions,actions)
        nxt,reward,terminal,boundary,finished=env.step(actions[:,None]);episodes.extend(finished)
        ids=(np.arange(n)+position)%capacity
        for key,value in [('obs',obs),('next',nxt[0][:,0]),('action',actions),('reward',reward),('terminal',boundary),('next_mask',nxt[2][:,0])]:
            storage[key][ids]=value
        position=(position+n)%capacity;size=min(size+n,capacity);total+=n
        if size>=4096:
            ids=rng.integers(0,size,256)
            ob=torch.tensor(storage['obs'][ids]);nb=torch.tensor(storage['next'][ids]);nm=torch.tensor(storage['next_mask'][ids])
            with torch.no_grad():
                best=online(nb).masked_fill(~nm,-1e9).argmax(-1)
                bootstrap=target(nb).gather(1,best[:,None]).squeeze(-1)
                expected=torch.tensor(storage['reward'][ids])+.995*(~torch.tensor(storage['terminal'][ids]))*bootstrap
            predicted=online(ob).gather(1,torch.tensor(storage['action'][ids])[:,None]).squeeze(-1)
            loss=nn.functional.smooth_l1_loss(predicted,expected);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(online.parameters(),10.);opt.step()
            updates+=1
            if updates%250==0:target.load_state_dict(online.state_dict())
        if total%32768==0 or total>=steps:
            recent=episodes[-100:];row={'steps':total,'episodes':env.episodes,'epsilon':epsilon,
                'ego_time_last100':float(np.mean([r['cost'] for r in recent])) if recent else 25.,
                'ego_variation_last100':float(np.mean([r['variation'] for r in recent])) if recent else 0.,
                'loss':float(loss.detach()) if updates else 0.,'elapsed_s':time.perf_counter()-started};history.append(row)
            torch.save({**meta,'steps':total,'model':online.state_dict()},out/'policy.pt')
            if total%262144==0 or total>=steps:
                print(json.dumps(row),flush=True);torch.save({**meta,'steps':total,'model':online.state_dict()},out/f'step_{total}.pt')
            with (out/'training.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=row,lineterminator='\n');w.writeheader();w.writerows(history)
    (out/'metadata.json').write_text(json.dumps({**meta,'steps':total,'episodes':env.episodes},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=601)
    p.add_argument('--steps',type=int,default=2097152);a=p.parse_args();train(a.out,a.seed,a.steps)
