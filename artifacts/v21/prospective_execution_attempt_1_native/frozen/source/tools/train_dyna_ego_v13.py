"""Complete-action Double DQN with explicitly counted model transitions.

Every visited state produces all admissible ego-action transitions through the
same shield and bounded-error dynamics. These are Bellman samples, not labels
from a planning teacher. No counterfactual simulation is used at inference.
"""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from coopgate.ego_v13 import EgoVector,ego_observations
from coopgate.marl_v12 import OBS_DIM,LEVELS,action_mask,batch_step


class DenseQPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(OBS_DIM+8,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
        self.value=nn.Linear(128,1);self.advantage=nn.Linear(128,len(LEVELS))

    def forward(self,obs):
        # Kinematic features are functions of the actor observation only.
        p=obs[...,0]*150.;v=obs[...,1]*20.
        own_eta=(-25.-p).clamp(min=0)/v.clamp(min=1.)
        features=[]
        for start in (6,12):
            present=obs[...,start];distance=obs[...,start+1]*150.
            neighbor_v=(v+obs[...,start+2]*20.).clamp(min=1.)
            eta=(-25.-p-distance).clamp(min=0)/neighbor_v
            features.extend([present*(eta-own_eta).clamp(-10,10)/10,
                present*distance/(v.clamp(min=1.)*10),present*(distance.abs()<35).float(),
                present*(eta<own_eta).float()])
        z=self.encoder(torch.cat((obs,torch.stack(features,dim=-1)),dim=-1));a=self.advantage(z)
        return self.value(z)+a-a.mean(-1,keepdim=True)

    @torch.no_grad()
    def proposal(self,obs,mask):
        return self(torch.as_tensor(obs,dtype=torch.float32)).masked_fill(~torch.as_tensor(mask,dtype=torch.bool),-1e9).argmax(-1).numpy()


def train(out,seed=601,steps=262144):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    online=DenseQPolicy();target=DenseQPolicy();target.load_state_dict(online.state_dict())
    opt=torch.optim.Adam(online.parameters(),lr=2e-4);n=32;env=EgoVector(n,seed+29000000,.05)
    capacity=262144;storage={k:np.empty((capacity,)+shape,dtype=dtype) for k,shape,dtype in
        [('obs',(OBS_DIM,),np.float32),('next',(OBS_DIM,),np.float32),('action',(),np.int64),('reward',(),np.float32),
         ('terminal',(),np.bool_),('next_mask',(len(LEVELS),),np.bool_)]}
    position=size=total=updates=model_samples=0;history=[];episodes=[];started=time.perf_counter()
    meta={'algorithm':'complete_action_double_dqn_ego_v13','seed':seed,'steps':0,'gamma':1.,'batch_size':512,
        'replay_capacity':capacity,'target_interval_updates':250,'updates_per_vector_step':4,'learned_vehicles':[1],
        'objective':'ego arrival + 0.05 ego variation until arrival','potential':'negative remaining ego distance / 20; terminal zero',
        'epsilon_start':1.,'epsilon_end':.03,'epsilon_decay_steps':131072,'training_backend':'exact longitudinal model',
        'observation_version':'markov_joint_memory_v1','sensor_range':150.,'inference':'network only; no model lookahead'}
    torch.save({**meta,'model':online.state_dict()},out/'initial.pt')
    while total<steps:
        obs,_,masks=env.inputs();obs=obs[:,0];mask=masks[:,0]
        rows,actions=np.nonzero(mask);count=len(rows)
        xx=env.x[rows].copy();prev=env.previous[rows].copy();clock=env.steps[rows].copy()
        joint_mask=action_mask(prev,env.c);joint=np.where(joint_mask,np.arange(len(LEVELS)),-1).max(-1);joint[:,1]=actions
        # The same sampled exogenous disturbance is reused across alternatives
        # at a state; it does not depend on the selected ego action.
        errors=rng.uniform(-.5,.5,(n,5,3))[rows]
        nxt,command,valid=batch_step(xx,prev,LEVELS[joint],errors,env.packed,.5)
        if not valid.all():raise RuntimeError('Counterfactual transition unsafe')
        boundary=(nxt[:,1,0]>=env.c.goal)|(clock+1>=50)
        phi=-np.maximum(env.c.goal-xx[:,1,0],0)/20.
        next_phi=-np.maximum(env.c.goal-nxt[:,1,0],0)/20.;next_phi[boundary]=0
        reward=-.5-.05*np.abs(command[:,1]-prev[:,1])+next_phi-phi
        ids=(np.arange(count)+position)%capacity
        values={'obs':obs[rows],'next':ego_observations(nxt,command,clock+1)[:,1],
            'action':actions,'reward':reward,'terminal':boundary,'next_mask':action_mask(command)[:,1]}
        for key,value in values.items():storage[key][ids]=value
        position=(position+count)%capacity;size=min(size+count,capacity);model_samples+=count
        epsilon=max(.03,1.-.97*total/131072)
        selected=online.proposal(obs,mask);random_actions=(rng.random(mask.shape)-np.where(mask,0.,2.)).argmax(-1)
        selected=np.where(rng.random(n)<epsilon,random_actions,selected)
        _,_,_,_,finished=env.step(selected[:,None]);episodes.extend(finished);total+=n
        if size>=4096:
            for _ in range(4):
                ids=rng.integers(0,size,512);ob=torch.tensor(storage['obs'][ids]);nb=torch.tensor(storage['next'][ids])
                nm=torch.tensor(storage['next_mask'][ids])
                with torch.no_grad():
                    best=online(nb).masked_fill(~nm,-1e9).argmax(-1)
                    bootstrap=target(nb).gather(1,best[:,None]).squeeze(-1)
                    expected=torch.tensor(storage['reward'][ids])+(~torch.tensor(storage['terminal'][ids]))*bootstrap
                predicted=online(ob).gather(1,torch.tensor(storage['action'][ids])[:,None]).squeeze(-1)
                loss=nn.functional.smooth_l1_loss(predicted,expected);opt.zero_grad();loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(),10.);opt.step();updates+=1
                if updates%250==0:target.load_state_dict(online.state_dict())
        if total%32768==0 or total>=steps:
            recent=episodes[-100:];row={'steps':total,'counterfactual_model_transitions':model_samples,'updates':updates,
                'episodes':env.episodes,'epsilon':epsilon,'ego_time_last100':float(np.mean([r['cost'] for r in recent])),
                'ego_variation_last100':float(np.mean([r['variation'] for r in recent])),
                'loss':float(loss.detach()),'elapsed_s':time.perf_counter()-started};history.append(row)
            print(json.dumps(row),flush=True)
            ckpt={**meta,'steps':total,'counterfactual_model_transitions':model_samples,'model':online.state_dict()}
            torch.save(ckpt,out/'policy.pt');torch.save(ckpt,out/f'step_{total}.pt')
            with (out/'training.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=row,lineterminator='\n');w.writeheader();w.writerows(history)
    (out/'metadata.json').write_text(json.dumps({**meta,'steps':total,'counterfactual_model_transitions':model_samples,
        'gradient_updates':updates,'episodes':env.episodes},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=601)
    p.add_argument('--steps',type=int,default=262144);a=p.parse_args();train(a.out,a.seed,a.steps)
