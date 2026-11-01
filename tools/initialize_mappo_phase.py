"""Cost-sensitive startup expert INITIALIZATION, explicitly not MAPPO updates.

Fixed planning targets never use a changing learned policy. Subsequent MAPPO
must collect its own stochastic trajectories; this dataset is not PPO replay.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from numba import njit
import torch

from coopgate.ego_v13 import ego_observations,ego_tail
from coopgate.fast_v8 import parameters,advance_one
from coopgate.mappo_ctde import MAPPOCTDE,PPOConfig,load_mappo
from coopgate.marl_v12 import LEVELS,action_mask,project_joint
from coopgate.merge_v4 import MergeConfig
from develop_v12 import initial


@njit(cache=True,fastmath=False)
def initial_action_costs(state,packed):
    previous=np.zeros(3)
    result=np.full(len(LEVELS),np.inf)
    transitions=0
    for j in range(len(LEVELS)):
        if abs(LEVELS[j])>2.5+1e-8:
            continue
        proposal=np.full(3,2.5);proposal[1]=LEVELS[j]
        command,_,ok=project_joint(state,previous,proposal,packed,.5)
        if not ok:
            raise RuntimeError('Expert candidate has no physical certificate')
        following=state.copy()
        if not advance_one(following,command,packed):
            raise RuntimeError('Expert candidate nominal physical failure')
        tail,elapsed,_=ego_tail(following,command,1,packed,.5,.05)
        transitions+=1+int(round(elapsed/.5))
        result[j]=.5+.05*abs(command[1])+tail
    return result,transitions


def initialize(out,parent,n=8192,seed=4506):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(seed)
    source,old=load_mappo(parent)
    if old['stage']!='single' or old['steps']!=old['target_steps'] or source.config.phase_actor:
        raise ValueError('Completed un-routed on-policy single PPO parent required')
    cfg=PPOConfig(kinematic_features=True,accelerated_features=True,phase_actor=True,
                  entropy=.005,gae_lambda=1.,actor_lr=3e-5,initial_actor_lr=3e-4)
    model=MAPPOCTDE(cfg)
    model.actor.initial.load_state_dict(source.actor.state_dict())
    model.actor.flow.load_state_dict(source.actor.state_dict())
    plan=dict(algorithm='planner_bc_initializer_20261101',
        initialization_method='local-clock phase heads, startup cost-sensitive fixed expert',
        parent_ppo=dict(path=str(parent),sha256=hashlib.sha256(Path(parent).read_bytes()).hexdigest()),
        ppo_config=asdict(cfg),seed=seed,source_seed_base=951000001,initials=n,
        source='fresh alternating nominal/dense starts; fixed nominal one-step plus rule-tail costs',
        cost_target='same objective as original strong ego planner first-action search',
        temperature_seconds=.25,epochs=400,bc_lr=3e-4,
        flow_head='frozen exact copy of completed parent PPO',
        meaning='Supervised initialization only; not PPO, no online planning',
        subsequent_ppo_data='must be freshly collected; expert dataset not read by PPO updates')
    (out/'protocol.json').write_text(json.dumps(plan,indent=2)+'\n')
    observations=[];masks=[];costs=[];teacher_transitions=0
    packed=parameters(MergeConfig())
    for i in range(n):
        state=initial(951000001+i,'nominal' if i%2==0 else 'dense')
        observations.append(ego_observations(state,np.zeros(3),0)[1])
        masks.append(action_mask(np.zeros(3))[1])
        q,work=initial_action_costs(state,packed)
        costs.append(q);teacher_transitions+=work
    obs=torch.tensor(np.array(observations));mask=torch.tensor(np.array(masks))
    q=torch.tensor(np.array(costs),dtype=torch.float32)
    targets=torch.softmax((-(q-q.min(-1,keepdim=True).values)/.25).masked_fill(~mask,-1e9),dim=-1)
    torch.save(dict(obs=obs,mask=mask,costs=q,targets=targets,protocol=plan),out/'dataset.pt')
    optimizer=torch.optim.Adam(model.actor.initial.parameters(),lr=3e-4)
    for epoch in range(400):
        for ids in torch.randperm(n).split(512):
            dist=model.distribution(obs[ids],mask[ids])
            loss=-(targets[ids]*dist.logits).sum(-1).mean()
            optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor.initial.parameters(),.5)
            optimizer.step()
    with torch.no_grad():
        prediction=model.distribution(obs,mask).logits.argmax(-1)
        regret=q.gather(1,prediction[:,None]).squeeze(-1)-q.min(-1).values
    checkpoint=dict(**plan,model=dict(actor=model.actor.state_dict()),
        dataset_sha256=hashlib.sha256((out/'dataset.pt').read_bytes()).hexdigest())
    torch.save(checkpoint,out/'initializer.pt')
    report=dict(training_initials=n,fixed_candidate_evaluations=int(mask.sum()),epochs=400,
        expert_counterfactual_joint_decisions=teacher_transitions,
        train_greedy_nominal_regret_s=float(regret.mean()),
        train_regret_above_one_second=int((regret>1.).sum()),
        initializer_sha256=hashlib.sha256((out/'initializer.pt').read_bytes()).hexdigest(),
        held_out_efficacy_claim=False)
    (out/'complete.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True)
    parser.add_argument('--parent',required=True);parser.add_argument('--n',type=int,default=8192)
    parser.add_argument('--seed',type=int,default=4506)
    args=parser.parse_args();initialize(args.out,args.parent,args.n,args.seed)
