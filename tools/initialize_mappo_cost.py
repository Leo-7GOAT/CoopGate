"""Expected fixed-expert regret initialization; subsequently evaluated separately.

This is a supervised/model-based initializer, never a substitute for on-policy
PPO. Reuses the full audited F training table without any new teacher queries.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from coopgate.mappo_ctde import MAPPOCTDE,PPOConfig,load_mappo


def initialize(out,parent,dataset,seed=4507,loss_kind='expected'):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(seed)
    source,meta=load_mappo(parent)
    if not source.config.phase_actor or meta['steps']!=meta['target_steps'] or meta['stage']!='single':
        raise ValueError('Completed phase-actor single PPO parent required')
    cfg=PPOConfig(kinematic_features=True,accelerated_features=True,phase_actor=True,
        startup_width=256,startup_relu=True,entropy=.005,gae_lambda=1.,
        actor_lr=3e-5,initial_actor_lr=3e-5)
    model=MAPPOCTDE(cfg)
    model.actor.flow.load_state_dict(source.actor.flow.state_dict())
    dataset=Path(dataset)
    data=torch.load(dataset,map_location='cpu',weights_only=True)
    if len(data['obs'])!=8192 or not (data['obs'][:,5]==0).all():
        raise ValueError('Use the entire fixed startup training table')
    if loss_kind not in ('expected','q_mse'):
        raise ValueError('Unknown supervised initialization loss')
    plan=dict(algorithm='planner_bc_initializer_20261101',
        initialization_method='expected expert regret plus .025s soft-target CE, ReLU256 startup only' if loss_kind=='expected' else 'masked fixed-expert regret-logit regression, ReLU256 startup only',
        loss_kind=loss_kind,regression_cost_scale_seconds=.5,
        ppo_config=asdict(cfg),seed=seed,epochs=400,initials=8192,
        parent_ppo=dict(path=str(parent),sha256=hashlib.sha256(Path(parent).read_bytes()).hexdigest()),
        dataset_source=dict(path=str(dataset),sha256=hashlib.sha256(dataset.read_bytes()).hexdigest()),
        fresh_expert_queries=0,flow_head='exact frozen parent PPO flow',
        startup_head='fresh random ReLU256',supervised_lr=3e-4,
        subsequent_ppo='new on-policy samples only; no expert table read by PPO')
    (out/'protocol.json').write_text(json.dumps(plan,indent=2)+'\n')
    (out/'dataset.pt').write_bytes(dataset.read_bytes())
    obs,mask,q,targets=[data[k] for k in ('obs','mask','costs','targets')]
    regret=(q-q.min(-1,keepdim=True).values).masked_fill(~mask,0.)
    optimizer=torch.optim.Adam(model.actor.initial.parameters(),lr=3e-4)
    for _ in range(400):
        for ids in torch.randperm(len(obs)).split(512):
            if loss_kind=='expected':
                dist=model.distribution(obs[ids],mask[ids])
                loss=((dist.probs*regret[ids]).sum(-1)-.025*(targets[ids]*dist.logits).sum(-1)).mean()
            else:
                raw=model.actor.initial(model.features(obs[ids]))
                target=-regret[ids]/.5
                loss=(raw-target).square()[mask[ids]].mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite expected-regret initializer loss')
            optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor.initial.parameters(),.5)
            optimizer.step()
    with torch.no_grad():
        prediction=model.distribution(obs,mask).logits.argmax(-1)
        greedy=regret.gather(1,prediction[:,None]).squeeze(-1)
    checkpoint=dict(**plan,model=dict(actor=model.actor.state_dict()),dataset_sha256=plan['dataset_source']['sha256'])
    torch.save(checkpoint,out/'initializer.pt')
    report=dict(training_initials=len(obs),new_expert_queries=0,epochs=400,
        train_greedy_nominal_regret_s=float(greedy.mean()),train_regret_above_one_second=int((greedy>1.).sum()),
        initializer_sha256=hashlib.sha256((out/'initializer.pt').read_bytes()).hexdigest(),
        held_out_efficacy_claim=False)
    (out/'complete.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True)
    parser.add_argument('--parent',required=True);parser.add_argument('--dataset',required=True)
    parser.add_argument('--seed',type=int,default=4507)
    parser.add_argument('--loss',choices=['expected','q_mse'],default='expected')
    args=parser.parse_args();initialize(args.out,args.parent,args.dataset,args.seed,args.loss)
