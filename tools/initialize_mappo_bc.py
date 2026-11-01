"""Separately labeled expert initializer; it is NOT PPO training.

Fresh planner trajectories provide proposals, not value targets. PPO later
uses its own new on-policy data and never reads these labels.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from coopgate.disturbance_v9 import disturbance_tape
from coopgate.ego_v13 import EgoRolloutGate, ego_observations
from coopgate.fast_v8 import parameters
from coopgate.mappo_ctde import MAPPOCTDE, PPOConfig
from coopgate.marl_v12 import action_mask, LEVELS, batch_step
from coopgate.merge_v4 import MergeConfig
from develop_v12 import initial


def initialize(out, n=8192, seed=4503, dataset=None, epochs=80, accelerated=False, actor_lr=3e-4, entropy=.02):
    out=Path(out); out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    cfg=PPOConfig(kinematic_features=True, entropy=entropy, gae_lambda=1.,
                  accelerated_features=accelerated,actor_lr=actor_lr)
    model=MAPPOCTDE(cfg)
    plan=dict(algorithm='planner_bc_initializer_20261101', purpose='expert initialization, NOT MAPPO',
        source='fresh strong ego planner proposals; no old policy-iteration weights',
        state_target_count=n, seed=seed, source_seed_base=948000001, epochs=epochs,
        reused_dataset=None if dataset is None else dict(path=str(dataset),sha256=hashlib.sha256(Path(dataset).read_bytes()).hexdigest()),
        learning_rate=3e-4, label_smoothing=.05, ppo_config=asdict(cfg),
        primary_checkpoint='fixed final epoch; no test-based checkpoint selection',
        information='same 18 local features and 8 deterministic kinematic functions')
    (out/'protocol.json').write_text(json.dumps(plan, indent=2)+'\n')
    obs=[]; masks=[]; labels=[]; audit=[]; episode=0; started=time.perf_counter()
    if dataset:
        existing=torch.load(dataset,map_location='cpu',weights_only=True)
        if len(existing['labels'])!=n:
            raise ValueError('Reuse the entire fixed expert dataset')
        obs=list(existing['obs'].numpy());masks=list(existing['mask'].numpy())
        labels=existing['labels'].tolist();audit=existing['audit_states']
    while len(obs)<n:
        source_seed=948000001+episode
        scenario='nominal' if episode%2==0 else 'dense'
        mode=('clean','uniform','bias','pulse')[(episode//2)%4]
        state=initial(source_seed, scenario); previous=np.zeros(3)
        tape=disturbance_tape(source_seed, mode, .5, 250)
        planner=EgoRolloutGate()
        for step in range(50):
            if state[1,0]>=60. or len(obs)>=n:
                break
            ob=ego_observations(state, previous, step)[1]
            mask=action_mask(previous)[1]
            command, info=planner.select(state, previous)
            label=int(np.argmin(np.abs(LEVELS-info['ego_proposal'])))
            if abs(LEVELS[label]-info['ego_proposal'])>1e-10 or not mask[label]:
                raise ValueError('Teacher proposal not a valid categorical action')
            obs.append(ob); masks.append(mask); labels.append(label)
            if len(audit)<32:
                audit.append(dict(source_seed=source_seed, scenario=scenario, mode=mode,
                    step=step, state=state.tolist(), previous=previous.tolist(), label=label))
            next_state, executed, valid=batch_step(state[None], previous[None], command[None],
                tape[None, step*5:step*5+5], parameters(MergeConfig()), .5)
            if not valid[0] or not np.array_equal(executed[0], command):
                raise RuntimeError('Teacher physical trajectory failed')
            state=next_state[0]; previous=command
        episode+=1
    obs=torch.as_tensor(np.asarray(obs)); mask=torch.as_tensor(np.asarray(masks)); labels=torch.tensor(labels)
    torch.save(dict(obs=obs, mask=mask, labels=labels, audit_states=audit, protocol=plan), out/'dataset.pt')
    optimizer=torch.optim.Adam(model.actor.parameters(), lr=3e-4)
    for epoch in range(epochs):
        for ids in torch.randperm(n).split(512):
            dist=model.distribution(obs[ids], mask[ids])
            chosen=dist.log_prob(labels[ids])
            uniform=(dist.logits*mask[ids]).sum(-1)/mask[ids].sum(-1)
            loss=-(.95*chosen+.05*uniform).mean()
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor.parameters(), .5)
            optimizer.step()
    with torch.no_grad():
        accuracy=float((model.distribution(obs, mask).logits.argmax(-1)==labels).float().mean())
    checkpoint=dict(**plan, model=dict(actor=model.actor.state_dict()),
                    dataset_sha256=hashlib.sha256((out/'dataset.pt').read_bytes()).hexdigest())
    torch.save(checkpoint, out/'initializer.pt')
    report=dict(expert_samples=n, new_expert_episodes=episode, epochs=epochs,
        train_action_accuracy=accuracy, elapsed_s=time.perf_counter()-started,
        initializer_sha256=hashlib.sha256((out/'initializer.pt').read_bytes()).hexdigest(),
        meaning='Supervised initialization only; no held-out or PPO benefit established')
    (out/'complete.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--out', required=True)
    parser.add_argument('--samples', type=int, default=8192); parser.add_argument('--seed', type=int, default=4503)
    parser.add_argument('--dataset');parser.add_argument('--epochs',type=int,default=80)
    parser.add_argument('--accelerated',action='store_true');parser.add_argument('--actor-lr',type=float,default=3e-4)
    parser.add_argument('--entropy',type=float,default=.02)
    args=parser.parse_args(); initialize(args.out,args.samples,args.seed,args.dataset,args.epochs,args.accelerated,args.actor_lr,args.entropy)
