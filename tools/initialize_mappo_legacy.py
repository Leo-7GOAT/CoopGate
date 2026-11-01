"""Explicit transfer initializer. Old PI weights/results are NEVER called PPO.

Only subsequent newly collected on-policy clipped-PPO updates create the new
MAPPO result. The unchanged initializer must remain a separate comparator.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from coopgate.ego_v13 import ego_observations
from coopgate.mappo_ctde import MAPPOCTDE,PPOConfig
from coopgate.marl_v12 import LEVELS,action_mask
from coopgate.policy_iteration_v14 import load_policy
from develop_v12 import initial
from prerequisite_v15 import require_single

ROOT=Path(__file__).resolve().parents[1]


def initialize(out,parent):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(4601)
    prior=require_single(ROOT)
    old,meta=load_policy(parent)
    cfg=PPOConfig(width=meta['width'],kinematic_features=True,actor_relu=True,role_actors=True,
                  actor_logit_scale=5.,actor_lr=1e-5,entropy=.001,gae_lambda=1.)
    model=MAPPOCTDE(cfg)
    state=old.state_dict()
    mapping={'0.weight':'encoder.0.weight','0.bias':'encoder.0.bias',
             '2.weight':'encoder.2.weight','2.bias':'encoder.2.bias',
             '4.weight':'advantage.weight','4.bias':'advantage.bias'}
    model.actor.roles[1].load_state_dict({new:state[origin] for new,origin in mapping.items()})
    with torch.no_grad():
        for i in (0,2):
            model.actor.roles[i][-1].weight.zero_()
            # Sampling logits have slope .4 after the positive factor five;
            # main-road exploration is not initialized at old saturated 3*LEVELS.
            model.actor.roles[i][-1].bias.copy_(torch.tensor(.08*LEVELS,dtype=torch.float32))
    x=np.array([initial(95400001+i,'nominal' if i%2==0 else 'dense') for i in range(1024)])
    prev=np.zeros((len(x),3));obs=ego_observations(x,prev,0);mask=action_mask(prev)
    np.testing.assert_array_equal(old.proposal(obs[:,1],mask[:,1]),model.proposal(obs[:,1:2],mask[:,1:2])[:,0])
    # Neighbor-role logits must reproduce the declared highest-admissible rule.
    proposed=model.proposal(obs,mask)
    highest=np.where(mask,np.arange(len(LEVELS)),-1).max(-1)
    np.testing.assert_array_equal(proposed[:,(0,2)],highest[:,(0,2)])
    plan=dict(algorithm='legacy_pi_actor_initializer_20261101',
        purpose='Transfer initialization and separate comparator ONLY; NOT MAPPO training/result',
        parent=dict(path=str(parent),sha256=hashlib.sha256(Path(parent).read_bytes()).hexdigest(),
                    algorithm=meta['algorithm'],checkpoint_seed=meta['seed']),
        original_single_prerequisite=prior,ppo_config=asdict(cfg),
        conversion='Exact ReLU weights and local feature functions; no online policy iteration',
        sampling_logit_scale=5.,greedy_action_conversion_cases=1024,conversion_all_equal=True,
        main_road_initialization='Highest-admissible greedy; non-saturated stochastic .4*LEVELS preference',
        certificate_transfer=False,original_training_cost='Inherited PI training budget; not new PPO sample efficiency')
    torch.save(dict(**plan,model=dict(actor=model.actor.state_dict())),out/'initializer.pt')
    (out/'protocol.json').write_text(json.dumps(plan,indent=2)+'\n')
    report=dict(conversion_all_equal=True,checked_initials=1024,
                original_single_prerequisite_verified=True,old_results_relabelled=False,
                initializer_sha256=hashlib.sha256((out/'initializer.pt').read_bytes()).hexdigest())
    (out/'complete.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);parser.add_argument('--parent',required=True)
    args=parser.parse_args();initialize(args.out,args.parent)
