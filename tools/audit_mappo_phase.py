"""Verify full fixed-expert startup table and preserved parent flow weights."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from coopgate.ego_v13 import EgoRolloutGate,ego_observations
from coopgate.fast_v8 import parameters
from coopgate.mappo_ctde import load_mappo
from coopgate.marl_v12 import LEVELS,action_mask
from coopgate.merge_v4 import MergeConfig
from develop_v12 import initial
from initialize_mappo_phase import initial_action_costs


def audit(directory):
    directory=Path(directory);torch.set_num_threads(1)
    source=torch.load(directory/'initializer.pt',map_location='cpu',weights_only=True)
    if source['initialization_method']!='local-clock phase heads, startup cost-sensitive fixed expert':
        raise ValueError('Wrong initializer identity')
    data=torch.load(directory/'dataset.pt',map_location='cpu',weights_only=True)
    if hashlib.sha256((directory/'dataset.pt').read_bytes()).hexdigest()!=source['dataset_sha256']:
        raise ValueError('Startup table hash mismatch')
    parent=Path(source['parent_ppo']['path'])
    if hashlib.sha256(parent.read_bytes()).hexdigest()!=source['parent_ppo']['sha256']:
        raise ValueError('Parent PPO changed')
    model,_=load_mappo(parent)
    for key,value in model.actor.state_dict().items():
        torch.testing.assert_close(value,source['model']['actor']['flow.'+key],rtol=0,atol=0)
    work=0
    for i in range(source['initials']):
        state=initial(source['source_seed_base']+i,'nominal' if i%2==0 else 'dense')
        obs=ego_observations(state,np.zeros(3),0)[1]
        mask=action_mask(np.zeros(3))[1]
        np.testing.assert_array_equal(obs,data['obs'][i].numpy())
        np.testing.assert_array_equal(mask,data['mask'][i].numpy())
        cost,count=initial_action_costs(state,parameters(MergeConfig()));work+=count
        np.testing.assert_allclose(cost.astype(np.float32),data['costs'][i].numpy(),atol=0,rtol=0)
        # An independent original planner call checks that these costs agree
        # with the unchanged strong planner's first-action optimization.
        _,info=EgoRolloutGate().select(state,np.zeros(3))
        index=int(np.argmin(np.abs(LEVELS-info['ego_proposal'])))
        if cost[index]>cost[mask].min()+1e-8:
            raise ValueError('Cost table differs from original strong ego objective')
        if abs(cost[index]-info['rollout_value'])>1e-8:
            raise ValueError('Cost scale differs from original strong ego planner')
    target=torch.softmax((-(data['costs']-data['costs'].min(-1,keepdim=True).values)/.25).masked_fill(~data['mask'],-1e9),-1)
    torch.testing.assert_close(target,data['targets'],rtol=0,atol=0)
    report=dict(training_initials=source['initials'],all_cost_targets_regenerated=True,
        original_strong_planner_argmin_verified=True,parent_flow_exactly_preserved=True,
        expert_counterfactual_joint_decisions=work,
        meaning='Expert initialization integrity, not on-policy PPO or traffic superiority')
    (directory/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory')
    audit(parser.parse_args().directory)
