"""Regenerate every separately labeled planner-BC training example."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from coopgate.disturbance_v9 import disturbance_tape
from coopgate.ego_v13 import EgoRolloutGate, ego_observations
from coopgate.fast_v8 import parameters
from coopgate.marl_v12 import action_mask, LEVELS, batch_step
from coopgate.merge_v4 import MergeConfig
from develop_v12 import initial


def audit(directory):
    directory=Path(directory); torch.set_num_threads(1)
    source=torch.load(directory/'initializer.pt', map_location='cpu', weights_only=True)
    if source['algorithm'] != 'planner_bc_initializer_20261101':
        raise ValueError('Not an explicit supervised initializer')
    if source['dataset_sha256'] != hashlib.sha256((directory/'dataset.pt').read_bytes()).hexdigest():
        raise ValueError('BC dataset hash mismatch')
    data=torch.load(directory/'dataset.pt', map_location='cpu', weights_only=True)
    rows=len(data['labels']); reviewed=episode=0
    while reviewed<rows:
        seed=948000001+episode
        scenario='nominal' if episode%2==0 else 'dense'
        mode=('clean','uniform','bias','pulse')[(episode//2)%4]
        state=initial(seed, scenario); previous=np.zeros(3)
        tape=disturbance_tape(seed, mode, .5, 250)
        planner=EgoRolloutGate()
        for step in range(50):
            if state[1,0]>=60. or reviewed>=rows:
                break
            np.testing.assert_array_equal(ego_observations(state, previous, step)[1], data['obs'][reviewed].numpy())
            np.testing.assert_array_equal(action_mask(previous)[1], data['mask'][reviewed].numpy())
            command, info=planner.select(state, previous)
            if abs(LEVELS[int(data['labels'][reviewed])]-info['ego_proposal'])>1e-10:
                raise ValueError('Teacher action label mismatch')
            nxt, actual, ok=batch_step(state[None], previous[None], command[None],
                tape[None, step*5:step*5+5], parameters(MergeConfig()), .5)
            assert ok[0] and np.array_equal(actual[0], command)
            state=nxt[0]; previous=command; reviewed+=1
        episode+=1
    report=dict(labels_independently_regenerated=reviewed, source_episodes=episode,
        all_match=True, algorithm_identity='supervised initializer, not PPO',
        held_out_seed_overlap=False,
        dataset_sha256=source['dataset_sha256'],
        meaning='Provenance and label reproducibility; not expert generalization or PPO benefit')
    (directory/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('directory')
    audit(parser.parse_args().directory)
