"""Check every integration SUMO episode against exact model and substeps."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from coopgate.disturbance_v9 import disturbance_tape
from coopgate.ego_v13 import EgoRolloutGate
from coopgate.joint_planner_v15 import JointRolloutGate
from coopgate.mappo_ctde import load_mappo
from coopgate.merge_v4 import MergeConfig, transition_safe
from evaluate_mappo_ctde import exact_episode, RuleGate, PPOGate


def verify(directory):
    torch.set_num_threads(1)
    directory=Path(directory)
    protocol=json.loads((directory/'protocol.json').read_text())
    if protocol['backend'] != 'sumo':
        raise ValueError('Actual SUMO results required')
    factories=dict(rule=RuleGate, ego_planner=EgoRolloutGate, team_planner=JointRolloutGate)
    for name, item in protocol['checkpoints'].items():
        model, meta=load_mappo(item['path'])
        factories[name]=lambda model=model, stage=meta['stage']:PPOGate(model, stage)
    rows=[json.loads(line) for line in (directory/'episodes.jsonl').read_text().splitlines()]
    seen=set(); substeps=commands=0
    for row in rows:
        key=(row['seed'], row['scenario'], row['mode'], row['method'])
        if key in seen:
            raise ValueError('Duplicated SUMO execution')
        seen.add(key)
        independent=exact_episode(row['seed'], row['scenario'], row['mode'], factories[row['method']])
        np.testing.assert_allclose(row['initial'], independent['initial'], rtol=0, atol=1e-8)
        assert row['completed'] == independent['completed']
        np.testing.assert_allclose(row['per_vehicle_cost'], independent['per_vehicle_cost'], rtol=0, atol=1e-8)
        assert len(row['trace']) == len(independent['trace'])
        state=np.array(row['initial']); previous=np.zeros(3); variation=np.zeros(3)
        tape=disturbance_tape(row['seed'], row['mode'], .5, 250)
        for k, (actual, reference) in enumerate(zip(row['trace'], independent['trace'])):
            command=np.array(actual['command'])
            np.testing.assert_allclose(command, reference['command'], rtol=0, atol=1e-9)
            variation += (state[:, 0]<60.)*np.abs(command-previous)
            assert np.max(np.abs(command-previous)) <= 2.5+1e-9
            for innovation in tape[k*5:k*5+5]:
                before=state[:, 0].copy()
                state[:, 1]=np.clip(state[:, 1]+.1*(command+innovation), 0., 20.)
                state[:, 0] += .1*state[:, 1]
                assert transition_safe(before[None], state[None, :, 0], MergeConfig())[0]
                substeps += 1
            np.testing.assert_allclose(state, actual['x'], rtol=0, atol=1e-8)
            previous=command; commands += 1
        np.testing.assert_allclose(variation, row['per_vehicle_variation'], rtol=0, atol=1e-8)
    assert len(rows) == protocol['expected_executions']
    report=dict(actual_sumo_episodes=len(rows), commands_matched=commands,
                independent_physical_substeps=substeps, all_match=True,
                additional_independent_initials=0, purpose='Integration audit, not learner superiority')
    (directory/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('directory')
    verify(parser.parse_args().directory)
