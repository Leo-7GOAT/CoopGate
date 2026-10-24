"""Check the pinned upstream planner's reset contract without training or SUMO.

The upstream file is supplied locally, not vendored or modified. These synthetic
checks are prerequisite engineering evidence, not a DMPS performance comparison.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


COMMIT = 'f37df2d3900cfdc330ebb39e9b89d05dd9b85a7e'
BLOB = '68a83a6eaf518cc4ed7d2f07ccf7cd7f7ccc686c'


class Space:
    shape = (1,)


class ResetContractEnv:
    """Deterministic scalar fixture with complete state save/restore."""
    observation_space = action_space = Space()

    def __init__(self):
        self.reset()

    def reset(self):
        self.x, self.steps = 0., 0
        return np.array([self.x])

    def step(self, action):
        self.x += float(np.asarray(action)[0])
        self.steps += 1
        return np.array([self.x]), -abs(self.x), False, {}

    def get_state(self):
        return self.x, self.steps

    def set_state(self, state):
        self.x, self.steps = state

    def safe(self):
        return self.x <= .2

    def stable(self):
        return self.x <= 0.

    def backup(self):
        return np.array([-.5])


def probe(source):
    raw = source.read_bytes()
    blob = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    if blob != BLOB:
        raise ValueError('Upstream source is not the pinned exact Git blob')
    spec = importlib.util.spec_from_file_location('pinned_dmps_mcts', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(1)
    np.random.seed(460046)

    def make():
        env = ResetContractEnv()
        def critic(obs, acts):
            return torch.zeros((len(obs), 1), device=obs.device)
        planner = module.MCTSPlanner(env, .99, 2, 4, 1,
                                    torch.device('cpu'), critic, single_action=True)
        return env, planner

    # Mirrors the first reset in upstream MCTSTrainer.evaluate: only env.reset.
    env, planner = make()
    env.reset()
    assert planner.obs is None
    try:
        planner.take_step(np.array([1.]))
    except TypeError as exc:
        first_reset_error = dict(type=type(exc).__name__, message=str(exc))
    else:
        raise AssertionError('Expected unsynchronized first-use reset to fail')

    # Positive control uses the planner's own reset API, no upstream edit.
    env, planner = make()
    planner.reset()
    data, (actions, intervention) = planner.take_step(np.array([1.]))
    assert intervention == 1 and len(data) == len(actions) == 1
    assert env.safe()
    np.testing.assert_array_equal(planner.obs, np.array([env.x]))

    # After a previous episode, an env-only reset can silently retain stale obs.
    env, planner = make()
    planner.reset()
    planner.take_step(np.array([-.5]))
    stale = planner.obs.copy()
    current = env.reset()
    assert not np.array_equal(stale, current)
    planner.reset()
    np.testing.assert_array_equal(planner.obs, current)

    # Recoverability forecasts must not change the real executed state/history.
    original = env.get_state()
    assert not planner.check_action(np.array([1.]))
    assert env.get_state() == original
    assert planner.check_action(np.array([-.5]))
    assert env.get_state() == original
    return dict(upstream_repository='https://github.com/arkobnrjee/dmps',
                upstream_commit=COMMIT, upstream_file='mcts.py', upstream_git_blob=blob,
                upstream_sha256=hashlib.sha256(raw).hexdigest(),
                checks_passed=4, source_modified=False, synthetic_fixture=True,
                sumo_runs=0, training_steps=0, efficacy_claimed=False,
                first_env_only_reset_error=first_reset_error,
                planner_reset_positive_control_passed=True,
                stale_observation_after_env_only_reset=stale.tolist(),
                current_observation=current.tolist(),
                complete_state_restoration_verified=True,
                interpretation='reset synchronization prerequisite only; no full trainer or paper reproduction')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
