"""Collect training/validation/calibration labels only; no test states accessed."""
import hashlib
import json
from pathlib import Path

import numpy as np

from coopgate.merge_v4 import MergeConfig, MergeEnv
from coopgate.planning_v5 import plan_episode


ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'artifacts/v6'


def main():
    protocol = json.loads((D/'protocol.json').read_text())
    out = D/'data'; out.mkdir(exist_ok=False)
    for name, digest in protocol['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    with MergeEnv(MergeConfig(**protocol['config']), 'constant') as env:
        for split in ('train', 'validation', 'calibration'):
            states, costs, seeds = [], [], []
            for i, seed in enumerate(protocol['data_seeds'][split]):
                env.reset(seed, 'nominal')
                plan = plan_episode(env.x, env.prev, env.c)
                states.append(env.x.copy()); costs.append(plan.costs); seeds.append(seed)
                if (i+1) % 64 == 0:
                    print(json.dumps({'split': split, 'cases': i+1}), flush=True)
            np.savez_compressed(out/f'{split}.npz', states=np.array(states), costs=np.array(costs), seeds=np.array(seeds))


if __name__ == '__main__':
    main()
