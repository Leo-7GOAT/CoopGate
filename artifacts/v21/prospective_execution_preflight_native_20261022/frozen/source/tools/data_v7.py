"""Fresh per-regime calibration labels only, with frozen V6 models."""
import hashlib
import json
from pathlib import Path
import numpy as np
from coopgate.merge_v4 import MergeConfig, MergeEnv
from coopgate.planning_v5 import plan_episode, portfolio
from coopgate.learning_v6 import load_models, predict

ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'artifacts/v7'


def main():
    protocol = json.loads((D/'protocol.json').read_text())
    for name, digest in protocol['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    out = D/'data'; out.mkdir(exist_ok=False)
    models = load_models(ROOT/'artifacts/v6', protocol['model_seeds']); profiles = portfolio()
    with MergeEnv(MergeConfig(**protocol['config']), 'constant') as env:
        for regime, seeds in protocol['calibration_seeds'].items():
            states, proposals, predictions, actual, costs = [], [], [], [], []
            for i, seed in enumerate(seeds):
                env.reset(seed, regime)
                action, estimate = predict(models, env.x[None]); action = int(action[0])
                pair = plan_episode(env.x, env.prev, env.c, profiles=[profiles[0], profiles[action]])
                states.append(env.x.copy()); proposals.append(action); predictions.append(float(estimate[0]))
                actual.append(float(pair.costs[1]-pair.costs[0])); costs.append(pair.costs.tolist())
                if (i+1) % 64 == 0:
                    print(json.dumps({'regime': regime, 'calibration_cases': i+1}), flush=True)
            np.savez_compressed(out/f'{regime}.npz', seeds=seeds, states=states, proposal=proposals,
                                predicted=predictions, actual=actual, costs=costs)


if __name__ == '__main__':
    main()
