"""Warm the compiled controller before connecting to SUMO; optional GUI demo."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from coopgate.experiment_v8 import prepare
from coopgate.learning_v6 import load_models
from coopgate.merge_v4 import MergeConfig, MergeEnv

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--gui', action='store_true')
    parser.add_argument('--method', choices=['fast_pair', 'fast_full', 'bb_natural', 'bb_learned'], default='bb_learned')
    parser.add_argument('--seed', type=int, default=12000001); parser.add_argument('--scenario', choices=['nominal', 'dense'], default='dense')
    parser.add_argument('--out', type=Path, default=Path('runs/v8_demo')); args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False); c = MergeConfig(); models = load_models(ROOT/'artifacts/v6', [201, 211, 223])
    start = time.perf_counter(); initial = np.array([[-80., 12.], [-85., 13.], [-120., 11.]])
    _, warm_gate, _ = prepare(args.method, models, initial, np.zeros(3), c)
    warm_gate.select(initial, np.zeros(3), np.full(3, c.accel))
    warmup_ms = (time.perf_counter()-start)*1000
    print(json.dumps({'ready': True, 'warmup_ms': warmup_ms, 'method': args.method}), flush=True)
    with MergeEnv(c, 'constant', gui=args.gui) as env:
        env.reset(args.seed, args.scenario); start = time.perf_counter()
        selected, env.gate, info = prepare(args.method, models, env.x, env.prev, c)
        selection_ms = (time.perf_counter()-start)*1000
        while True:
            _, _, done, truncated, metrics = env.step([4]*3)
            if done or truncated:
                break
        result = {**metrics, 'method': args.method, 'selected': selected, 'selection_ms': selection_ms, 'warmup_ms': warmup_ms, **info}
        (args.out/'result.json').write_text(json.dumps(result, indent=2)+'\n', newline='\n')
        (args.out/'trace.json').write_text(json.dumps(env.trace, indent=2)+'\n', newline='\n')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
