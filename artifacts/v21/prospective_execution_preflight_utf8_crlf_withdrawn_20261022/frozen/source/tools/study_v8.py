"""New matched SUMO evaluation with randomized execution order and full costs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import numba
from coopgate.experiment_v8 import prepare
from coopgate.fast_v8 import plan_fast
from coopgate.learning_v6 import load_models
from coopgate.merge_v4 import MergeConfig, MergeEnv
from coopgate.planning_v5 import plan_episode

ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'artifacts/v8'


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, default=Path('runs/v8_reproduction'))
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((D/'protocol.json').read_text())
    for name, digest in protocol['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    models = load_models(ROOT/'artifacts/v6', protocol['model_seeds']); config = MergeConfig(**protocol['config'])
    development_state = np.array([[-80., 12.], [-85., 13.], [-120., 11.]])
    # All methods are warmed on the same declared non-test development state.
    for method in protocol['methods']:
        _, gate, _ = prepare(method, models, development_state, np.zeros(3), config)
        gate.select(development_state, np.zeros(3), np.full(3, config.accel))
    rows, labels, timings = [], [], []
    with MergeEnv(config, 'constant') as env:
        for regime, seeds in protocol['test_seeds'].items():
            for index, seed in enumerate(seeds):
                env.reset(seed, regime); reference = plan_episode(env.x, env.prev, env.c)
                # Full compiled cost vector independently audited even if a
                # pruned method never computes that candidate to completion.
                fast, _ = plan_fast(env.x, env.prev, env.c)
                np.testing.assert_array_equal(fast.costs, reference.costs)
                order = np.random.default_rng(seed+808).permutation(protocol['methods']).tolist()
                labels.append({'seed': seed, 'scenario': regime, 'initial': env.x.tolist(),
                               'costs': reference.costs.tolist(), 'execution_order': order})
                selected_by_method = {}
                for order_index, method in enumerate(order):
                    env.reset(seed, regime); start = time.perf_counter()
                    selected, env.gate, info = prepare(method, models, env.x, env.prev, env.c)
                    selection_ms = (time.perf_counter()-start)*1000; cycle = []
                    selected_by_method[method] = selected
                    while True:
                        start = time.perf_counter(); _, _, done, truncated, metrics = env.step([4]*3)
                        cycle.append((time.perf_counter()-start)*1000)
                        if done or truncated:
                            break
                    assert abs(metrics['mean_capped_finish_s']-reference.costs[selected]) < 1e-6
                    if method in ('fast_full', 'bb_natural', 'bb_learned', 'reference_full'):
                        assert selected == reference.selected
                    rows.append({**metrics, 'method': method, 'selected': selected, **info, 'execution_order': order_index,
                                 'baseline_s': float(reference.costs[0]), 'teacher_s': float(reference.costs.min()),
                                 'selection_ms': selection_ms, 'first_response_ms': selection_ms+cycle[0],
                                 'plan_fallback_count': getattr(env.gate, 'fallback_count', 0)})
                    timings.append({'seed': seed, 'scenario': regime, 'method': method,
                                    'selection_ms': selection_ms, 'cycle_ms': cycle})
                    if index == 0:
                        (args.out/f'trace_{regime}_{method}.json').write_text(json.dumps(env.trace, indent=2), newline='\n')
                assert selected_by_method['fast_pair'] == selected_by_method['reference_pair']
                if (index+1) % 10 == 0:
                    print(json.dumps({'regime': regime, 'scenes': index+1, 'executions': len(rows)}), flush=True)
    with (args.out/'episodes.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator='\n'); writer.writeheader(); writer.writerows(rows)
    for name, value in [('teacher_labels.json', labels), ('timing.json', timings)]:
        (args.out/name).write_text(json.dumps(value, indent=2)+'\n', newline='\n')
    manifest = {'episodes': len(rows), 'python': platform.python_version(), 'platform': platform.platform(),
                'numpy': np.__version__, 'numba': numba.__version__, 'single_worker': True,
                'source_sha256': {'artifacts/v8/protocol.json': hashlib.sha256((D/'protocol.json').read_bytes()).hexdigest()}}
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', newline='\n')


if __name__ == '__main__':
    main()
