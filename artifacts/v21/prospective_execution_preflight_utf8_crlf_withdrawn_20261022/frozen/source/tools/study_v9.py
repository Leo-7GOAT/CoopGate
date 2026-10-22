"""Frozen paired perturbation study, with failures retained in the denominator."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import numba
from coopgate.disturbance_v9 import DisturbedMergeEnv
from coopgate.experiment_v9 import prepare
from coopgate.learning_v6 import load_models
from coopgate.merge_v4 import MergeConfig

ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'artifacts/v9'


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((D/'protocol.json').read_text())
    for name, digest in protocol['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    models = load_models(ROOT/'artifacts/v6', protocol['model_seeds']); config = MergeConfig(**protocol['config'])
    radius = protocol['radius']; methods = protocol['methods']
    warm = np.array([[-80., 12.], [-85., 13.], [-120., 11.]])
    for method in methods:
        gate, _ = prepare(method, models, warm, np.zeros(3), config, radius, 0)
        gate.select(warm, np.zeros(3), np.full(3, config.accel))
    rows, timings, initials = [], [], []
    with DisturbedMergeEnv(config, 'constant') as env:
        for regime, seeds in protocol['test_seeds'].items():
            for index, seed in enumerate(seeds):
                for mode_index, mode in enumerate(protocol['disturbances']):
                    order = np.random.default_rng(seed+903+mode_index).permutation(methods).tolist()
                    for order_index, method in enumerate(order):
                        env.reset_disturbed(seed, regime, mode, radius)
                        if mode_index == 0 and order_index == 0:
                            initials.append({'seed': seed, 'scenario': regime, 'state': env.x.tolist()})
                        start = time.perf_counter(); info = {'selected': -1, 'proposal': -1, 'nominal_predicted_cost_s': config.horizon}
                        cycle = []
                        try:
                            env.gate, info = prepare(method, models, env.x, env.prev, config, radius, seed)
                        except (ValueError, RuntimeError) as exc:
                            env.abort_reason = 'initialization: '+str(exc)
                        selection_ms = (time.perf_counter()-start)*1000
                        if not env.abort_reason:
                            while True:
                                start = time.perf_counter(); _, _, done, truncated, _ = env.step([4]*3)
                                cycle.append((time.perf_counter()-start)*1000)
                                if done or truncated: break
                        metrics = env.metrics()
                        rows.append({**metrics, 'method': method, **info, 'execution_order': order_index,
                                     'selection_ms': selection_ms, 'first_response_ms': selection_ms+(cycle[0] if cycle else 0.),
                                     'fallback_steps': sum(bool(t['plan_fallback']) for t in env.trace)})
                        timings.append({'seed': seed, 'scenario': regime, 'disturbance': mode, 'method': method, 'cycle_ms': cycle})
                        if index == 0:
                            (out/f'trace_{regime}_{mode}_{method}.json').write_text(json.dumps(env.trace, indent=2)+'\n', newline='\n')
                if (index+1) % 5 == 0:
                    print(json.dumps({'regime': regime, 'scenes': index+1, 'executions': len(rows)}), flush=True)
                    # Durable incremental evidence if interrupted; no silent restart.
                    with (out/'episodes.csv').open('w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator='\n'); writer.writeheader(); writer.writerows(rows)
    for name, value in [('initial_states.json', initials), ('timing.json', timings)]:
        (out/name).write_text(json.dumps(value, indent=2)+'\n', newline='\n')
    manifest = {'episodes': len(rows), 'python': platform.python_version(), 'platform': platform.platform(), 'numpy': np.__version__,
                'numba': numba.__version__, 'single_worker': True,
                'source_sha256': {'artifacts/v9/protocol.json': hashlib.sha256((D/'protocol.json').read_bytes()).hexdigest()}}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', newline='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, default=Path('runs/v9_reproduction'))
    run(parser.parse_args().out)
