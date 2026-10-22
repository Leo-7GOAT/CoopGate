"""New held-out SUMO study. Teacher labels never enter learned decisions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from coopgate.merge_v4 import MergeConfig, MergeEnv
from coopgate.planning_v5 import plan_episode, PlannedGate
from coopgate.learning_v6 import ProfileGate, decide, load_models


ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'artifacts/v6'


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, default=Path('runs/v6_reproduction'))
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((D/'protocol.json').read_text()); cal = json.loads((D/'calibration.json').read_text())
    for source in (protocol, cal):
        for name, digest in source['source_sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    models = load_models(D, protocol['training']['seeds'])
    imitation = load_models(D, protocol['training']['seeds'], 'imitation.pt')
    fixed = int(np.argmin(np.load(D/'data/train.npz')['costs'].mean(axis=0)))
    rows, labels, timings = [], [], []
    with MergeEnv(MergeConfig(**protocol['config']), 'constant') as env:
        for split, seeds in protocol['test_seeds'].items():
            for index, seed in enumerate(seeds):
                env.reset(seed, split)
                teacher = plan_episode(env.x, env.prev, env.c)
                labels.append({'seed': seed, 'scenario': split, 'initial': env.x.tolist(), 'costs': teacher.costs.tolist()})
                for method in protocol['methods']:
                    env.reset(seed, split)
                    start = time.perf_counter()
                    info = {'proposal': 0, 'estimated_excess_s': None, 'upper_excess_s': None, 'accepted': False}
                    if method == 'teacher':
                        selected = teacher.selected; env.gate = PlannedGate(env.c, teacher)
                    elif method == 'constant':
                        selected = 0; env.gate = ProfileGate(env.c, selected)
                    elif method == 'fixed':
                        selected = fixed; env.gate = ProfileGate(env.c, selected)
                    else:
                        selected, info = decide(imitation if method == 'imitation' else models, env.x,
                                                cal['upper_residual_quantile_s'] if method == 'calibrated' else None)
                        env.gate = ProfileGate(env.c, selected)
                    selection_ms = (time.perf_counter()-start)*1000+(teacher.planning_ms if method == 'teacher' else 0)
                    cycle = []
                    while True:
                        start = time.perf_counter(); _, _, terminal, truncated, metrics = env.step([4]*3)
                        cycle.append((time.perf_counter()-start)*1000)
                        if terminal or truncated:
                            break
                    assert abs(metrics['mean_capped_finish_s']-teacher.costs[selected]) < 1e-6
                    row = {**metrics, 'method': method, 'selected': selected, **info,
                           'baseline_s': float(teacher.costs[0]), 'teacher_s': float(teacher.costs.min()),
                           'proposal_actual_excess_s': float(teacher.costs[info['proposal']]-teacher.costs[0]),
                           'selection_ms': selection_ms, 'first_response_ms': selection_ms+cycle[0]}
                    rows.append(row)
                    timings.append({'seed': seed, 'scenario': split, 'method': method,
                                    'selection_ms': selection_ms, 'cycle_ms': cycle})
                    if index == 0:
                        (args.out/f'trace_{split}_{method}.json').write_text(json.dumps(env.trace, indent=2), newline='\n')
                if (index+1) % 10 == 0:
                    print(json.dumps({'split': split, 'scenes': index+1, 'executions': len(rows)}), flush=True)
    with (args.out/'episodes.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator='\n'); writer.writeheader(); writer.writerows(rows)
    for name, data in [('teacher_labels.json', labels), ('timing.json', timings)]:
        (args.out/name).write_text(json.dumps(data, indent=2)+'\n', newline='\n')
    sources = [D/'protocol.json', D/'calibration.json', *[D/f'seed{s}'/name for s in protocol['training']['seeds'] for name in ('policy.pt', 'imitation.pt')]]
    manifest = {'episodes': len(rows), 'fixed_profile_from_training_only': fixed,
                'source_sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', newline='\n')


if __name__ == '__main__':
    main()
