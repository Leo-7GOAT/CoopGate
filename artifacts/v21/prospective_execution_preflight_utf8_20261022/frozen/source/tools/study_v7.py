"""Frozen V7 matched SUMO executions; evaluation labels never enter gates."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
from coopgate.learning_v6 import ProfileGate, load_models
from coopgate.merge_v4 import MergeConfig, MergeEnv
from coopgate.planning_v5 import plan_episode, PlannedGate
from coopgate.risk_v7 import decide

ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'artifacts/v7'


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, default=Path('runs/v7_reproduction'))
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((D/'protocol.json').read_text()); cal = json.loads((D/'calibration.json').read_text())
    old_q = json.loads((ROOT/'artifacts/v6/calibration.json').read_text())['upper_residual_quantile_s']
    for source in (protocol, cal):
        for name, digest in source['source_sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    models = load_models(ROOT/'artifacts/v6', protocol['model_seeds'])
    rows, labels, timings = [], [], []
    with MergeEnv(MergeConfig(**protocol['config']), 'constant') as env:
        for regime, seeds in protocol['test_seeds'].items():
            for index, seed in enumerate(seeds):
                env.reset(seed, regime); teacher = plan_episode(env.x, env.prev, env.c)
                labels.append({'seed': seed, 'scenario': regime, 'initial': env.x.tolist(), 'costs': teacher.costs.tolist()})
                for method in protocol['methods']:
                    env.reset(seed, regime); start = time.perf_counter()
                    if method == 'teacher':
                        selected = teacher.selected; env.gate = PlannedGate(env.c, teacher)
                        info = {'proposal': selected, 'estimated_excess_s': None, 'threshold_s': None}
                    else:
                        selected, info = decide(models, env.x, method, cal, old_q)
                        env.gate = ProfileGate(env.c, selected)
                    selection_ms = (time.perf_counter()-start)*1000+(teacher.planning_ms if method == 'teacher' else 0)
                    cycle = []
                    while True:
                        start = time.perf_counter(); _, _, done, truncated, metrics = env.step([4]*3)
                        cycle.append((time.perf_counter()-start)*1000)
                        if done or truncated:
                            break
                    assert abs(metrics['mean_capped_finish_s']-teacher.costs[selected]) < 1e-6
                    rows.append({**metrics, 'method': method, 'selected': selected, **info,
                                 'baseline_s': float(teacher.costs[0]), 'teacher_s': float(teacher.costs.min()),
                                 'proposal_actual_excess_s': float(teacher.costs[info['proposal']]-teacher.costs[0]),
                                 'selection_ms': selection_ms, 'first_response_ms': selection_ms+cycle[0]})
                    timings.append({'seed': seed, 'scenario': regime, 'method': method,
                                    'selection_ms': selection_ms, 'cycle_ms': cycle})
                    if index == 0:
                        (args.out/f'trace_{regime}_{method}.json').write_text(json.dumps(env.trace, indent=2), newline='\n')
                if (index+1) % 10 == 0:
                    print(json.dumps({'regime': regime, 'scenes': index+1, 'executions': len(rows)}), flush=True)
    with (args.out/'episodes.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator='\n'); writer.writeheader(); writer.writerows(rows)
    for name, value in [('teacher_labels.json', labels), ('timing.json', timings)]:
        (args.out/name).write_text(json.dumps(value, indent=2)+'\n', newline='\n')
    files = [D/'protocol.json', D/'calibration.json']
    (args.out/'manifest.json').write_text(json.dumps({'episodes': len(rows), 'source_sha256': {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}, indent=2)+'\n', newline='\n')


if __name__ == '__main__':
    main()
