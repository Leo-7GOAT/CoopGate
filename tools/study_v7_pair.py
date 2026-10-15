"""Four methods on 80 additional independent SUMO scenes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
from coopgate.merge_v4 import MergeConfig, MergeEnv
from coopgate.learning_v6 import ProfileGate, decide, load_models
from coopgate.planning_v5 import plan_episode, PlannedGate
from coopgate.verify_v7 import verify_proposal

ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'artifacts/v7_pair'


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, default=Path('runs/v7_pair_reproduction'))
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((D/'protocol.json').read_text())
    for name, digest in protocol['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    models = load_models(ROOT/'artifacts/v6', protocol['model_seeds']); rows, labels, timings = [], [], []
    with MergeEnv(MergeConfig(**protocol['config']), 'constant') as env:
        for regime, seeds in protocol['test_seeds'].items():
            for index, seed in enumerate(seeds):
                env.reset(seed, regime); teacher = plan_episode(env.x, env.prev, env.c)
                labels.append({'seed': seed, 'scenario': regime, 'initial': env.x.tolist(), 'costs': teacher.costs.tolist()})
                for method in protocol['methods']:
                    env.reset(seed, regime); start = time.perf_counter()
                    info = {'proposal': 0, 'estimated_excess_s': None, 'verified_excess_s': None}
                    if method == 'teacher':
                        selected = teacher.selected; env.gate = PlannedGate(env.c, teacher)
                    elif method == 'verified_pair':
                        selected, pair, info = verify_proposal(models, env.x, env.prev, env.c)
                        assert abs(pair.costs[0]-teacher.costs[0]) < 1e-6
                        assert abs(pair.costs[1]-teacher.costs[info['proposal']]) < 1e-6
                        env.gate = PlannedGate(env.c, pair)
                    elif method == 'learned':
                        selected, raw = decide(models, env.x); info.update(proposal=selected, estimated_excess_s=raw['estimated_excess_s'])
                        env.gate = ProfileGate(env.c, selected)
                    else:
                        selected = 0; env.gate = ProfileGate(env.c, selected)
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
                                 'selection_ms': selection_ms, 'first_response_ms': selection_ms+cycle[0],
                                 'plan_fallback_count': getattr(env.gate, 'fallback_count', 0)})
                    timings.append({'seed': seed, 'scenario': regime, 'method': method, 'selection_ms': selection_ms, 'cycle_ms': cycle})
                    if index == 0:
                        (args.out/f'trace_{regime}_{method}.json').write_text(json.dumps(env.trace, indent=2), newline='\n')
                if (index+1) % 10 == 0:
                    print(json.dumps({'regime': regime, 'scenes': index+1, 'executions': len(rows)}), flush=True)
    with (args.out/'episodes.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator='\n'); writer.writeheader(); writer.writerows(rows)
    for name, value in [('teacher_labels.json', labels), ('timing.json', timings)]:
        (args.out/name).write_text(json.dumps(value, indent=2)+'\n', newline='\n')
    (args.out/'manifest.json').write_text(json.dumps({'episodes': len(rows), 'source_sha256': {
        'artifacts/v7_pair/protocol.json': hashlib.sha256((D/'protocol.json').read_bytes()).hexdigest()}}, indent=2)+'\n', newline='\n')


if __name__ == '__main__':
    main()
