"""Independent SUMO evaluation; the protocol and controller are frozen first."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from coopgate.merge_v4 import MergeConfig, MergeEnv
from coopgate.learning_v4 import load
from coopgate.planning_v5 import plan_episode, PlannedGate


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('runs/v5_reproduction'))
    args = parser.parse_args()
    protocol_path = ROOT/'artifacts/v5/protocol.json'
    protocol = json.loads(protocol_path.read_text())
    for name, expected in protocol['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
    args.out.mkdir(parents=True, exist_ok=False)
    model, _ = load(ROOT/'artifacts/v4/seed127/policy.pt')
    rows, candidate_costs, timing = [], [], []
    selection = {'short_portfolio': 'short', 'no_coast': 'no_coast', 'full_portfolio': 'full'}
    with MergeEnv(MergeConfig(**protocol['config']), method='constant') as env:
        for method in protocol['methods']:
            for split, scenes in protocol['splits'].items():
                for index, scene in enumerate(scenes):
                    obs = env.reset(**scene)
                    planning_ms, plan = 0., None
                    if method in selection:
                        plan = plan_episode(env.x, env.prev, env.c, selection=selection[method])
                        planning_ms = plan.planning_ms
                        env.gate = PlannedGate(env.c, plan)
                    elif method == 'frozen_rl127':
                        env.gate.method = 'joint_rl'
                    cycles = []
                    while True:
                        start = time.perf_counter()
                        actions = model.act(obs, True)[0] if method == 'frozen_rl127' else [4]*3
                        obs, _, terminal, truncated, metrics = env.step(actions)
                        cycles.append((time.perf_counter()-start)*1000)
                        if terminal or truncated:
                            break
                    row = {**metrics, 'method': method, 'planning_ms': planning_ms,
                           'first_response_ms': planning_ms+cycles[0],
                           'cycle_p95_ms': float(np.percentile(cycles, 95)),
                           'cycle_max_ms': max(cycles), 'plan_fallbacks': 0,
                           'selected': '', 'predicted_cost': '', 'predicted_baseline': ''}
                    if plan is not None:
                        row.update(plan_fallbacks=env.gate.fallback_count,
                                   selected=plan.profiles[plan.selected]['name'],
                                   predicted_cost=plan.predicted_cost, predicted_baseline=float(plan.costs[0]))
                        assert abs(row['mean_capped_finish_s']-plan.predicted_cost) < 1e-6
                        assert env.gate.fallback_count == 0
                        if method == 'full_portfolio':
                            candidate_costs.append({**scene, 'costs': plan.costs.tolist(),
                                                    'short_scores': plan.short_scores.tolist(), 'selected': plan.selected})
                    rows.append(row)
                    timing.append({**scene, 'method': method, 'planning_ms': planning_ms, 'cycle_ms': cycles,
                                   'gate_ms': [r['gate_ms'] for r in env.trace]})
                    if index == 0:
                        (args.out/f'trace_{split}_{method}.json').write_text(json.dumps(env.trace, indent=2), newline='\n')
                    if (index+1) % 10 == 0:
                        print(json.dumps({'method': method, 'split': split, 'completed_cases': index+1}), flush=True)
    with (args.out/'episodes.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    for name, content in [('candidate_costs.json', candidate_costs), ('timing.json', timing)]:
        (args.out/name).write_text(json.dumps(content, indent=2)+'\n', newline='\n')
    manifest = {'episodes': len(rows), 'protocol_sha256': hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                'source_sha256': protocol['source_sha256']}
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', newline='\n')
    print(json.dumps({'episodes': len(rows), 'out': str(args.out)}), flush=True)


if __name__ == '__main__':
    main()
