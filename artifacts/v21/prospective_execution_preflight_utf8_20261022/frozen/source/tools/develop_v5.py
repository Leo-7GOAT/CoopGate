"""Development-only headroom experiment; never reads V4 held-out scenes."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from coopgate.merge_v4 import MergeEnv
from coopgate.planning_v5 import plan_episode, PlannedGate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('runs/v5_development'))
    parser.add_argument('--count', type=int, default=3)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    with MergeEnv(method='constant') as env:
        for scenario in ('nominal', 'dense'):
            for seed in range(500, 500+args.count):
                env.reset(seed, scenario)
                plan = plan_episode(env.x, env.prev, env.c)
                env.gate = PlannedGate(env.c, plan)
                while True:
                    _, _, terminal, truncated, m = env.step([4]*3)
                    if terminal or truncated:
                        break
                assert abs(m['mean_capped_finish_s']-plan.predicted_cost) < 1e-6
                assert not m['collision'] and m['contract_violations'] == m['uncertified_steps'] == m['outside_steps'] == 0
                assert env.gate.fallback_count == 0
                row = {**m, 'baseline_predicted_s': float(plan.costs[0]),
                       'seconds_saved': float(plan.costs[0]-plan.predicted_cost),
                       'selected': plan.profiles[plan.selected]['name'], 'planning_ms': plan.planning_ms}
                rows.append(row)
                print(json.dumps(row), flush=True)
    with (args.out/'episodes.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({'development_cases': len(rows), 'mean_saved_s': np.mean([r['seconds_saved'] for r in rows])}))


if __name__ == '__main__':
    main()
