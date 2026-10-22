"""Run the full-portfolio controller in SUMO or the real SUMO GUI."""
import argparse
import json
from pathlib import Path

from coopgate.merge_v4 import MergeEnv
from coopgate.planning_v5 import plan_episode, PlannedGate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--seed', type=int, default=500)
    parser.add_argument('--scenario', choices=['nominal', 'dense'], default='nominal')
    parser.add_argument('--out', type=Path, default=Path('runs/v5_demo'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    with MergeEnv(method='constant', gui=args.gui) as env:
        env.reset(args.seed, args.scenario)
        plan = plan_episode(env.x, env.prev, env.c)
        env.gate = PlannedGate(env.c, plan)
        while True:
            _, _, terminal, truncated, metrics = env.step([4]*3)
            if terminal or truncated:
                break
        (args.out/'trace.json').write_text(json.dumps(env.trace, indent=2), newline='\n')
        report = {**metrics, 'method': 'full_portfolio', 'predicted_baseline_s': float(plan.costs[0]),
                  'selected_profile': plan.profiles[plan.selected], 'planning_ms': plan.planning_ms,
                  'plan_fallbacks': env.gate.fallback_count}
        (args.out/'results.json').write_text(json.dumps(report, indent=2), newline='\n')
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
