"""Demonstrate calibrated learned selection in actual SUMO, optionally its GUI."""
import argparse
import json
from pathlib import Path

from coopgate.learning_v6 import ProfileGate, decide, load_models
from coopgate.merge_v4 import MergeEnv
from coopgate.planning_v5 import portfolio


ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'artifacts/v6'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=7000001)
    parser.add_argument('--scenario', choices=['nominal', 'dense'], default='nominal')
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--out', type=Path, default=Path('runs/v6_demo'))
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((D/'protocol.json').read_text()); cal = json.loads((D/'calibration.json').read_text())
    models = load_models(D, protocol['training']['seeds'])
    with MergeEnv(method='constant', gui=args.gui) as env:
        env.reset(args.seed, args.scenario)
        index, decision = decide(models, env.x, cal['upper_residual_quantile_s'])
        env.gate = ProfileGate(env.c, index)
        while True:
            _, _, terminal, truncated, metrics = env.step([4]*3)
            if terminal or truncated:
                break
        result = {**metrics, 'method': 'calibrated', 'decision': decision, 'profile': portfolio()[index]}
        (args.out/'results.json').write_text(json.dumps(result, indent=2), newline='\n')
        (args.out/'trace.json').write_text(json.dumps(env.trace, indent=2), newline='\n')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
