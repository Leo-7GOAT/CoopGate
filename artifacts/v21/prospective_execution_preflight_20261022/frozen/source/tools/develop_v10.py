"""V10 development: published V9 scenes only, never fresh test evidence."""
import argparse
import json
from pathlib import Path
import numpy as np
from coopgate.disturbance_v9 import DisturbedMergeEnv
from coopgate.merge_v4 import MergeConfig
from coopgate.robust_v9 import RobustGate
from coopgate.rollout_v10 import RolloutGate
from coopgate.learning_v6 import load_models, predict

ROOT = Path(__file__).resolve().parents[1]


def run(count, out):
    models = load_models(ROOT/'artifacts/v6', [201, 211, 223])
    old = json.loads((ROOT/'artifacts/v9/protocol.json').read_text())
    rows = []
    methods = ('v9_pair', 'v9_full', 'rollout_constant', 'rollout_pair_binary', 'rollout_pair', 'rollout_pair_twice', 'rollout_fixed')
    with DisturbedMergeEnv(MergeConfig(), 'constant') as env:
        for scenario, seeds in old['test_seeds'].items():
            for seed in seeds[:count]:
                for mode in old['disturbances']:
                    for method in methods:
                        env.reset_disturbed(seed, scenario, mode, .5)
                        p = int(predict(models, env.x[None])[0][0])
                        if method == 'rollout_fixed': p = 1
                        cls = RolloutGate if method.startswith('rollout') else RobustGate
                        strategy = 'full' if method == 'v9_full' else 'constant' if method == 'rollout_constant' else 'pair'
                        options = {'levels': 2 if method.endswith('binary') else 5} if cls == RolloutGate else {}
                        if method == 'rollout_pair_twice': options['passes'] = 2
                        env.gate = cls(env.x, env.prev, env.c, .5, strategy, p, **options)
                        while True:
                            _, _, done, truncated, metrics = env.step([4]*3)
                            if done or truncated:
                                break
                        rows.append({**metrics, 'method': method,
                                     'cycle_p95_ms': float(np.percentile([r['gate_ms'] for r in env.trace],95)),
                                     'updates': sum(r.get('accepted_updates',0) for r in env.trace)})
                print(json.dumps({'scenario':scenario,'seed':seed,'rows':len(rows)}),flush=True)
                out.write_text(json.dumps(rows,indent=2))
    for mode in old['disturbances']:
        for method in methods:
            group = [r for r in rows if r['disturbance']==mode and r['method']==method]
            print(mode,method,'cost',round(np.mean([r['mean_capped_finish_s'] for r in group]),3),
                  'complete',sum(r['completed'] for r in group),'abort',sum(r['aborted'] for r in group),
                  'p95',round(np.percentile([r['cycle_p95_ms'] for r in group],95),2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--count',type=int,default=12)
    parser.add_argument('--out',type=Path,default=Path('runs/v10_development.json'))
    args=parser.parse_args(); run(args.count,args.out)
