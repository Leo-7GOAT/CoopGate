"""Freeze the independent planning-headroom protocol before evaluation."""
from pathlib import Path
import hashlib
import json

import numpy as np

from coopgate.merge_v4 import MergeConfig
from coopgate.planning_v5 import portfolio


ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'artifacts/v5'


def main():
    D.mkdir(parents=True, exist_ok=True)
    target = D/'protocol.json'
    if target.exists():
        raise RuntimeError('Refusing to overwrite a frozen protocol')
    v4 = json.loads((ROOT/'artifacts/v4/protocol.json').read_text())
    old = {s['seed'] for scenes in v4['splits'].values() for s in scenes}
    rng = np.random.default_rng(2026101505)
    seeds = rng.choice(np.arange(1000000, 2000000), size=80, replace=False).tolist()
    assert not old.intersection(seeds)
    files = ['src/coopgate/merge_v4.py', 'src/coopgate/learning_v4.py',
             'src/coopgate/planning_v5.py', 'tools/study_v5.py',
             'artifacts/v4/seed127/policy.pt']
    protocol = {
        'version': 'v5', 'scope': 'Independent planning opportunity experiment; no new learning claim.',
        'config': MergeConfig().dict(), 'portfolio': portfolio(),
        'development_scenes': [{'seed': s, 'scenario': d} for d in ('nominal', 'dense') for s in range(500, 503)],
        'splits': {d: [{'seed': s, 'scenario': d} for s in seeds[k*40:(k+1)*40]] for k, d in enumerate(('nominal', 'dense'))},
        'methods': ['constant', 'frozen_rl127', 'short_portfolio', 'no_coast', 'full_portfolio'],
        'executions': 400,
        'primary': 'Matched mean capped per-car arrival time at s=60; deadline 25 seconds, collisions cost 25.',
        'ablations': {
            'short_portfolio': 'Select by 2-second clipped progress; execute the selected profile full-episode witness. Not receding-horizon MPC.',
            'no_coast': 'Full arrival cost, restricted to incumbent plus nine weighted profiles with no coasting prefix.',
            'full_portfolio': 'Full arrival cost across all 37 certified profiles, incumbent wins ties.',
            'frozen_rl127': 'Previously published final V4 checkpoint, selected before accessing these new test scenes.'},
        'timing': 'Single worker; report initial model planning and per-cycle execution separately. First response includes planning. No communication-delay metric.',
        'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files},
        'limits': ['Exact sampled dynamics, three fully controlled cars, fixed routes, no disturbance or background traffic.',
                   'Finite portfolio optimum is not a globally optimal solution.',
                   'Headroom is an opportunity for future learning, not evidence that MARL obtains the gain.']}
    target.write_text(json.dumps(protocol, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(target)


if __name__ == '__main__':
    main()
