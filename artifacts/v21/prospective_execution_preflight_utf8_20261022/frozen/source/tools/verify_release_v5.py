"""Check frozen evidence and replay preselected SUMO traces independently."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from coopgate.merge_v4 import MergeConfig, MergeEnv
from coopgate.learning_v4 import load
from coopgate.planning_v5 import plan_episode, PlannedGate


ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'artifacts/v5'


def main(replay=False):
    hashes = json.loads((D/'SHA256SUMS.json').read_text())
    for name, expected in hashes.items():
        assert hashlib.sha256((D/name).read_bytes()).hexdigest() == expected, name
    protocol = json.loads((D/'protocol.json').read_text())
    for name, expected in protocol['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
    manifest = json.loads((D/'study/manifest.json').read_text())
    assert manifest['protocol_sha256'] == hashes['protocol.json']
    rows = list(csv.DictReader((D/'study/episodes.csv').open()))
    assert len(rows) == manifest['episodes'] == protocol['executions'] == 400
    lookup = {(r['method'], r['scenario'], int(r['seed'])): r for r in rows}
    expected = {(m, s['scenario'], s['seed']) for m in protocol['methods'] for scenes in protocol['splits'].values() for s in scenes}
    assert set(lookup) == expected
    costs = json.loads((D/'study/candidate_costs.json').read_text())
    assert len(costs) == 80
    for case in costs:
        split, seed = case['scenario'], case['seed']
        best = int(np.argmin(case['costs']))
        assert best == case['selected']
        assert abs(min(case['costs'])-float(lookup['full_portfolio', split, seed]['mean_capped_finish_s'])) < 1e-6
        assert abs(case['costs'][0]-float(lookup['constant', split, seed]['mean_capped_finish_s'])) < 1e-6
    if replay:
        model, _ = load(ROOT/'artifacts/v4/seed127/policy.pt')
        modes = {'short_portfolio': 'short', 'no_coast': 'no_coast', 'full_portfolio': 'full'}
        with MergeEnv(MergeConfig(**protocol['config']), method='constant') as env:
            for method in protocol['methods']:
                for split, scenes in protocol['splits'].items():
                    obs = env.reset(**scenes[0])
                    if method in modes:
                        plan = plan_episode(env.x, env.prev, env.c, selection=modes[method])
                        env.gate = PlannedGate(env.c, plan)
                        assert plan.predicted_cost == float(lookup[method, split, scenes[0]['seed']]['predicted_cost'])
                    elif method == 'frozen_rl127':
                        env.gate.method = 'joint_rl'
                    trace = json.loads((D/f'study/trace_{split}_{method}.json').read_text())
                    for row in trace:
                        obs, _, t, u, m = env.step(model.act(obs, True)[0] if method == 'frozen_rl127' else [4]*3)
                        assert np.allclose(env.x, row['x'], atol=1e-6, rtol=0)
                        assert np.allclose(env.prev, row['command'], atol=1e-7, rtol=0)
                        assert env.trace[-1]['certified'] == row['certified']
                    assert t or u
                    assert m['mean_capped_finish_s'] == float(lookup[method, split, scenes[0]['seed']]['mean_capped_finish_s'])
                    print(json.dumps({'method': method, 'split': split, 'matched_steps': len(trace)}), flush=True)
    print(f'Verified {len(hashes)} V5 artifacts, 400 records, 80 candidate-cost tables and frozen sources.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--replay', action='store_true')
    main(parser.parse_args().replay)
