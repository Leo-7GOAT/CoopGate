"""Verify frozen provenance, calibration, decisions and preselected SUMO replay."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from coopgate.learning_v6 import ProfileGate, decide, load_models, predict, upper_quantile
from coopgate.merge_v4 import MergeConfig, MergeEnv
from coopgate.planning_v5 import plan_episode, PlannedGate


ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'artifacts/v6'


def main(replay):
    hashes = json.loads((D/'SHA256SUMS.json').read_text())
    for name, digest in hashes.items():
        assert hashlib.sha256((D/name).read_bytes()).hexdigest() == digest, name
    protocol = json.loads((D/'protocol.json').read_text())
    cal = json.loads((D/'calibration.json').read_text())
    manifest = json.loads((D/'study/manifest.json').read_text())
    for source in (protocol, cal, manifest):
        for name, digest in source['source_sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    groups = [*protocol['data_seeds'].values(), *protocol['test_seeds'].values()]
    assert len({s for group in groups for s in group}) == sum(map(len, groups)) == 1272
    for split, seeds in protocol['data_seeds'].items():
        data = np.load(D/f'data/{split}.npz')
        np.testing.assert_array_equal(data['seeds'], seeds)
        assert data['states'].shape == (len(seeds), 3, 2) and data['costs'].shape == (len(seeds), 37)
    models = load_models(D, protocol['training']['seeds'])
    imitation = load_models(D, protocol['training']['seeds'], 'imitation.pt')
    data = np.load(D/'data/calibration.npz')
    action, predicted = predict(models, data['states'])
    actual = data['costs'][np.arange(len(action)), action]-data['costs'][:, 0]
    np.testing.assert_array_equal(action, cal['actions'])
    np.testing.assert_allclose(predicted, cal['predicted_excess_s'], atol=1e-6)
    np.testing.assert_allclose(actual, cal['actual_excess_s'], atol=1e-6)
    assert abs(upper_quantile(actual-predicted, cal['alpha'])-cal['upper_residual_quantile_s']) < 1e-6
    for seed in protocol['training']['seeds']:
        history = list(csv.DictReader((D/f'seed{seed}/training.csv').open()))
        selected = min((r for r in history if int(r['epoch']) >= 20), key=lambda r: float(r['validation_arrival_s']))
        checkpoint = torch.load(D/f'seed{seed}/policy.pt', map_location='cpu', weights_only=True)
        assert checkpoint['epoch'] == int(selected['epoch'])
    rows = list(csv.DictReader((D/'study/episodes.csv').open()))
    lookup = {(r['method'], r['scenario'], int(r['seed'])): r for r in rows}
    expected = {(m, s, seed) for m in protocol['methods'] for s, seeds in protocol['test_seeds'].items() for seed in seeds}
    assert len(rows) == 720 and set(lookup) == expected
    labels = json.loads((D/'study/teacher_labels.json').read_text())
    fixed = int(np.argmin(np.load(D/'data/train.npz')['costs'].mean(axis=0)))
    assert manifest['fixed_profile_from_training_only'] == fixed
    for label in labels:
        state = np.array(label['initial']); split, seed = label['scenario'], label['seed']
        for method in protocol['methods']:
            row = lookup[method, split, seed]
            if method in ('imitation', 'learned', 'calibrated'):
                selected, info = decide(imitation if method == 'imitation' else models, state,
                                        cal['upper_residual_quantile_s'] if method == 'calibrated' else None)
                assert selected == int(row['selected']) and info['proposal'] == int(row['proposal'])
            else:
                selected = fixed if method == 'fixed' else (int(np.argmin(label['costs'])) if method == 'teacher' else 0)
            assert selected == int(row['selected'])
            assert abs(float(row['mean_capped_finish_s'])-label['costs'][selected]) < 1e-6
            assert row['collision'] == 'False' and int(row['contract_violations']) == int(row['outside_steps']) == int(row['uncertified_steps']) == 0
    if replay:
        with MergeEnv(MergeConfig(**protocol['config']), 'constant') as env:
            for split, seeds in protocol['test_seeds'].items():
                for method in protocol['methods']:
                    env.reset(seeds[0], split); row = lookup[method, split, seeds[0]]
                    env.gate = PlannedGate(env.c, plan_episode(env.x, env.prev, env.c)) if method == 'teacher' else ProfileGate(env.c, int(row['selected']))
                    trace = json.loads((D/f'study/trace_{split}_{method}.json').read_text())
                    for expected_row in trace:
                        _, _, terminal, truncated, metrics = env.step([4]*3)
                        np.testing.assert_allclose(env.x, expected_row['x'], atol=1e-6, rtol=0)
                        np.testing.assert_allclose(env.prev, expected_row['command'], atol=1e-7, rtol=0)
                    assert terminal or truncated
                    assert metrics['mean_capped_finish_s'] == float(row['mean_capped_finish_s'])
                    print(json.dumps({'method': method, 'split': split, 'matched_steps': len(trace)}), flush=True)
    print(f'Verified {len(hashes)} V6 artifacts, data separation, checkpoint selection, calibration and 720 decisions.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--replay', action='store_true')
    main(parser.parse_args().replay)
