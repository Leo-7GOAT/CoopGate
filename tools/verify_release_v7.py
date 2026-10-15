"""Verify both V7 protocols, calibration decisions and real SUMO traces."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from coopgate.learning_v6 import ProfileGate, load_models, predict, upper_quantile
from coopgate.merge_v4 import MergeConfig, MergeEnv
from coopgate.planning_v5 import PlannedGate, plan_episode, portfolio
from coopgate.risk_v7 import calibrate_threshold, decide

ROOT = Path(__file__).resolve().parents[1]


def check_sources(value):
    for name, digest in value['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name


def main(replay=False):
    d = ROOT/'artifacts/v7'; protocol = json.loads((d/'protocol.json').read_text())
    cal = json.loads((d/'calibration.json').read_text()); check_sources(cal)
    models = load_models(ROOT/'artifacts/v6', protocol['model_seeds'])
    groups = {s: dict(np.load(d/f'data/{s}.npz')) for s in protocol['calibration_seeds']}
    for regime, data in groups.items():
        np.testing.assert_array_equal(data['seeds'], protocol['calibration_seeds'][regime])
        proposal, estimate = predict(models, data['states'])
        np.testing.assert_array_equal(proposal, data['proposal'])
        np.testing.assert_allclose(estimate, data['predicted'], atol=2e-6, rtol=0)
        np.testing.assert_allclose(data['actual'], data['costs'][:, 1]-data['costs'][:, 0], atol=1e-10)
        assert upper_quantile(data['actual']-data['predicted'], protocol['risk']) == cal['conformal_quantiles_s'][regime]
    assert cal['max_conformal_quantile_s'] == max(cal['conformal_quantiles_s'].values())
    recalculated = calibrate_threshold(groups, protocol['thresholds_s'], protocol['risk'], protocol['delta'], protocol['tolerance_s'])
    # beta.ppf may differ by floating-point roundoff between SciPy/platforms.
    # Bound values get a tight absolute tolerance; discrete counts, certification
    # flags and the final deployed threshold must still match exactly.
    errors = []
    for current, saved in zip(recalculated['candidates'], cal['pac']['candidates'], strict=True):
        for regime in current['regimes']:
            a, b = current['regimes'][regime], saved['regimes'][regime]
            errors.append(abs(a['upper']-b['upper']))
            np.testing.assert_allclose(a['upper'], b['upper'], rtol=0, atol=1e-12)
            a['upper'] = b['upper']
    assert recalculated == cal['pac']
    print(f'Max recalculated binomial-bound roundoff: {max(errors):.3g}', flush=True)
    old = json.loads((ROOT/'artifacts/v6/protocol.json').read_text())
    all_seeds = [s for group in [*old['data_seeds'].values(), *old['test_seeds'].values(), *protocol['calibration_seeds'].values()] for s in group]
    old_q = json.loads((ROOT/'artifacts/v6/calibration.json').read_text())['upper_residual_quantile_s']
    for version in ('v7', 'v7_pair'):
        d = ROOT/f'artifacts/{version}'
        hashes = json.loads((d/'SHA256SUMS.json').read_text())
        for name, digest in hashes.items():
            assert hashlib.sha256((d/name).read_bytes()).hexdigest() == digest, name
        protocol = json.loads((d/'protocol.json').read_text()); check_sources(protocol)
        manifest = json.loads((d/'study/manifest.json').read_text()); check_sources(manifest)
        all_seeds += [s for group in protocol['test_seeds'].values() for s in group]
        rows = list(csv.DictReader((d/'study/episodes.csv').open()))
        lookup = {(r['method'], r['scenario'], int(r['seed'])): r for r in rows}
        expected = {(m, s, seed) for m in protocol['methods'] for s, seeds in protocol['test_seeds'].items() for seed in seeds}
        assert len(rows) == len(lookup) == manifest['episodes'] == protocol['executions'] and set(lookup) == expected
        labels = json.loads((d/'study/teacher_labels.json').read_text())
        assert len(labels) == sum(map(len, protocol['test_seeds'].values()))
        for label in labels:
            state = np.array(label['initial']); costs = np.array(label['costs'])
            action, _ = predict(models, state[None]); proposal = int(action[0])
            for method in protocol['methods']:
                row = lookup[method, label['scenario'], label['seed']]
                if method == 'teacher':
                    selected = int(np.argmin(costs))
                elif version == 'v7':
                    selected, info = decide(models, state, method, cal, old_q)
                    assert int(row['proposal']) == info['proposal']
                elif method == 'verified_pair':
                    selected = proposal if costs[proposal] < costs[0] else 0
                    assert abs(float(row['verified_excess_s'])-(costs[proposal]-costs[0])) < 1e-6
                    assert int(row['plan_fallback_count']) == 0
                else:
                    selected = proposal if method == 'learned' else 0
                assert int(row['selected']) == selected
                assert abs(float(row['mean_capped_finish_s'])-costs[selected]) < 1e-6
                assert row['completed'] == 'True' and row['collision'] == 'False'
                assert int(row['contract_violations']) == int(row['outside_steps']) == int(row['uncertified_steps']) == 0
        if replay:
            with MergeEnv(MergeConfig(**protocol['config']), 'constant') as env:
                for regime, seeds in protocol['test_seeds'].items():
                    for method in protocol['methods']:
                        env.reset(seeds[0], regime); row = lookup[method, regime, seeds[0]]
                        if method == 'teacher':
                            env.gate = PlannedGate(env.c, plan_episode(env.x, env.prev, env.c))
                        elif method == 'verified_pair':
                            profiles = portfolio(); proposal = int(row['proposal'])
                            env.gate = PlannedGate(env.c, plan_episode(env.x, env.prev, env.c, profiles=[profiles[0], profiles[proposal]]))
                        else:
                            env.gate = ProfileGate(env.c, int(row['selected']))
                        trace = json.loads((d/f'study/trace_{regime}_{method}.json').read_text())
                        for expected_row in trace:
                            _, _, done, truncated, metrics = env.step([4]*3)
                            np.testing.assert_allclose(env.x, expected_row['x'], atol=1e-6, rtol=0)
                            np.testing.assert_allclose(env.prev, expected_row['command'], atol=1e-7, rtol=0)
                        assert done or truncated
                        assert abs(metrics['mean_capped_finish_s']-float(row['mean_capped_finish_s'])) < 1e-6
                        print(json.dumps({'version': version, 'regime': regime, 'method': method, 'matched_steps': len(trace)}), flush=True)
        print(f"Verified {version}: {len(hashes)} artifacts and {len(rows)} held-out decisions.", flush=True)
    assert len(all_seeds) == len(set(all_seeds)) == 2536
    print('Calibration/test splits and prior V6 data are disjoint.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--replay', action='store_true')
    main(parser.parse_args().replay)
