"""Audit frozen provenance, full cost equivalence, pruning and SUMO replay."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from coopgate.experiment_v8 import prepare
from coopgate.fast_v8 import learned_order, plan_fast
from coopgate.learning_v6 import load_models, predict
from coopgate.merge_v4 import MergeConfig, MergeEnv

ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'artifacts/v8'


def main(replay):
    hashes = json.loads((D/'SHA256SUMS.json').read_text())
    for name, digest in hashes.items():
        assert hashlib.sha256((D/name).read_bytes()).hexdigest() == digest, name
    protocol = json.loads((D/'protocol.json').read_text())
    manifest = json.loads((D/'study/manifest.json').read_text())
    for source in (protocol, manifest):
        for name, digest in source['source_sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    new_seeds = [seed for group in protocol['test_seeds'].values() for seed in group]
    assert len(set(new_seeds)) == len(new_seeds) == 200
    for version in ('v6', 'v7', 'v7_pair'):
        old = json.loads((ROOT/f'artifacts/{version}/protocol.json').read_text())
        old_seeds = [seed for key in ('data_seeds', 'calibration_seeds', 'test_seeds') for group in old.get(key, {}).values() for seed in group]
        assert not set(new_seeds).intersection(old_seeds)
    rows = list(csv.DictReader((D/'study/episodes.csv').open()))
    lookup = {(r['method'], r['scenario'], int(r['seed'])): r for r in rows}
    expected = {(m, s, seed) for m in protocol['methods'] for s, seeds in protocol['test_seeds'].items() for seed in seeds}
    assert len(rows) == len(lookup) == protocol['executions'] == manifest['episodes'] == 1400 and set(lookup) == expected
    labels = json.loads((D/'study/teacher_labels.json').read_text())
    assert len(labels) == 200 and {(r['scenario'], r['seed']) for r in labels} == {(s, seed) for s, group in protocol['test_seeds'].items() for seed in group}
    models = load_models(ROOT/'artifacts/v6', protocol['model_seeds']); c = MergeConfig(**protocol['config'])
    for label in labels:
        x = np.array(label['initial']); costs = np.array(label['costs']); scenario, seed = label['scenario'], label['seed']
        order = np.random.default_rng(seed+808).permutation(protocol['methods']).tolist()
        assert label['execution_order'] == order
        full, _ = plan_fast(x, np.zeros(3), c)
        np.testing.assert_array_equal(full.costs, costs)
        proposal, _ = predict(models, x[None]); proposal = int(proposal[0])
        for method in protocol['methods']:
            row = lookup[method, scenario, seed]
            expected_index = 0 if method == 'constant' else int(np.argmin(costs))
            if method in ('reference_pair', 'fast_pair'):
                expected_index = proposal if costs[proposal] < costs[0] else 0
            assert int(row['selected']) == expected_index
            assert int(row['execution_order']) == order.index(method)
            assert abs(float(row['mean_capped_finish_s'])-costs[expected_index]) < 1e-6
            assert row['completed'] == 'True' and row['collision'] == 'False'
            assert all(int(row[k]) == 0 for k in ('contract_violations', 'outside_steps', 'uncertified_steps', 'plan_fallback_count'))
            if method in ('fast_pair', 'fast_full', 'bb_natural', 'bb_learned'):
                selected, _, info = prepare(method, models, x, np.zeros(3), c)
                assert selected == expected_index
                assert info['certificate_calls'] == int(row['certificate_calls'])
                assert info['pruned_profiles'] == int(row['pruned_profiles'])
    summary = json.loads((D/'summary.json').read_text())
    assert summary['all_criteria_passed'] and all(summary['criteria'].values()), 'Predeclared improvement criteria not met'
    assert set(summary['criteria']) == set(protocol['criteria'])
    timing = summary['timing']
    assert timing['fast_pair']['first_response_p95_ms']/timing['reference_pair']['first_response_p95_ms'] <= .2
    assert timing['bb_learned']['first_response_p95_ms']/timing['reference_pair']['first_response_p95_ms'] <= .2
    assert summary['mechanism']['learned_reduction_vs_full'] >= .3
    # Recompute published runtime percentiles from raw records, without assuming
    # a CI host should reproduce the original workstation's absolute latency.
    for method in protocol['methods']:
        group = [r for r in rows if r['method'] == method]
        np.testing.assert_allclose(np.percentile([float(r['first_response_ms']) for r in group], 95),
                                   timing[method]['first_response_p95_ms'], atol=1e-10, rtol=0)
    if replay:
        with MergeEnv(c, 'constant') as env:
            for regime, seeds in protocol['test_seeds'].items():
                for method in protocol['methods']:
                    env.reset(seeds[0], regime)
                    selected, env.gate, _ = prepare(method, models, env.x, env.prev, env.c)
                    trace = json.loads((D/f'study/trace_{regime}_{method}.json').read_text())
                    for expected_row in trace:
                        _, _, done, truncated, metrics = env.step([4]*3)
                        np.testing.assert_allclose(env.x, expected_row['x'], atol=1e-6, rtol=0)
                        np.testing.assert_allclose(env.prev, expected_row['command'], atol=1e-7, rtol=0)
                    assert done or truncated
                    assert abs(metrics['mean_capped_finish_s']-float(lookup[method, regime, seeds[0]]['mean_capped_finish_s'])) < 1e-6
                    print(json.dumps({'regime': regime, 'method': method, 'matched_steps': len(trace)}), flush=True)
    print(f'Verified {len(hashes)} V8 artifacts, 7400 candidate costs, 1400 executions and all predeclared improvement criteria.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--replay', action='store_true')
    main(parser.parse_args().replay)
