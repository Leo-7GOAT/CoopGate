"""Two explicit adoption-conditioned risk targets on frozen IID SUMO validation.

Never run this to certify a development study. Selecting a controller after
certification requires a separate final performance test on new initial seeds.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from coopgate.risk_comparators_v19 import conditional_ltt, paired_mean_improvement_bands
from probe_disturbed_gate_v19 import digest, save


def certify(study, output):
    study, output = Path(study), Path(output)
    if output.exists():
        raise FileExistsError('Keep earlier calibration results immutable')
    protocol = json.loads((study / 'protocol.json').read_text())
    spec = protocol['spec']
    if spec.get('data_role') != 'risk_validation' or spec['sampling'] != 'iid_mixture':
        raise ValueError('Only a frozen, independent IID risk-validation study qualifies')
    config = spec['risk_validation']
    if config['targets'] != ['actual_random_mode', 'worst_mode_bundle']:
        raise ValueError('Both adoption-conditioned targets are required')
    if config['method'] != 'bonferroni':
        raise ValueError('This prespecified study uses Bonferroni')
    if config['mode_selection_stream'] != 19297:
        raise ValueError('Unexpected outcome-independent mode assignment')
    for path, expected in config['statistical_source_sha256'].items():
        root = Path(__file__).resolve().parents[1]
        if digest(root / path) != expected:
            raise ValueError('Statistical analysis changed after freezing: ' + path)
    summary = json.loads((study / 'summary.json').read_text())
    assert summary['episodes_sha256'] == digest(study / 'episodes.json')
    assert summary['protocol_sha256'] == digest(study / 'protocol.json')
    for path, expected in json.loads((study / 'sha256.json').read_text()).items():
        assert digest(study / path) == expected, path
    rows = json.loads((study / 'episodes.json').read_text())
    lookup = {(r['initial_index'], r['disturbance'], r['controller']): r for r in rows}
    n, modes = spec['n_initials'], spec['modes']
    names = [name for name in spec['methods'] if name != 'baseline']
    assert len(lookup) == len(rows) == spec['expected_actual_sumo_runs']
    expected = {(i, mode, name) for i in range(n) for mode in modes for name in spec['methods']}
    assert set(lookup) == expected
    adopted = np.zeros((n, len(modes), len(names)), bool)
    harmful = np.zeros_like(adopted)
    gains = np.zeros(adopted.shape)
    for i in range(n):
        for mi, mode in enumerate(modes):
            base = lookup[i, mode, 'baseline']
            for j, name in enumerate(names):
                row = lookup[i, mode, name]
                np.testing.assert_allclose(row['initial'], base['initial'], atol=1e-8, rtol=0)
                delta = np.column_stack((row['per_vehicle_cost'], row['per_vehicle_finish'])) - np.column_stack(
                    (base['per_vehicle_cost'], base['per_vehicle_finish']))
                adopted[i, mi, j] = row['learned_action_steps'] > 0
                harmful[i, mi, j] = (np.any(delta > 1e-9) or not row['completed'] or row['collision'] or
                                     row['aborted'] or row['contract_violations'])
                gains[i, mi, j] = -delta[:, 0].mean()
    ids = list(range(spec['seed_base'], spec['seed_base'] + n))
    selected_modes = np.array([np.random.default_rng(np.random.SeedSequence([seed, 19297])).integers(len(modes)) for seed in ids])
    actual_a = adopted[np.arange(n), selected_modes]
    actual_h = harmful[np.arange(n), selected_modes]
    actual_gain = gains[np.arange(n), selected_modes]
    targets = {
        'actual_random_mode': conditional_ltt(ids, names, actual_a, actual_h, alpha=config['alpha'], delta=config['delta'] / 2),
        'worst_mode_bundle': conditional_ltt(ids, names, adopted.any(1), harmful.any(1), alpha=config['alpha'], delta=config['delta'] / 2)}
    certified = [name for j, name in enumerate(names) if all(t['candidates'][j]['certified'] for t in targets.values())]
    report = dict(purpose='Finite-sample conditional risk validation of fixed policies; final gain/usefulness test still required',
                  alpha=config['alpha'], total_delta=config['delta'], independent_initials=n,
                  targets=targets, both_targets_certified=certified,
                  actual_random_modes={str(seed): modes[int(m)] for seed, m in zip(ids, selected_modes)},
                  mean_improvement_comparator=paired_mean_improvement_bands(ids, names, actual_gain, delta=config['delta']),
                  model_records=protocol['models'], candidate_definitions=spec['methods'],
                  study_hashes={name: digest(study / name) for name in ('protocol.json', 'episodes.json', 'summary.json', 'sha256.json')},
                  scope='Nominal/dense equally likely IID initial generator and equally likely clean/uniform/bias/pulse mode; unknown parameters remain fixed as declared. The pressure-bundle target is separate. No per-state probability, arbitrary-noise guarantee, per-mode conditional guarantee, or distribution-shift guarantee.')
    save(output, report)
    print(json.dumps(dict(both_targets_certified=certified, independent_initials=n,
                          upper_actual={r['candidate']: r['threshold_upper'] for r in targets['actual_random_mode']['candidates']},
                          upper_bundle={r['candidate']: r['threshold_upper'] for r in targets['worst_mode_bundle']['candidates']})), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('study')
    parser.add_argument('output')
    args = parser.parse_args()
    certify(args.study, args.output)
