"""Post-development paired ablation; preserve baseline parity and all failures."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from analyze_independent_v18 import bootstrap_indices
from probe_disturbed_gate_v19 import digest, save


def read_study(directory):
    directory = Path(directory)
    for name, expected in json.loads((directory / 'sha256.json').read_text()).items():
        assert digest(directory / name) == expected, (directory, name)
    protocol = json.loads((directory / 'protocol.json').read_text())
    rows = json.loads((directory / 'episodes.json').read_text())
    lookup = {(r['initial_index'], r['disturbance'], r['controller']): r for r in rows}
    assert len(lookup) == len(rows) == protocol['spec']['expected_actual_sumo_runs']
    return directory, protocol, lookup


def compare(bank_directory, portfolio_directory, output):
    bank_dir, bp, bank = read_study(bank_directory)
    port_dir, pp, portfolio = read_study(portfolio_directory)
    for key in ('n_initials', 'seed_base', 'sampling', 'modes'):
        assert bp['spec'][key] == pp['spec'][key], key
    assert bp['models'] == pp['models']
    n, modes = pp['spec']['n_initials'], pp['spec']['modes']
    seeds = sorted(bp['models'])
    # Timing is intentionally excluded: it depends on machine load. All state,
    # physical outcome and performance fields must match the repeated baseline.
    fields = ('initial', 'per_vehicle_cost', 'per_vehicle_finish', 'per_vehicle_variation',
              'completed', 'collision', 'contract_violations', 'aborted', 'control_steps')
    for i in range(n):
        for mode in modes:
            left, right = bank[i, mode, 'baseline'], portfolio[i, mode, 'baseline']
            for field in fields:
                np.testing.assert_array_equal(left[field], right[field], err_msg=f'{i}/{mode}/{field}')
    arrays = {}
    use = {}
    harm = {}
    for family, lookup in (('bank', bank), ('portfolio0', portfolio)):
        arrays[family] = np.array([[[lookup[i, mode, f'{family}_{seed}']['per_vehicle_cost']
                                    for mode in modes] for i in range(n)] for seed in seeds])
        use[family] = np.array([[any(lookup[i, mode, f'{family}_{seed}']['learned_action_steps'] > 0
                                    for mode in modes) for i in range(n)] for seed in seeds])
        harm[family] = np.array([[any(
            np.any(np.array(lookup[i, mode, f'{family}_{seed}']['per_vehicle_cost']) >
                   np.array(lookup[i, mode, 'baseline']['per_vehicle_cost']) + 1e-9) or
            np.any(np.array(lookup[i, mode, f'{family}_{seed}']['per_vehicle_finish']) >
                   np.array(lookup[i, mode, 'baseline']['per_vehicle_finish']) + 1e-9) or
            not lookup[i, mode, f'{family}_{seed}']['completed'] or
            lookup[i, mode, f'{family}_{seed}']['collision'] or
            lookup[i, mode, f'{family}_{seed}']['aborted'] or
            lookup[i, mode, f'{family}_{seed}']['contract_violations']
            for mode in modes) for i in range(n)] for seed in seeds])
    gains = (arrays['bank'] - arrays['portfolio0']).mean((0, 2, 3))
    regimes = np.array([bank[i, modes[0], 'baseline']['scenario'] for i in range(n)])
    indices = bootstrap_indices(regimes, np.ones(n, bool), np.random.default_rng(19319))
    baseline_mean = np.mean([bank[i, m, 'baseline']['team_cost'] for i in range(n) for m in modes])
    composition_counts = {}
    for row in portfolio.values():
        if row['controller'].startswith('portfolio0_'):
            for code in row['chosen_compositions']:
                composition_counts[str(code)] = composition_counts.get(str(code), 0) + 1
    report = dict(
        purpose='Post-probe paired development ablation, not an independent confirmation or risk certificate',
        independent_initials=n, fixed_model_seeds=seeds, baseline_parity_pairs=n * len(modes),
        parity_fields=list(fields),
        extra_team_gain_seconds=float(gains.mean()),
        extra_team_gain_ci95=np.quantile(gains[indices].mean(1), [.025, .975]).tolist(),
        extra_team_gain_percentage_points=float(100 * gains.mean() / baseline_mean),
        composition_counts=composition_counts,
        adoption_transitions=dict(
            bank_only=int((use['bank'] & ~use['portfolio0']).sum()),
            portfolio_only=int((~use['bank'] & use['portfolio0']).sum()),
            both=int((use['bank'] & use['portfolio0']).sum()),
            neither=int((~use['bank'] & ~use['portfolio0']).sum())),
        families={name: dict(
            mean_team_cost=float(values.mean()),
            mean_team_gain_percent=float(100 * (baseline_mean - values.mean()) / baseline_mean),
            adopted_model_initial_pairs=int(use[name].sum()),
            harmful_model_initial_pairs=int(harm[name].sum()),
            harmful_adopted_pairs=int((harm[name] & use[name]).sum())) for name, values in arrays.items()},
        risk_certified=False,
        interval_scope='Modes/models averaged within each initial; stratified initial-cluster bootstrap, conditional on the three fixed models. Development analysis.',
        source_studies={str(directory): {name: digest(directory / name) for name in
                       ('protocol.json', 'episodes.json', 'sha256.json')} for directory in (bank_dir, port_dir)})
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'analysis.json', report)
    shutil.copy2(__file__, out / Path(__file__).name)
    save(out / 'sha256.json', {p.name: digest(p) for p in out.iterdir() if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bank')
    parser.add_argument('portfolio')
    parser.add_argument('output')
    args = parser.parse_args()
    compare(args.bank, args.portfolio, args.output)
