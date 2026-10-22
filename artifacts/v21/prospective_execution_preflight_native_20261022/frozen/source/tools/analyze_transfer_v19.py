"""Paired direct-transfer analysis, retaining baseline failures and all models."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from probe_disturbed_gate_v19 import digest, save


def analyze(study, output):
    study, output = Path(study), Path(output)
    protocol = json.loads((study / 'protocol.json').read_text())
    spec = protocol['spec']
    if spec.get('data_role') != 'direct_transfer' or spec['mode'] != 'uniform':
        raise ValueError('Expected the frozen direct-transfer study')
    for path, expected in json.loads((study / 'sha256.json').read_text()).items():
        assert digest(study / path) == expected, path
    summary = json.loads((study / 'summary.json').read_text())
    assert summary['protocol_sha256'] == digest(study / 'protocol.json')
    assert summary['episodes_sha256'] == digest(study / 'episodes.json')
    rows = json.loads((study / 'episodes.json').read_text())
    assert len(rows) == spec['expected_actual_sumo_runs'] == summary['actual_sumo_runs']
    lookup = {(r['scenario'], r['initial_index'], r['plant_name'], r['controller']): r for r in rows}
    n = spec['n_initials_per_scenario']
    seeds = sorted(protocol['models'])
    assert len(lookup) == len(rows)
    bootstrap = np.random.default_rng(19457).integers(n, size=(5000, n))
    cells = []
    for scenario in spec['scenarios']:
        for plant in spec['plants']:
            base_rows = [lookup[scenario, i, plant, 'baseline'] for i in range(n)]
            base = np.array([np.column_stack((r['per_vehicle_cost'], r['per_vehicle_finish'])) for r in base_rows])
            for family in ('raw', 'nominal', 'portfolio0'):
                differences, used, harmed, data, per_model = [], [], [], [], []
                for seed in seeds:
                    name = family + '_' + seed
                    selected = [lookup[scenario, i, plant, name] for i in range(n)]
                    values = np.array([np.column_stack((r['per_vehicle_cost'], r['per_vehicle_finish'])) for r in selected])
                    delta = base - values
                    adopted = np.array([r['learned_action_steps'] > 0 for r in selected])
                    physical = np.array([not r['completed'] or r['collision'] or r['aborted'] or bool(r['contract_violations']) for r in selected])
                    harmful = np.any(delta < -1e-9, axis=(1, 2)) | physical
                    for actual, baseline in zip(selected, base_rows):
                        np.testing.assert_array_equal(actual['initial'], baseline['initial'])
                        assert actual['plant'] == baseline['plant']
                    differences.append(delta)
                    used.append(adopted)
                    harmed.append(harmful)
                    data.extend(selected)
                    per_model.append(dict(seed=seed, adopted=int(adopted.sum()), harmful=int(harmful.sum()),
                                          harmful_adopted=int((harmful & adopted).sum()),
                                          completed=sum(r['completed'] for r in selected),
                                          mean_team_gain_seconds=float(delta[:, :, 0].mean())))
                delta = np.array(differences).mean(0)
                team = delta[:, :, 0].mean(1)
                used, harmed = np.array(used), np.array(harmed)
                timings = [v for r in data for v in r['controller_ms']]
                cells.append(dict(
                    scenario=scenario, plant=plant, family=family, independent_initials=n,
                    fixed_models=seeds, policy_initial_pairs=n * len(seeds), per_model=per_model,
                    baseline_completed=sum(r['completed'] for r in base_rows), baseline_runs=n,
                    baseline_aborted=sum(r['aborted'] for r in base_rows),
                    baseline_collisions=sum(r['collision'] for r in base_rows),
                    baseline_contract_violations=sum(r['contract_violations'] for r in base_rows),
                    controller_completed=sum(r['completed'] for r in data), controller_runs=len(data),
                    controller_aborted=sum(r['aborted'] for r in data),
                    controller_collisions=sum(r['collision'] for r in data),
                    controller_contract_violations=sum(r['contract_violations'] for r in data),
                    adopted_policy_initial_pairs=int(used.sum()), harmful_policy_initial_pairs=int(harmed.sum()),
                    harmful_adopted_pairs=int((used & harmed).sum()),
                    conditional_harm_descriptive=float((used & harmed).sum() / used.sum()) if used.any() else None,
                    mean_team_gain_seconds=float(team.mean()),
                    mean_team_gain_percent=float(100 * team.mean() / base[:, :, 0].mean()),
                    mean_team_gain_ci95=np.quantile(team[bootstrap].mean(1), [.025, .975]).tolist(),
                    per_vehicle_cost_gain=delta[:, :, 0].mean(0).tolist(),
                    per_vehicle_time_gain=delta[:, :, 1].mean(0).tolist(),
                    declared_error_assumption_covers_entire_parameter_range=plant != 'lag_strong',
                    maximum_actual_error=max(r['max_actual_error'] for r in data),
                    error_contract_exceeded_runs=sum(r['error_contract_exceeded_substeps'] > 0 for r in data),
                    error_contract_exceeded_substeps=sum(r['error_contract_exceeded_substeps'] for r in data),
                    controller_p95_ms=float(np.percentile(timings, 95)) if timings else None,
                    controller_p99_ms=float(np.percentile(timings, 99)) if timings else None,
                    controller_max_ms=float(max(timings)) if timings else None,
                    steps_over_500ms=sum(t > 500 for t in timings), control_steps=len(timings)))
    report = dict(
        purpose='Direct-transfer range and failure analysis, with frozen weights and gates',
        actual_sumo_runs=len(rows), cells=cells, scenarios=spec['scenarios'], plants=spec['plants'],
        risk_certified=False, weights_or_thresholds_recalibrated=False,
        interval_scope='Pointwise exploratory initial-cluster bootstrap within each scenario/plant cell, conditional on the three fixed trained models; not simultaneous coverage across all cells or a training-seed population interval.',
        risk_scope='Correlated model-initial counts are descriptive; per-model rates retain n independent initials. No inherited IID conditional-risk certificate under distribution shift.',
        failure_scope='All runs, including capped 25 s noncompletion and baseline failures, remain in the primary mean. Never interpret a gain against a failed baseline as a physical safety guarantee.',
        study_sha256={name: digest(study / name) for name in ('protocol.json', 'episodes.json', 'summary.json', 'sha256.json')})
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'analysis.json', report)
    shutil.copy2(__file__, output / Path(__file__).name)
    plot(report, output)
    save(output / 'sha256.json', {p.name: digest(p) for p in output.iterdir() if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps(dict(actual_sumo_runs=len(rows), cells=len(cells), risk_certified=False)), flush=True)


def plot(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    lookup = {(c['scenario'], c['plant'], c['family']): c for c in report['cells']}
    families = ('raw', 'nominal', 'portfolio0')
    scenario_labels = ['Nominal starts', 'Dense starts', 'Slow yielding', 'Speed asymmetry', 'Tight headway']
    plant_labels = ['Nominal', 'Gain / drag', 'Mild lag', 'Strong lag*']
    limit = max(1., np.ceil(max(abs(c['mean_team_gain_percent']) for c in report['cells'])))
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 5.6), layout='constrained')
    for ax, family, title in zip(axes, families, ('Raw MARL', 'Nominal gate', 'Strict joint composition')):
        values = np.array([[lookup[s, p, family]['mean_team_gain_percent'] for p in report['plants']]
                           for s in report['scenarios']])
        im = ax.imshow(values, cmap='RdBu', norm=TwoSlopeNorm(vmin=-limit, vcenter=0., vmax=limit), aspect='auto')
        for i, scenario in enumerate(report['scenarios']):
            for j, plant in enumerate(report['plants']):
                cell = lookup[scenario, plant, family]
                incomplete = cell['baseline_completed'] < cell['baseline_runs'] or cell['controller_completed'] < cell['controller_runs']
                label = f"{values[i,j]:+.2f}%" + ('†' if incomplete else '')
                ax.text(j, i, label, ha='center', va='center', fontsize=9,
                        color='white' if abs(values[i, j]) > .55 * limit else '#111')
        ax.set_xticks(range(len(report['plants'])), plant_labels, rotation=22, ha='right', fontsize=9)
        ax.set_yticks(range(len(report['scenarios'])), scenario_labels, fontsize=9)
        ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=axes, label='Mean team cost improvement (%)', shrink=.75)
    fig.suptitle('Frozen direct transfer: all three trained models retained\n'
                 '† Some baseline/controller runs did not complete; capped costs retained.\n'
                 '* Strong lag exceeds the declared physical error envelope. No shifted-risk certificate.', fontsize=11)
    fig.savefig(output / 'direct_transfer.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('study')
    parser.add_argument('output')
    args = parser.parse_args()
    analyze(args.study, args.output)
