"""Paired development analysis, including negative tradeoffs and noncertification."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from analyze_independent_v18 import bootstrap_indices
from probe_disturbed_gate_v19 import digest, save


def analyze(study, out):
    study, out = Path(study), Path(out)
    protocol = json.loads((study / 'protocol.json').read_text())
    summary = json.loads((study / 'summary.json').read_text())
    assert summary['episodes_sha256'] == digest(study / 'episodes.json')
    assert summary['protocol_sha256'] == digest(study / 'protocol.json')
    for path, expected in json.loads((study / 'sha256.json').read_text()).items():
        assert digest(study / path) == expected, path
    rows = json.loads((study / 'episodes.json').read_text())
    spec = protocol['spec']
    n = spec['n_initials']
    modes = spec['modes']
    methods = list(summary['results'])
    lookup = {(r['initial_index'], r['disturbance'], r['controller']): r for r in rows}
    assert len(lookup) == len(rows) == spec['expected_actual_sumo_runs']
    arrays = {k: np.array([[np.column_stack((lookup[i, m, k]['per_vehicle_cost'],
                                            lookup[i, m, k]['per_vehicle_finish']))
                            for m in modes] for i in range(n)]) for k in methods}
    regimes = np.array([lookup[i, modes[0], 'baseline']['scenario'] for i in range(n)])
    families = sorted({k.rsplit('_', 1)[0] for k in methods if k != 'baseline'})
    results = {}
    for family in families:
        members = [k for k in methods if k.startswith(family + '_')]
        differences = np.stack([arrays['baseline'] - arrays[k] for k in members])
        harmful = np.any(differences < -1e-9, axis=(2, 3, 4))
        admitted = np.array([[any(lookup[i, m, k]['learned_action_steps'] > 0 for m in modes)
                              for i in range(n)] for k in members])
        if family == 'raw':
            # The first development executor's action-step counter only tracked
            # performance gates. Raw always executes the learned policy itself.
            admitted[:] = True
        delta = differences.mean(0).mean(1)
        team = delta[:, 0:3, 0].mean(1)
        indices = bootstrap_indices(regimes, np.ones(n, bool), np.random.default_rng(19267))
        subset = [r for r in rows if r['controller'] in members]
        timings = [value for r in subset for value in r['controller_ms']]
        results[family] = dict(
            members=members, independent_initials=n, policy_initial_pairs=n * len(members),
            mean_team_gain=float(team.mean()),
            mean_team_gain_percent=float(100. * team.mean() / arrays['baseline'][:, :, :, 0].mean()),
            mean_team_gain_ci95=np.quantile(team[indices].mean(1), [.025, .975]).tolist(),
            per_vehicle_cost_gain=delta[:, :, 0].mean(0).tolist(),
            per_vehicle_time_gain=delta[:, :, 1].mean(0).tolist(),
            adopted_policy_initial_pairs=int(admitted.sum()), harmful_policy_initial_pairs=int(harmful.sum()),
            harmful_adopted_pairs=int((harmful & admitted).sum()),
            conditional_harm_descriptive=float((harmful & admitted).sum() / admitted.sum()) if admitted.any() else None,
            completed=sum(r['completed'] for r in subset), runs=len(subset),
            collisions=sum(r['collision'] for r in subset), aborted=sum(r['aborted'] for r in subset),
            contract_violations=sum(r['contract_violations'] for r in subset),
            controller_p95_ms=float(np.percentile(timings, 95)), controller_p99_ms=float(np.percentile(timings, 99)),
            controller_max_ms=float(max(timings)),
            steps_over_500ms=sum(value > 500. for value in timings), total_control_steps=len(timings))
    record = dict(purpose=spec['purpose'], actual_sumo_runs=len(rows), results=results,
                  risk_certified=False,
                  interval_scope='Development data, conditional on three fixed models; average modes/models within independent initial, then stratified initial-cluster bootstrap. No training-seed population interval.',
                  risk_denominator='Model-initial counts are descriptive and correlated; not binomial independent sample counts.',
                  study_hashes={name: digest(study / name) for name in ('protocol.json', 'episodes.json', 'summary.json', 'sha256.json')})
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'analysis.json', record)
    shutil.copy2(Path(__file__), out / Path(__file__).name)
    plot(record, out)
    save(out / 'sha256.json', {p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps(record), flush=True)


def plot(record, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = list(record['results'])
    data = [record['results'][name] for name in names]
    labels = {'raw': 'Raw MARL', 'nominal': 'Nominal gate', 'bank': 'Full-action bank',
              'portfolio0': 'Joint portfolio: strict', 'portfolio1': 'Joint portfolio: 1 failure'}
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.5), layout='constrained')
    colors = ['#608b9a', '#214e64', '#cb8142', '#607c4c'][:len(names)]
    axes[0].bar(x, [r['mean_team_gain_percent'] for r in data], color=colors)
    axes[0].set_ylabel('Mean team cost improvement (%)')
    axes[0].axhline(0, color='#888', lw=.8)
    axes[1].bar(x - .18, [r['adopted_policy_initial_pairs'] / r['policy_initial_pairs'] for r in data],
                width=.34, color='#214e64', label='Actual scene use')
    axes[1].bar(x + .18, [r['conditional_harm_descriptive'] or 0. for r in data],
                width=.34, color='#cb8142', label='Harm among used scenes')
    axes[1].set_ylabel('Descriptive fraction (no risk certificate)')
    axes[1].set_ylim(0., 1.05)
    axes[1].legend(fontsize=8)
    axes[2].bar(x, [r['controller_p95_ms'] for r in data], color=colors)
    axes[2].axhline(500., color='#a33d3d', linestyle='--', label='Decision period: 500 ms')
    axes[2].set_ylabel('Measured controller p95 (ms)')
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(x, [labels.get(name, name) for name in names], rotation=17, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=.15)
    fig.suptitle(f"Development tradeoffs: {record['actual_sumo_runs']} actual SUMO runs\nAll three trained models retained; not an independent risk certification", fontsize=11)
    fig.savefig(out / 'gate_tradeoffs.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('study')
    parser.add_argument('out')
    args = parser.parse_args()
    analyze(args.study, args.out)
