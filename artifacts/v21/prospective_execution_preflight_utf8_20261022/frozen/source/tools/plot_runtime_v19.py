"""Visualize matched decisions without pooling distinct runtime studies."""
import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from probe_disturbed_gate_v19 import digest, save
from start_final_v19 import verify_manifest

ROOT = Path(__file__).resolve().parents[1]


def plot(output):
    source = ROOT / 'artifacts/v19/bounded_gate_audit'
    verify_manifest(source)
    plan = json.loads((source / 'plan.json').read_text())
    audit = json.loads((source / 'audit.json').read_text())
    rows = json.loads((source / 'episodes.json').read_text())
    lookup = {(r['initial_index'], r['disturbance'], r['controller'], r['implementation']): r for r in rows}
    timings = {}
    for implementation in ('cached', 'bounded'):
        timings[implementation] = np.array([value for i, mode, name in plan['cases']
                                            for value in lookup[i, mode, name, implementation]['controller_ms']])
        assert len(timings[implementation]) == audit['exact_paired_decision_steps']
        np.testing.assert_allclose(timings[implementation].sum(), audit['timings'][implementation]['total_controller_ms'])
    x, y = timings['cached'], timings['bounded']
    assert np.all(x > 0) and np.all(y > 0)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), layout='constrained')
    for key, label, color in [('cached', 'Exact cache', '#536078'), ('bounded', 'Cache + rejection bound', '#06766f')]:
        values = np.sort(timings[key])
        axes[0].step(values, np.arange(1, len(values) + 1) / len(values), where='post', color=color, label=label)
    axes[0].axvline(500, color='#b33a28', linestyle='--', linewidth=1.2, label='500 ms control period')
    axes[0].set(xscale='log', xlabel='Controller time per decision (ms)', ylabel='Empirical cumulative fraction',
                title='Matched latency distributions', ylim=(0, 1.02))
    axes[0].legend(loc='upper left', fontsize=8)
    axes[0].grid(alpha=.2, which='both')
    limit = [min(x.min(), y.min()) * .8, max(x.max(), y.max()) * 1.2]
    axes[1].scatter(x, y, s=10, alpha=.45, color='#06766f', edgecolors='none')
    axes[1].plot(limit, limit, color='#536078', linewidth=1, label='Equal time')
    axes[1].axhline(500, color='#b33a28', linestyle='--', linewidth=.8)
    axes[1].axvline(500, color='#b33a28', linestyle='--', linewidth=.8)
    axes[1].set(xscale='log', yscale='log', xlim=limit, ylim=limit,
                xlabel='Exact cache (ms)', ylabel='Cache + rejection bound (ms)', title='667 identical paired decisions')
    axes[1].grid(alpha=.2, which='both')
    axes[1].legend(loc='upper left', fontsize=8)
    fig.suptitle('58 SUMO executions: total controller time -6.84%; P95 703 to 648 ms\n'
                 'P99 and maximum did not improve. Concurrent experiments; no real-time guarantee.', fontsize=11)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    fig.savefig(output / 'paired_runtime.png', dpi=200)
    plt.close(fig)
    save(output / 'metadata.json', dict(source_audit_sha256=digest(source / 'audit.json'), paired_decisions=len(x),
         scope='Same bounded-gate audit only; no pooling with earlier cache benchmark',
         runtime_requirement_met=False))
    shutil.copy2(__file__, output / Path(__file__).name)
    save(output / 'sha256.json', {p.name: digest(p) for p in output.iterdir() if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps(dict(figure=str(output / 'paired_runtime.png'), paired_decisions=len(x))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output')
    args = parser.parse_args()
    plot(args.output)
