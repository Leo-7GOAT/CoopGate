"""Report complete paired-call timing, including CPU and speculative work."""
import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from audit_parallel_gate_v19 import same_decision
from probe_disturbed_gate_v19 import digest, save
from start_final_v19 import verify_manifest

ROOT = Path(__file__).resolve().parents[1]


def analyze(output):
    source = ROOT / 'artifacts/v19/parallel_gate_audit'
    verify_manifest(source)
    plan = json.loads((source / 'plan.json').read_text())
    audit = json.loads((source / 'audit.json').read_text())
    rows = json.loads((source / 'episodes.json').read_text())
    lookup = {(r['initial_index'], r['disturbance'], r['controller'], r['implementation']): r for r in rows}
    assert len(rows) == len(lookup) == 2 * len(plan['cases']) == 58
    measurements = {name: {'wall': [], 'cpu': []} for name in ('bounded', 'parallel')}
    by_model, admissions = {}, 0
    for ci, (initial, mode, controller) in enumerate(plan['cases']):
        traces = {name: json.loads((source / f'trace_{ci}_{name}.json').read_text()) for name in measurements}
        assert len(traces['bounded']) == len(traces['parallel'])
        for left, right in zip(traces['bounded'], traces['parallel']):
            for key in ('command', 'x', 't', 'disturbance'):
                assert left[key] == right[key], (ci, key)
            same_decision(left, right)
            admissions += int(right['portfolio_joint_action_admitted'])
        model = controller.rsplit('_', 1)[1]
        by_model.setdefault(model, {name: {'wall': [], 'cpu': []} for name in measurements})
        for name, trace in traces.items():
            row = lookup[initial, mode, controller, name]
            for kind, trace_key in (('wall', 'measured_select_wall_ms'), ('cpu', 'measured_select_process_cpu_ms')):
                values = [t[trace_key] for t in trace]
                np.testing.assert_array_equal(values, row[f'measured_{kind}_ms'])
                measurements[name][kind].extend(values)
                by_model[model][name][kind].extend(values)

    def summarize(data):
        result = {}
        for name, measured in data.items():
            wall, cpu = np.asarray(measured['wall']), np.asarray(measured['cpu'])
            assert np.all(wall > 0) and np.all(cpu >= 0)
            result[name] = dict(decisions=len(wall), total_wall_ms=float(wall.sum()), total_cpu_ms=float(cpu.sum()),
                                p50_ms=float(np.median(wall)), p95_ms=float(np.percentile(wall, 95)),
                                p99_ms=float(np.percentile(wall, 99)), max_ms=float(wall.max()),
                                over_500ms=int(np.sum(wall > 500)))
        left, right = result['bounded'], result['parallel']
        result['wall_reduction_fraction'] = 1 - right['total_wall_ms'] / left['total_wall_ms']
        result['cpu_increase_fraction'] = right['total_cpu_ms'] / left['total_cpu_ms'] - 1
        return result

    summary = summarize(measurements)
    x = np.asarray(measurements['bounded']['wall'])
    y = np.asarray(measurements['parallel']['wall'])
    assert len(x) == len(y) == audit['exact_paired_decisions'] == 667
    for name in measurements:
        np.testing.assert_allclose(summary[name]['total_wall_ms'], audit['timings'][name]['total_select_wall_ms'])
        np.testing.assert_allclose(summary[name]['total_cpu_ms'], audit['timings'][name]['total_select_process_cpu_ms'])
    report = dict(summary=summary, per_model={k: summarize(v) for k, v in by_model.items()},
                  paired_decisions=len(x), actual_sumo_replays=len(rows), exact_learning_admission_steps=admissions,
                  faster_paired_decisions=int(np.sum(y < x)), slower_paired_decisions=int(np.sum(y > x)),
                  bounded_only_deadline_misses=int(np.sum((x > 500) & (y <= 500))),
                  parallel_only_deadline_misses=int(np.sum((x <= 500) & (y > 500))),
                  both_deadline_misses=int(np.sum((x > 500) & (y > 500))),
                  source_audit_sha256=digest(source / 'audit.json'),
                  source_episodes_sha256=digest(source / 'episodes.json'),
                  source_plan_sha256=digest(source / 'plan.json'),
                  source_code_sha256=digest(Path(__file__)),
                  risk_certified=False, hard_real_time_certified=False,
                  scope='Post-hoc paired development replay audit. Full select wall time includes all submitted work and pool teardown. CPU is this process only, excluding the SUMO server. Concurrent frozen studies; selected cases are not a random latency sample. No new independent efficacy observations.')

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), layout='constrained')
    for values, label, color in ((x, 'Serial rejection bound', '#536078'), (y, 'Two-worker disturbance evaluation', '#06766f')):
        axes[0].step(np.sort(values), np.arange(1, len(values) + 1) / len(values), where='post', color=color, label=label)
    axes[0].axvline(500, color='#b33a28', linestyle='--', linewidth=1.2, label='500 ms control period')
    axes[0].set(xscale='log', xlabel='Complete select call (ms)', ylabel='Empirical cumulative fraction',
                title='Same 667 decisions', ylim=(0, 1.02))
    axes[0].legend(loc='upper left', fontsize=8)
    limits = [min(x.min(), y.min()) * .8, max(x.max(), y.max()) * 1.2]
    axes[1].scatter(x, y, s=11, alpha=.45, color='#06766f', edgecolors='none')
    axes[1].plot(limits, limits, color='#536078', linewidth=1, label='Equal time')
    axes[1].axhline(500, color='#b33a28', linestyle='--', linewidth=.8)
    axes[1].axvline(500, color='#b33a28', linestyle='--', linewidth=.8)
    axes[1].set(xscale='log', yscale='log', xlim=limits, ylim=limits,
                xlabel='Serial complete call (ms)', ylabel='Parallel complete call (ms)', title='Identical commands and admissions')
    axes[1].legend(loc='upper left', fontsize=8)
    for axis in axes:
        axis.grid(alpha=.2, which='both')
    fig.suptitle(f"58 SUMO replays: wall time {100 * summary['wall_reduction_fraction']:.1f}% lower; "
                 f"process CPU {100 * summary['cpu_increase_fraction']:+.1f}%\n"
                 'All 17 scenarios retained for admitted actions; concurrent load; no hard real-time guarantee.', fontsize=11)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    fig.savefig(output / 'paired_parallel_runtime.png', dpi=200)
    plt.close(fig)
    save(output / 'analysis.json', report)
    shutil.copy2(__file__, output / Path(__file__).name)
    save(output / 'sha256.json', {p.name: digest(p) for p in output.iterdir() if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output')
    args = parser.parse_args()
    analyze(args.output)
