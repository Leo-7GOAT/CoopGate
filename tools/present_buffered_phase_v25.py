"""Report the completed paired audit without changing its frozen evidence."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from audit_buffered_phase_v25 import comparable_row, comparable_trace
from profile_phase_runtime_v24 import digest, save

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'artifacts/v25/buffered_phase_sumo_audit'
OUT = ROOT / 'artifacts/v25/buffered_phase_presentation'


def main():
    source = AUDIT / 'results'
    manifest = json.loads((source / 'sha256.json').read_text())
    if set(manifest) != {p.name for p in source.iterdir() if p.is_file() and p.name != 'sha256.json'}:
        raise ValueError('Incomplete result manifest')
    for name, expected in manifest.items():
        if digest(source / name) != expected:
            raise ValueError('Changed audit result: ' + name)
    plan = json.loads((AUDIT / 'plan.json').read_text())
    report = json.loads((source / 'report.json').read_text())
    if (report['plan_sha256'] != digest(AUDIT / 'plan.json')
            or report['actual_sumo_runs'] != 24 or report['paired_cases'] != 12
            or not report['all_24_trajectories_match_reviewed_archive']
            or report['independent_review_counts']['physical_traces'] != 24):
        raise ValueError('Require the completed frozen 24-run audit')
    times = {name: [] for name in plan['implementations']}
    for index, (case, pair) in enumerate(zip(plan['cases'], report['pairs'], strict=True)):
        rows = {}
        for implementation in plan['implementations']:
            item = json.loads((source / f'case_{index}_{implementation}.json').read_text())
            if item['case'] != case or item['implementation'] != implementation:
                raise ValueError('Substituted paired case')
            rows[implementation] = item
            values = item['row']['controller_ms']
            np.testing.assert_equal(values, [step['gate_ms'] for step in item['trace']])
            np.testing.assert_equal(sum(values), pair[f'{implementation}_select_ms'])
            times[implementation].extend(values)
        np.testing.assert_equal(comparable_row(rows['reference']['row']), comparable_row(rows['buffered']['row']))
        np.testing.assert_equal(comparable_trace(rows['reference']['trace']), comparable_trace(rows['buffered']['trace']))
        if not pair['all_execution_and_admission_fields_exact'] or pair['case'] != case:
            raise ValueError('Incomplete paired comparison')
    grouped = []
    for mode in ('clean', 'uniform', 'bias', 'pulse', 'all'):
        pairs = [p for p in report['pairs'] if mode == 'all' or p['case'][1] == mode]
        ref = sum(p['reference_select_ms'] for p in pairs)
        new = sum(p['buffered_select_ms'] for p in pairs)
        grouped.append(dict(mode=mode, decisions=sum(p['decisions'] for p in pairs),
            adopted_steps=sum(p['buffered_adopted_steps'] for p in pairs),
            reference_select_ms=ref, buffered_select_ms=new, reduction_percent=100 * (1-new/ref)))
    latency = {}
    for name, values in times.items():
        latency[name] = dict(decisions=len(values),
            p50_p95_p99_max_ms=np.percentile(values, [50, 95, 99, 100]).tolist(),
            over_500ms=sum(value > 500 for value in values))
    work = {name: sum(p[f'{name}_model_transitions'] for p in report['pairs']) for name in times}
    np.testing.assert_equal(work['reference'], work['buffered'])
    output = dict(source_plan_sha256=digest(AUDIT / 'plan.json'),
        source_report_sha256=digest(source / 'report.json'),
        source_manifest_sha256=digest(source / 'sha256.json'),
        actual_sumo_runs=24, paired_cases=12, old_initials=1, new_independent_initials=0,
        active_calibration_outcomes_used=False, all_non_timing_fields_exact=True,
        group_timings=grouped, sample_latency=latency, model_transitions=work,
        independent_review_counts=report['independent_review_counts'],
        execution_observation=dict(session_id=89137, exit_code=0,
            source='Actual exec/write_stdin completion observed before presentation; no reconstructed native exit receipt'),
        timing_scope=report['timing_scope'], new_risk_certificate=False, hard_real_time_certified=False)
    OUT.mkdir(exist_ok=False)
    save(OUT / 'report.json', output)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), layout='constrained')
    x = np.arange(4)
    axes[0].bar(x-.18, [r['reference_select_ms']/1000 for r in grouped[:4]], .36, label='Reference', color='#65788a')
    axes[0].bar(x+.18, [r['buffered_select_ms']/1000 for r in grouped[:4]], .36, label='Buffered', color='#167e9b')
    axes[0].set_xticks(x, [r['mode'].title() for r in grouped[:4]])
    axes[0].set(ylabel='Total select time (s)', title='Paired controller computation')
    axes[0].legend(frameon=False)
    for name, color in (('reference', '#65788a'), ('buffered', '#167e9b')):
        values = np.sort(times[name])
        axes[1].step(values, np.arange(1, len(values)+1)/len(values), where='post', color=color, label=name.title())
    axes[1].axvline(500, color='#bc4c3a', linestyle='--', linewidth=1, label='500 ms period')
    axes[1].set(xscale='log', xlabel='Select time (ms, logarithmic)', ylabel='Empirical fraction', title='277 paired decisions per implementation')
    axes[1].legend(frameon=False, fontsize=8, loc='lower right')
    for ax in axes:
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle('Same decisions and full checks; measured time reduced 75.4%\nOne old initial, 3 fixed models, 4 modes; concurrent load; no real-time certification', fontsize=10)
    fig.savefig(OUT / 'paired_gate_timing.png', dpi=180)
    fig.savefig(OUT / 'paired_gate_timing.svg')
    plt.close(fig)
    save(OUT / 'sha256.json', {p.name: digest(p) for p in OUT.iterdir() if p.is_file()})
    print(json.dumps(output, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
