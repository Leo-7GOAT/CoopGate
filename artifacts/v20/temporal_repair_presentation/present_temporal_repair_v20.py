"""Present all completed paired temporal gates and action-repair models."""
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
PARTNER = ROOT / 'artifacts/v20/temporal_development'
STUDY = ROOT / 'artifacts/v20/temporal_repair_extension'
EXECUTION = ROOT / 'artifacts/v20/temporal_repair_extension_execution'
OUTPUT = ROOT / 'artifacts/v20/temporal_repair_presentation'
FAMILIES = ('portfolio0', 'alternating', 'sampled', 'combined', 'repair')
LABELS = ('Original\n17', 'Alternating\n33', 'Sampled\n33', 'Combined\n49', 'Repaired\n33')
COLORS = ('#687787', '#0072B2', '#E69F00', '#9467BD', '#009E73')


def run():
    if OUTPUT.exists():
        raise FileExistsError('Preserve the completed figure and provenance')
    for directory in (PARTNER, STUDY, EXECUTION):
        verify_manifest(directory)
    state = json.loads((EXECUTION / 'state.json').read_text())
    native = json.loads((EXECUTION / 'supervisor_native_exit.json').read_text())
    finalizer = json.loads((EXECUTION / 'finalize_exit.json').read_text())
    if state['phase'] != 'completed' or native['exit_code'] != 0 or finalizer['exit_code'] != 0:
        raise ValueError('Native collection and independent review must finish successfully')
    if state['summary_sha256'] != digest(STUDY / 'summary.json'):
        raise ValueError('Completion receipt targets a different summary')
    report = json.loads((STUDY / 'summary.json').read_text())
    if report['partner_manifest_sha256'] != digest(PARTNER / 'sha256.json'):
        raise ValueError('Paired study identity changed')
    protocol = json.loads((PARTNER / 'protocol.json').read_text())
    spec = protocol['spec']
    rows = json.loads((PARTNER / 'episodes.json').read_text())
    bases = {(r['initial_index'], r['disturbance']): r['team_cost'] for r in rows if r['controller'] == 'baseline'}
    selected = [spec['modes'][int(np.random.default_rng(np.random.SeedSequence(
        [spec['seed_base'] + i, spec['mode_selection_stream']])).integers(len(spec['modes'])))]
        for i in range(spec['n_initials'])]
    base_mean = float(np.mean([bases[i, mode] for i, mode in enumerate(selected)]))
    n = report['independent_initials']
    if n != spec['n_initials'] or report['additional_independent_initials'] != 0:
        raise ValueError('Paired sample unit changed')
    records = {str(seed): {family: report['results'][f'{family}_{seed}'] for family in FAMILIES}
               for seed in (1802, 1812, 1822)}
    for seed in records:
        for family, result in records[seed].items():
            independent = report['independently_recounted_outcomes'][f'{family}_{seed}']
            for key, value in independent.items():
                np.testing.assert_allclose(result[key], value, atol=1e-12, rtol=1e-12)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(3, 3, figsize=(14.2, 10.4), gridspec_kw={'height_ratios': [1.15, 1., .85]})
    xs = np.arange(len(FAMILIES))
    gain_lows, gain_highs, adoption_rates = [], [], []
    for records_for_model in records.values():
        for row in records_for_model.values():
            gain_lows.append(row['actual_mean_gain_pointwise_bootstrap95'][0])
            gain_highs.append(row['actual_mean_gain_pointwise_bootstrap95'][1])
            adoption_rates.append(100. * row['actual_adopted'] / n)
    gain_range = (min(0., min(gain_lows)) - .007, max(gain_highs) * 1.18)
    adoption_max = max(30., max(adoption_rates) * 1.2)
    for col, (seed, methods) in enumerate(records.items()):
        top, mid, table_ax = axes[:, col]
        gain = np.asarray([methods[f]['actual_mean_gain'] for f in FAMILIES])
        ci = np.asarray([methods[f]['actual_mean_gain_pointwise_bootstrap95'] for f in FAMILIES])
        top.bar(xs, gain, color=COLORS, width=.65)
        top.vlines(xs, ci[:, 0], ci[:, 1], color='#26313b', linewidth=1.1)
        top.hlines(ci[:, 0], xs - .08, xs + .08, color='#26313b', linewidth=1.1)
        top.hlines(ci[:, 1], xs - .08, xs + .08, color='#26313b', linewidth=1.1)
        top.axhline(0., color='#6b7280', linewidth=.7)
        top.axhline(.005 * base_mean, color='#bc3d3d', linestyle='--', linewidth=1., label='0.5% of observed baseline mean')
        top.set_title(f'Training model {seed}', fontsize=12, fontweight='bold')
        top.set_ylim(*gain_range)
        top.set_ylabel('Mean cost improvement (s)')
        top.set_xticks(xs, LABELS, fontsize=8.5)
        top.grid(axis='y', alpha=.2)
        top.set_axisbelow(True)
        adopted = np.asarray([methods[f]['actual_adopted'] for f in FAMILIES])
        rate = 100. * adopted / n
        mid.bar(xs, rate, color=COLORS, width=.65)
        mid.axhline(20., color='#bc3d3d', linestyle='--', linewidth=1.)
        for x, y, count in zip(xs, rate, adopted):
            mid.text(x, y + .8, f'{count}/{n}', ha='center', va='bottom', fontsize=8.5)
        mid.set_ylim(0., adoption_max)
        mid.set_ylabel('Episodes using learning (%)')
        mid.set_xticks(xs, LABELS, fontsize=8.5)
        mid.grid(axis='y', alpha=.2)
        mid.set_axisbelow(True)
        table_ax.axis('off')
        cells = [[label.replace('\n', ' '), f"{methods[f]['actual_harmful_adopted']}/{methods[f]['actual_adopted']}",
                  f"{methods[f]['bundle_harmful_adopted']}/{methods[f]['bundle_adopted']}"]
                 for f, label in zip(FAMILIES, LABELS)]
        table = table_ax.table(cellText=cells, colLabels=('Gate', 'Actual H/A', '4-mode H/A'),
                               cellLoc='center', loc='upper center', colWidths=[.42, .28, .30])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1., 1.35)
        for (row, _), cell in table.get_celld().items():
            cell.set_edgecolor('#d7dde3')
            if row == 0:
                cell.set_facecolor('#e9edf2')
                cell.set_text_props(weight='bold')
        table_ax.set_title('Observed harmful/adopted initials', fontsize=10, pad=9)
    fig.suptitle('Temporal gating and action repair: complete paired development', fontsize=15, fontweight='bold', y=.98)
    fig.text(.5, .943, f'{n} independent initials; all three models; no risk certification', ha='center', fontsize=11)
    fig.text(.5, .025, 'Error bars: pointwise 95% paired-initial bootstrap intervals, in seconds.\n'
             'Dashed: 0.5% of observed baseline cost (top), 20% adoption (middle). Counts share initials.\n'
             'A=0 would leave conditional risk undefined. Development selection requires fresh calibration and confirmation.',
             ha='center', fontsize=9, color='#4b5563', linespacing=1.4)
    fig.subplots_adjust(left=.065, right=.985, top=.895, bottom=.11, hspace=.42, wspace=.28)
    OUTPUT.mkdir(parents=True)
    for suffix in ('png', 'pdf'):
        fig.savefig(OUTPUT / f'paired_temporal_repair.{suffix}', dpi=180, facecolor='white')
    plt.close(fig)
    save(OUTPUT / 'presentation.json', dict(independent_initials=n, total_unique_paired_sumo_runs=report['total_unique_paired_sumo_runs'],
        baseline_actual_mean_cost=base_mean, records=records, risk_certified=False,
        intervals='Frozen pointwise development mean-gain bootstrap intervals in seconds; no ratio or post-selection interval claim',
        source_sha256=digest(__file__), study_manifest_sha256=digest(STUDY / 'sha256.json'),
        execution_manifest_sha256=digest(EXECUTION / 'sha256.json')))
    shutil.copy2(__file__, OUTPUT / Path(__file__).name)
    save(OUTPUT / 'sha256.json', {p.name: digest(p) for p in OUTPUT.iterdir() if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps(dict(output=str(OUTPUT), independent_initials=n, risk_certified=False)), flush=True)


if __name__ == '__main__':
    run()
