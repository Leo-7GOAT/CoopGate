"""Plot every completed comparison cell without rankings or new inference."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from execute_prospective_v23 import exclusive_json, utc
from profile_phase_runtime_v24 import committed, digest
import analyze_physical_switch_v27 as analysis

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/v27/physical_switch_presentation'


def save_figure(figure, name):
    figure.savefig(OUT/(name+'.png'), dpi=180, facecolor='white')
    figure.savefig(OUT/(name+'.svg'), facecolor='white')
    plt.close(figure)


def development(report):
    data = report['phase']['methods']
    names = [r['method'] for r in data]
    if len(names) != 19 or len(set(names)) != 19:
        raise ValueError('Every development method must be plotted')
    y = np.arange(len(data))
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 10), sharey=True,
                                gridspec_kw=dict(width_ratios=[1.4, 1, 1]))
    for i, item in enumerate(data):
        row = item['actual']
        low, high = row['gain_pointwise_bootstrap95']
        axes[0].plot([low, high], [i, i], color='#246b8e', lw=1.8)
        axes[0].plot(row['mean_gain_seconds'], i, 'o', color='#246b8e', ms=4)
        risk = row['empirical_conditional_risk']
        if risk is not None:
            axes[1].plot(100*risk, i, 'o', color='#9a4b3c', ms=4)
            axes[1].annotate(f" {row['harmful_adopted']}/{row['adopted']}",
                             (100*risk, i), fontsize=7, va='center')
        else:
            axes[1].text(.03, i, 'undefined (no adoption)', fontsize=7,
                         transform=axes[1].get_yaxis_transform(), va='center', color='#555555')
        axes[2].plot(100*row['adoption_rate'], i, 'o', color='#326c55', ms=4)
        axes[2].annotate(f" {row['adopted']}/{row['independent_initials']}",
                         (100*row['adoption_rate'], i), fontsize=7, va='center')
        if row['failed']:
            axes[0].annotate(f"  [{row['failed']} failed]", (row['mean_gain_seconds'], i),
                             fontsize=7, va='center', color='#a80000')
    axes[0].set_yticks(y, names, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].axvline(0, color='#444444', lw=.7)
    for axis, title in zip(axes, ['Baseline cost minus method cost (s)', 'Harmful / adopted (%)', 'Actual adoption (%)']):
        axis.set_title(title, fontsize=10, pad=12)
        axis.grid(axis='y', color='#dddddd', linewidth=.5)
        axis.set_axisbelow(True)
        axis.spines[['top', 'right']].set_visible(False)
    axes[1].set_xlim(-2, 123)
    axes[2].set_xlim(-2, 123)
    figure.suptitle('All 19 methods on 128 development initials', fontsize=15, y=.98)
    figure.text(.02, .025,
        'One preassigned actual disturbance mode per initial. Cost intervals: pointwise paired bootstrap 95%.\n'
        'Risk is an empirical conditional proportion, not a certificate. All failures retained; fixed method order.', fontsize=9)
    figure.subplots_adjust(left=.17, right=.98, top=.92, bottom=.1, wspace=.22)
    save_figure(figure, 'development_gain_risk_adoption')


def transfer(report, comparison, methods):
    spec = comparison['protocols']['transfer']['spec']
    columns = [(scene, plant) for scene in spec['scenarios'] for plant in spec['plants']]
    lookup = {(r['method'], r['scenario'], r['plant']):r for r in report['transfer']['cells']}
    expected = {(method, scene, plant) for method in methods for scene, plant in columns}
    if len(lookup) != 500 or set(lookup) != expected:
        raise ValueError('All 500 transfer cells must be plotted')
    settings = [
        ('mean_gain_percent', 'transfer_cost_gain', 'Mean cost gain relative to baseline (%)', 'RdBu', None),
        ('empirical_conditional_risk', 'transfer_conditional_risk', 'Empirical harmful / adopted (%)', 'YlOrRd', (0, 100)),
        ('adoption_rate', 'transfer_adoption', 'Actual-command adoption (%)', 'YlGnBu', (0, 100))]
    for field, filename, title, colors, limits in settings:
        values = np.full((len(methods), len(columns)), np.nan)
        labels = []
        for i, method in enumerate(methods):
            for j, (scene, plant) in enumerate(columns):
                row = lookup[method, scene, plant]
                value = row[field]
                if value is not None:
                    values[i, j] = value if field == 'mean_gain_percent' else 100*value
                labels.append((i, j, row))
        cmap = plt.get_cmap(colors).copy()
        cmap.set_bad('#e8e8e8')
        figure, axis = plt.subplots(figsize=(20, 13))
        kwargs = {}
        if limits:
            kwargs.update(vmin=limits[0], vmax=limits[1])
        else:
            # A symmetric full-data scale shows every observed value; no clipping.
            limit = max(float(np.nanmax(np.abs(values))), 1e-6)
            kwargs['norm'] = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
        im = axis.imshow(values, cmap=cmap, aspect='auto', **kwargs)
        for i, j, row in labels:
            value = values[i, j]
            label = 'N/A' if np.isnan(value) else f'{value:.1f}'
            if row['failed']:
                label += '*'
            rgba = cmap(im.norm(value)) if np.isfinite(value) else (1, 1, 1, 1)
            lightness = .2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
            axis.text(j, i, label, ha='center', va='center', fontsize=6.7,
                      color='black' if lightness > .52 else 'white')
        axis.set_yticks(np.arange(len(methods)), methods, fontsize=8)
        axis.set_xticks(np.arange(len(columns)), [plant for _, plant in columns], rotation=55, ha='right', fontsize=8)
        for k, scene in enumerate(spec['scenarios']):
            start, width = k*len(spec['plants']), len(spec['plants'])
            axis.text(start+(width-1)/2, 1.015, scene, transform=axis.get_xaxis_transform(),
                      ha='center', fontsize=9)
            if k:
                axis.axvline(start-.5, color='black', linewidth=1)
        figure.colorbar(im, ax=axis, fraction=.025, pad=.02, label=title)
        figure.suptitle(title+' — all 500 transfer cells', fontsize=16, y=.98)
        figure.text(.02, .018,
            'Each cell: 32 paired initials. * includes one or more failed executions; baseline failures are retained in the accompanying tables.\n'
            'Gray N/A: undefined conditional risk with no adoption. Plants and models do not add independent initials. No transfer risk certificate.', fontsize=10)
        figure.subplots_adjust(left=.15, right=.96, top=.9, bottom=.14)
        save_figure(figure, filename)


def main():
    committed(ROOT/'tools/present_physical_comparison_v27.py')
    source = analysis.OUT/'results/report.json'
    committed(source)
    plan = analysis.inputs()
    evidence = analysis.completion_gate(plan)
    report = analysis.read(source)
    manifest = analysis.read(source.parent/'sha256.json')
    if (digest(source) != manifest['report.json'] or report['plan_sha256'] != digest(analysis.OUT/'plan.json')
            or report['input_evidence_sha256'] != evidence):
        raise ValueError('Completed report or input evidence changed')
    OUT.mkdir(parents=True, exist_ok=False)
    development(report)
    transfer(report, analysis.read(analysis.physical.OUT/'plan.json'), plan['methods']['transfer'])
    exclusive_json(OUT/'provenance.json', dict(utc=utc(), source_report_sha256=digest(source),
        renderer_sha256=digest(ROOT/'tools/present_physical_comparison_v27.py'),
        plotted_development_methods=19, plotted_transfer_cells_per_figure=500,
        new_sumo_runs=0, new_independent_initials=0, new_inference=False))
    exclusive_json(OUT/'sha256.json', {p.name:digest(p) for p in OUT.iterdir() if p.is_file()})
    print(json.dumps(dict(figures=4, plotted_development_methods=19, transfer_cells=500)), flush=True)


if __name__ == '__main__':
    main()
