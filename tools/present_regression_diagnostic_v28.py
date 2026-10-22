"""Present every completed nominal admission and all selected failure mechanisms."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import diagnose_regression_v28 as diagnostic
from execute_prospective_v23 import exclusive_json, utc
from profile_phase_runtime_v24 import committed, digest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/v28/regression_presentation'


def verified_report():
    source = diagnostic.OUT/'analysis/report.json'
    committed(source)
    plan = diagnostic.inputs()
    offline, replays = diagnostic.completed(plan)
    report = diagnostic.read(source)
    hashes = diagnostic.read(source.parent/'sha256.json')
    local = [dict(case=list(key), **item) for key, _, trace in offline for item in trace]
    selected = [dict(case=list(key), **record, local_diagnostics=trace)
                for key, record, trace in offline if list(key) in plan['regression_cases']]
    expected = dict(nominal_episodes=len(offline), admitted_steps=len(local),
        adopted_episodes=sum(record['original_adopted'] for _, record, _ in offline),
        local_complete=sum(item['both_branches_complete'] for item in local),
        local_regressions=sum(item['realized_local_regression'] is True for item in local),
        local_incomplete=sum(not item['both_branches_complete'] for item in local),
        actual_sumo_replays=len(replays), replayed_commands=sum(len(trace) for _, _, trace in replays))
    if (digest(source) != hashes['report.json']
            or report['plan_sha256'] != digest(diagnostic.OUT/'plan.json')
            or report['actual_exit_sha256'] != digest(diagnostic.OUT/'execution/observation.json')
            or report['all_local_diagnostics'] != local or report['selected_regressions'] != selected
            or any(report[k] != value for k, value in expected.items())):
        raise ValueError('Full completed report must match sealed observations and actual exit evidence')
    if expected != dict(nominal_episodes=480, admitted_steps=87, adopted_episodes=81,
                        local_complete=87, local_regressions=4, local_incomplete=0,
                        actual_sumo_replays=12, replayed_commands=277):
        raise ValueError('Complete declared diagnostic scope required')
    return report, expected, source


def save(figure, name):
    for extension in ('png', 'svg'):
        figure.savefig(OUT/(name+'.'+extension), dpi=160, facecolor='white')
    plt.close(figure)


def admissions(report):
    items = report['all_local_diagnostics']
    index = np.arange(1, len(items)+1)
    bank_cost = np.array([np.asarray(item['original_bank_maximum'])[:, 0].max() for item in items])
    actual_cost = np.array([np.asarray(item['realized_difference'])[:, 0].max() for item in items])
    bank_phase = np.array([max(item['original_phase_maximum']) for item in items])
    actual_phase = np.array([max(item['realized_phase_delta']) for item in items])
    harmful = np.array([item['realized_local_regression'] for item in items])
    figure, axes = plt.subplots(2, 1, figsize=(13.6, 7.6), sharex=True)
    for axis, bank, actual, ylabel in zip(axes, (bank_cost, bank_phase), (actual_cost, actual_phase),
            ('Maximum per-vehicle cost difference (s)', 'Maximum crossing-time difference (s)')):
        axis.plot(index, bank, 's', ms=4, mfc='none', mec='#287c9b', label='Original finite-bank maximum')
        axis.plot(index[~harmful], actual[~harmful], '.', color='#3b6f50', ms=5,
                  label='Realized-noise local counterfactual')
        axis.plot(index[harmful], actual[harmful], 'x', color='#b63b32', ms=8, mew=1.5,
                  label='Original-metric local regression')
        axis.axhline(0, color='#555555', lw=.7)
        axis.set_ylabel(ylabel, fontsize=10)
        axis.grid(axis='y', color='#dddddd', lw=.6)
        axis.spines[['top', 'right']].set_visible(False)
        axis.set_xlim(0, 88)
    axes[1].axhline(.05, color='#996d35', lw=.8, ls='--', label='Original 50 ms phase tolerance')
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .935), ncol=2, fontsize=9)
    axes[1].set_xlabel('All 87 admitted decisions in frozen case/step order (not independent samples)')
    figure.suptitle('Finite-bank admission versus realized-noise continuation', fontsize=16, y=.97)
    figure.text(.02, .025,
        'All 480 nominal episodes / 81 adopted episodes; 87 complete local branch pairs, four local regressions.\n'
        'Each local comparison executes one original action then the feedback baseline. Known future noise is diagnostic only.\n'
        'Four regressions share only two distinct initials. This is not a new controller, risk estimate or independent evaluation.', fontsize=9)
    figure.subplots_adjust(left=.085, right=.98, top=.82, bottom=.17, hspace=.28)
    save(figure, 'all_nominal_admissions')


def mechanisms(report):
    rows = []
    for item in report['selected_regressions']:
        d = item['decomposition']
        harm = np.flatnonzero(np.asarray(d['original_cost_delta']) > 1e-9)
        if len(harm) != 1:
            raise ValueError('All selected outcomes must retain their one observed harmed vehicle')
        vehicle = int(harm[0])
        values = [d[k][vehicle] for k in ('interpolated_crossing_delta', 'decision_rounding_delta', 'variation_cost_delta')]
        np.testing.assert_allclose(sum(values), d['original_cost_delta'][vehicle], rtol=0, atol=1e-12)
        rows.append(dict(case=item['case'], vehicle_index=vehicle, components=values,
                         original_cost_delta=d['original_cost_delta'][vehicle]))
    values = np.array([r['components'] for r in rows])
    figure, axis = plt.subplots(figsize=(12.8, 5.8))
    positive, negative = np.zeros(len(rows)), np.zeros(len(rows))
    for j, (label, color) in enumerate(zip(
            ['Interpolated crossing', 'Decision-grid rounding', 'Control variation cost'],
            ['#277f9d', '#c28b37', '#54825a'])):
        current = values[:, j]
        left = np.where(current >= 0, positive, negative)
        axis.barh(np.arange(len(rows)), current, left=left, height=.5, color=color, label=label)
        for i, value in enumerate(current):
            if abs(value) < .03:
                axis.annotate(f'{value:+.4f}', xy=(left[i]+value/2, i), xytext=(0, 24),
                              textcoords='offset points', ha='center', va='bottom', fontsize=8,
                              color=color, arrowprops=dict(arrowstyle='-', color=color, lw=.6))
            else:
                axis.text(left[i]+value/2, i, f'{value:+.4f}', ha='center', va='center', fontsize=8,
                          color='white' if j != 1 else '#222222', rotation=0)
        positive += np.maximum(current, 0)
        negative += np.minimum(current, 0)
    total = np.array([r['original_cost_delta'] for r in rows])
    axis.plot(total, np.arange(len(rows)), 'D', color='#222222', ms=6, label='Original total difference')
    for i, value in enumerate(total):
        axis.text(max(positive[i], value)+.018, i, f'total +{value:.4f}', va='center', fontsize=9)
    axis.set_yticks(np.arange(len(rows)), [f"{r['case'][1]} / {r['case'][-1].split('_')[-1]}\nvehicle {r['vehicle_index']+1}" for r in rows], fontsize=10)
    axis.invert_yaxis()
    axis.axvline(0, lw=.7, color='#555555')
    axis.set_xlim(-.13, .68)
    axis.set_xlabel('Original per-vehicle cost difference, candidate minus baseline (s)')
    axis.spines[['top', 'right']].set_visible(False)
    axis.legend(loc='upper center', bbox_to_anchor=(.5, -.15), ncol=2, fontsize=9)
    figure.suptitle('All four observed in-bound regressions: original cost decomposition', fontsize=15, y=.97)
    figure.text(.02, .025,
        '12 real SUMO replays / 277 commands matched the original trajectories and the single-action counterfactuals.\n'
        'The three slow-yield rows use one initial and identical interventions. All original failures and metrics remain unchanged.\n'
        'Substep-linear crossing time explains the grid effect; it is not an exact continuous-physics ground truth or a replacement metric.', fontsize=9)
    figure.subplots_adjust(left=.19, right=.97, top=.87, bottom=.30)
    save(figure, 'regression_cost_decomposition')
    return rows


def main():
    committed(Path(__file__))
    report, counts, source = verified_report()
    OUT.mkdir(parents=True, exist_ok=False)
    admissions(report)
    decomposition = mechanisms(report)
    exclusive_json(OUT/'plotted_data.json', dict(admissions=report['all_local_diagnostics'], decomposition=decomposition))
    exclusive_json(OUT/'provenance.json', dict(utc=utc(), source_report_sha256=digest(source),
        renderer_sha256=digest(Path(__file__)), completed_inputs_reverified=True, **counts,
        new_sumo_runs=0, new_independent_initials=0, new_risk_certificate=False,
        previous_goal_turn='verified_wait', previous_turn_evidence='Four native experiment workers were observed live; the V28 completed report was inspected.'))
    exclusive_json(OUT/'sha256.json', {p.name:digest(p) for p in OUT.iterdir() if p.is_file()})
    print(json.dumps(dict(figures=2, **counts)), flush=True)


if __name__ == '__main__':
    main()
