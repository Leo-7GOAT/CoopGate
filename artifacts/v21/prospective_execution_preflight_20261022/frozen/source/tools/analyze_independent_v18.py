"""Frozen three-model study: paired initial-cluster estimates, not risk certification."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def bootstrap_indices(regimes, mask, rng, repetitions=5000):
    """Keep the prespecified regime counts; modes/models stay inside each initial."""
    local = regimes[mask]
    pieces = []
    for regime in sorted(set(local)):
        ids = np.flatnonzero(local == regime)
        pieces.append(ids[rng.integers(0, len(ids), (repetitions, len(ids)))])
    return np.concatenate(pieces, axis=1)


def analyze(study, out, audit_path):
    study, out = Path(study), Path(out)
    p = json.loads((study / 'protocol.json').read_text())
    done = json.loads((study / 'completion.json').read_text())
    audit = json.loads(Path(audit_path).read_text())
    digest = hashlib.sha256((study / 'episodes.json').read_bytes()).hexdigest()
    assert done['episodes_sha256'] == audit['episodes_sha256'] == digest
    rows = json.loads((study / 'episodes.json').read_text())
    n, modes, methods = p['n_initials'], p['modes'], p['methods']
    seeds = list(range(p['seed_base'], p['seed_base'] + n))
    lookup = {(r['seed'], r['disturbance'], r['controller']): r for r in rows}
    assert len(rows) == len(lookup) == p['expected_sumo_runs']
    assert set(lookup) == {(s, m, k) for s in seeds for m in modes for k in methods}
    arrays = {k: np.array([[np.column_stack((lookup[s, m, k]['per_vehicle_cost'],
                                           lookup[s, m, k]['per_vehicle_finish']))
                           for m in modes] for s in seeds]) for k in methods}
    regimes = np.array([lookup[s, modes[0], 'baseline']['scenario'] for s in seeds])
    groups = np.array([lookup[s, modes[0], 'baseline']['group'] for s in seeds])
    selections = {k: np.array([[lookup[s, m, k]['learned_action_steps'] > 0 for m in modes]
                              for s in seeds]).any(1) if k.startswith('gate_')
                  else np.full(n, k != 'baseline', dtype=bool) for k in methods}
    model_ids = list(p['models'])
    base = arrays['baseline']
    results = {}
    names = methods + ['raw_average', 'gate_average']
    for name in names:
        members = [name] if name in methods else [name.split('_')[0] + '_' + s for s in model_ids]
        member_array = np.stack([arrays[k] for k in members])
        member_delta = base[None] - member_array
        # Never average model losses before deciding whether an individual regressed.
        harmed = np.any(member_delta < -1e-9, axis=(2, 3, 4))
        admitted = np.stack([selections[k] for k in members])
        difference = member_delta.mean(0)
        summaries = {}
        for group, mask in [('all', np.ones(n, bool))] + [(g, groups == g) for g in sorted(set(groups))]:
            delta = difference[mask].mean(1)
            team = delta[:, :, 0].mean(1)
            ids = bootstrap_indices(regimes, mask, np.random.default_rng(18190001))
            total = int(mask.sum()) * len(members)
            selected = int(admitted[:, mask].sum())
            harm_count = int(harmed[:, mask].sum())
            selected_harm = int((harmed[:, mask] & admitted[:, mask]).sum())
            summaries[group] = dict(
                initials=int(mask.sum()), fixed_models=len(members), policy_initial_pairs=total,
                baseline_cost=float(base[mask, :, :, 0].mean()),
                method_cost=float(member_array[:, mask, :, :, 0].mean()),
                mean_team_gain=float(team.mean()),
                team_gain_percent=100 * float(team.mean() / base[mask, :, :, 0].mean()),
                paired_team_gain_ci95=np.quantile(team[ids].mean(1), [.025, .975]).tolist(),
                mean_individual_gain=delta.mean(0).tolist(),
                paired_individual_gain_ci95=np.quantile(delta[ids].mean(1), [.025, .975], axis=0).tolist(),
                admitted_policy_initial_pairs=selected,
                harmful_policy_initial_pairs=harm_count,
                harmful_admitted_policy_initial_pairs=selected_harm,
                empirical_harm_all=harm_count / total,
                empirical_harm_among_admitted=selected_harm / selected if selected else None,
                worst_cost_loss_quantiles=np.quantile(np.maximum(
                    -member_delta[:, mask, :, :, 0].min(axis=(2, 3)), 0), [.5, .9, .95, 1]).tolist())
        data = [r for r in rows if r['controller'] in members]
        results[name] = dict(
            groups=summaries, members=members, runs=len(data),
            completed=sum(r['completed'] for r in data),
            collisions=sum(r['collision'] for r in data),
            contract_violations=sum(r['contract_violations'] for r in data),
            aborted=sum(r['aborted'] for r in data),
            learned_action_steps=sum(r['learned_action_steps'] for r in data),
            control_steps=sum(r['control_steps'] for r in data),
            p95_computation_ms=float(np.percentile([v for r in data for v in r['controller_ms']], 95)))
    raw, gate = [results[k]['groups']['all'] for k in ('raw_average', 'gate_average')]
    high = results['raw_average']['groups']['high_conflict']
    nominal = audit['nominal_reconstruction']
    checks = dict(
        every_raw_seed_positive=all(results['raw_' + s]['groups']['all']['mean_team_gain'] > 0 for s in model_ids),
        raw_average_positive_ci=raw['paired_team_gain_ci95'][0] > 0,
        raw_high_conflict_positive=high['mean_team_gain'] > 0,
        gate_one_percent_positive_ci=gate['team_gain_percent'] >= 1 and gate['paired_team_gain_ci95'][0] > 0,
        gate_all_individual_means_nonnegative=bool(np.all(np.array(gate['mean_individual_gain']) >= -1e-9)),
        nominal_identities=nominal['identities'] == n * len(model_ids),
        gate_regressions_reduced_both_denominators=(gate['empirical_harm_all'] < raw['empirical_harm_all'] and
            gate['empirical_harm_among_admitted'] is not None and
            gate['empirical_harm_among_admitted'] < raw['empirical_harm_among_admitted']),
        all_actual_runs_complete_and_no_recorded_events=all(r['completed'] and not r['collision'] and
            not r['aborted'] and not r['contract_violations'] for r in rows),
        each_gate_p95_below_decision_period=all(results['gate_' + s]['p95_computation_ms'] < 500 for s in model_ids))
    record = dict(purpose=p['spec']['purpose'], actual_sumo_runs=len(rows), n_initials=n,
        interval_scope='Conditional on the three fixed trained models; modes and models averaged within initial; stratified initial-cluster bootstrap with fixed nominal/dense counts',
        risk_denominator='Policy-initial pairs are descriptive counts, not independent trials for a risk confidence bound',
        results=results, checks=checks, passed_all_mean_performance_checks=all(checks.values()),
        disturbance_performance_risk_certified=False,
        source_sha256={k: hashlib.sha256((study / k).read_bytes()).hexdigest()
                       for k in ('protocol.json', 'episodes.json', 'completion.json')},
        audit_sha256=hashlib.sha256(Path(audit_path).read_bytes()).hexdigest())
    out.mkdir(parents=True, exist_ok=False)
    (out / 'analysis.json').write_text(json.dumps(record, indent=2) + '\n')
    plot(record, out)
    print(json.dumps(dict(checks=checks, raw=raw, gate=gate)), flush=True)
    return record


def plot(record, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.3), layout='constrained')
    labels = ['Raw MARL', 'Joint action gate']
    colors = ['#cb7549', '#287b8e']
    for i, name in enumerate(('raw_average', 'gate_average')):
        data = record['results'][name]['groups']['all']
        gain = np.r_[np.array(data['mean_individual_gain'])[:, 0], data['mean_team_gain']]
        ci = np.column_stack((np.array(data['paired_individual_gain_ci95'])[:, :, 0], data['paired_team_gain_ci95']))
        axes[0].errorbar(np.arange(4) + (i - .5) * .15, gain,
                        yerr=np.maximum(0, np.vstack((gain - ci[0], ci[1] - gain))),
                        fmt='o', capsize=3, color=colors[i], label=labels[i])
        for j, group in enumerate(('low_conflict', 'high_conflict')):
            g = record['results'][name]['groups'][group]
            axes[1].bar(j + (i - .5) * .3, g['team_gain_percent'], width=.28, color=colors[i], label=labels[i] if j == 0 else None)
        axes[2].bar(i - .16, data['empirical_harm_all'], width=.3, color=colors[i])
        axes[2].bar(i + .16, data['empirical_harm_among_admitted'], width=.3, color=colors[i], alpha=.45)
    axes[0].axhline(0, color='#888', linewidth=.8)
    axes[0].set_xticks(range(4), ['Main 0', 'Ramp 1', 'Main 2', 'Team'])
    axes[0].set_ylabel('Cost improvement (s)'); axes[0].legend(fontsize=8)
    axes[1].set_xticks(range(2), ['Low conflict', 'High conflict'])
    axes[1].set_ylabel('Mean team cost improvement (%)'); axes[1].axhline(0, color='#888', linewidth=.8)
    axes[1].legend(fontsize=8)
    axes[2].set_xticks(range(2), labels); axes[2].set_ylim(0, 1.02)
    axes[2].set_ylabel('Any individual cost/time regression')
    axes[2].set_title('Solid: all policy-initial pairs\nFaded: admitted pairs', fontsize=9)
    for ax in axes: ax.grid(axis='y', alpha=.18)
    fig.suptitle('Frozen three-model SUMO comparison\nPaired initial-cluster 95% intervals; regression is not a collision', fontsize=11)
    fig.savefig(out / 'independent_gates.png', dpi=190)
    fig.savefig(out / 'independent_gates.pdf'); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('study'); parser.add_argument('--out', required=True); parser.add_argument('--audit', required=True)
    a = parser.parse_args(); analyze(a.study, a.out, a.audit)
