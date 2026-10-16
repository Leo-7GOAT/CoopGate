"""Complete paired development analysis, with the initial state as sampling unit.

All methods and failed episodes stay in the comparison. Intervals describe the
fixed trained models, not the distribution of future independent training runs.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np


def analyze(directory, out):
    directory, out = Path(directory), Path(out)
    protocol = json.loads((directory/'protocol.json').read_text())
    rows = json.loads((directory/'episodes.json').read_text())
    methods, modes = protocol['methods'], protocol['modes']
    n = protocol['n_initials']; seeds = list(range(protocol['seed_base'], protocol['seed_base']+n))
    lookup = {(r['seed'], r['disturbance'], r['controller']): r for r in rows}
    expected = {(s,m,p) for s in seeds for m in modes for p in methods}
    if len(rows) != protocol['expected_runs'] or len(lookup) != len(rows) or set(lookup) != expected:
        raise ValueError('Missing, duplicate, or unexpected episodes')
    group = []
    for seed in seeds:
        paired = [lookup[seed,m,p] for m in modes for p in methods]
        if any(r['initial'] != paired[0]['initial'] or r['group'] != paired[0]['group'] for r in paired):
            raise ValueError('Unpaired initials or conflict groups')
        group.append(paired[0]['group'])
    group = np.asarray(group)
    arrays = {}
    safety = {}
    for method in methods:
        costs = np.array([[lookup[s,m,method]['per_vehicle_cost'] for m in modes] for s in seeds])
        arrival = np.array([[lookup[s,m,method]['per_vehicle_finish'] for m in modes] for s in seeds])
        variation = np.array([[lookup[s,m,method]['per_vehicle_variation'] for m in modes] for s in seeds])
        np.testing.assert_allclose(costs, arrival+.05*variation, atol=1e-12, rtol=0)
        if not all(np.isfinite(a).all() for a in (costs, arrival, variation)):
            raise ValueError('Nonfinite episode outcome')
        arrays[method] = dict(cost=costs, time=arrival, variation=variation)
        data = [r for r in rows if r['controller'] == method]
        safety[method] = {key: sum(r[key] for r in data) for key in
            ('completed','collision','aborted','contract_violations','outside_steps','uncertified_steps')}
        safety[method]['runs'] = len(data)
    result = dict(purpose='Development, not a formal acceptance verdict',
        sampling_unit='initial state; four disturbance modes averaged inside each initial',
        uncertainty='Descriptive paired bootstrap conditional on the fixed trained models',
        n_initials=n, safety=safety, comparisons={},
        source_sha256={f:hashlib.sha256((directory/f).read_bytes()).hexdigest() for f in ('protocol.json','episodes.json')})
    for baseline in [m for m in methods if m in ('ego_planner','team_planner') or m.startswith('single_')]:
        result['comparisons'][baseline] = {}
        for method in [m for m in methods if m != baseline]:
            comparison = {}
            for name, selected in [('all',np.ones(n,bool))]+[(g,group==g) for g in sorted(set(group))]:
                count = int(selected.sum()); record = dict(n_initials=count, metrics={})
                rng = np.random.default_rng(15159001)
                indices = rng.integers(0,count,(5000,count))
                for metric in ('cost','time','variation'):
                    base = arrays[baseline][metric][selected]
                    learned = arrays[method][metric][selected]
                    delta = (base-learned).mean(1)
                    team = delta.mean(1)
                    record['metrics'][metric] = dict(
                        baseline_per_vehicle=base.mean((0,1)).tolist(),
                        method_per_vehicle=learned.mean((0,1)).tolist(),
                        gain_per_vehicle=delta.mean(0).tolist(),
                        ci95_per_vehicle=np.quantile(delta[indices].mean(1),[.025,.975],axis=0).T.tolist(),
                        gain_team=float(team.mean()),
                        gain_team_percent=100*float(team.mean()/base.mean()),
                        ci95_team=np.quantile(team[indices].mean(1),[.025,.975]).tolist(),
                        worst_vehicle_or_mode_harmed_initials=int(np.any(learned>base+1e-9,axis=(1,2)).sum()),
                        team_win_initial_fraction=float(np.mean(team>1e-9)),
                        team_loss_initial_fraction=float(np.mean(team < -1e-9)))
                comparison[name] = record
            result['comparisons'][baseline][method] = comparison
    out.mkdir(parents=True,exist_ok=False)
    (out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    learners = [m for m in methods if m.startswith(('mappo_','ippo_'))]
    fig, axes = plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
    for ax, baseline in zip(axes, ('ego_planner','team_planner')):
        for k, method in enumerate(learners):
            r=result['comparisons'][baseline][method]['all']['metrics']['cost']
            values=r['gain_per_vehicle']+[r['gain_team']]
            bounds=np.array(r['ci95_per_vehicle']+[r['ci95_team']])
            ax.errorbar(np.arange(4)+(k-(len(learners)-1)/2)*.16,values,
                yerr=np.maximum(0,np.array([np.array(values)-bounds[:,0],bounds[:,1]-np.array(values)])),
                fmt='o',capsize=3,label=method)
        ax.axhline(0,color='#777',linestyle='--',linewidth=.8)
        ax.set_xticks(range(4),['Mainroad 0','Ramp 1','Mainroad 2','Team'])
        ax.set_ylabel('Baseline minus learned cost (s)')
        ax.set_title('Compared with '+baseline); ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    fig.suptitle(f'Development: {n} paired initials, four disturbance modes\nPositive favors learning; paired initial-cluster 95% intervals',fontsize=11)
    fig.savefig(out/'cooperative.png',dpi=180);fig.savefig(out/'cooperative.pdf');plt.close(fig)
    concise = {b:{m:{g:r['metrics']['cost'] for g,r in groups.items()} for m,groups in ms.items() if m in learners}
        for b,ms in result['comparisons'].items()}
    print(json.dumps(dict(safety=safety, comparisons=concise)),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');parser.add_argument('--out',required=True)
    args=parser.parse_args();analyze(args.directory,args.out)
