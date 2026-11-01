"""Single-seed development metrics using original cost/time/variation definitions."""
import argparse
import json
from pathlib import Path

import numpy as np


def assess(directory,method):
    directory=Path(directory)
    report=json.loads((directory/'report.json').read_text())
    rows=[json.loads(s) for s in (directory/'episodes.jsonl').read_text().splitlines()]
    own=[r for r in rows if r['method']==method]
    references={(r['seed'],r['mode']):r for r in rows if r['method']=='ego_planner'}
    if len(own)!=report['protocol']['n_independent_initials']*4 or any(not r['completed'] for r in own):
        raise ValueError('Complete four-mode single-policy comparison required')
    fields=('per_vehicle_cost','per_vehicle_finish','per_vehicle_variation')
    metrics={};rng=np.random.default_rng(202611014601)
    n=report['protocol']['n_independent_initials']
    for name,field in zip(('cost','time','variation'),fields):
        base=np.array([references[r['seed'],r['mode']][field][1] for r in own])
        actual=np.array([r[field][1] for r in own])
        cluster=(base-actual).reshape(n,4).mean(1)
        draws=np.array([cluster[rng.integers(n,size=n)].mean() for _ in range(10000)])
        metrics[name]=dict(baseline_mean=float(base.mean()),policy_mean=float(actual.mean()),
            aggregate_mean_gain_percent=float(100*(base.mean()-actual.mean())/base.mean()),
            mean_paired_gain=float(cluster.mean()),development_cluster_bootstrap_ci95=np.quantile(draws,[.025,.975]).tolist())
    comparisons=report['results'][method]['comparisons']
    result=dict(method=method,backend=report['protocol']['backend'],n_independent_initials=n,
        repeated_mode_pairs=len(own),metrics=metrics,
        original_numeric_thresholds=dict(cost_gain_percent_min=1.,time_gain_percent_min=.5,variation_gain_percent_min=10.),
        numeric_thresholds_met=bool(metrics['cost']['aggregate_mean_gain_percent']>=1.
            and metrics['time']['aggregate_mean_gain_percent']>=.5
            and metrics['variation']['aggregate_mean_gain_percent']>=10.
            and metrics['cost']['development_cluster_bootstrap_ci95'][0]>0
            and metrics['time']['development_cluster_bootstrap_ci95'][0]>0),
        new_training_seeds_in_this_comparison=1,
        full_original_three_seed_acceptance_claim=False,
        ancestry='PI warm start; this does not establish from-scratch PPO sample efficiency',
        initializer_contrast=comparisons.get('legacy_initial'),
        interpretation='Single-seed development competence only, not a new risk certificate or three-seed formal acceptance')
    (directory/'single_competence.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');parser.add_argument('--method',required=True)
    args=parser.parse_args();assess(args.directory,args.method)
