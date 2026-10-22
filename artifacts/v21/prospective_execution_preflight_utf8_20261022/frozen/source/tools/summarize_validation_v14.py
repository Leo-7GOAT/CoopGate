"""Paired initial-cluster summaries of development validation; no acceptance flag."""
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np


def summarize(folders,out):
    pairs=defaultdict(list);labels={};total_runs=0
    for folder in map(Path,folders):
        protocol=json.loads((folder/'protocol.json').read_text())
        rows=json.loads((folder/'episodes.json').read_text());total_runs+=len(rows)
        baseline={(r['seed'],r['scenario'],r['disturbance']):r for r in rows if r['controller']=='rollout'}
        for row in rows:
            if row['controller']=='rollout':continue
            path=protocol['methods'][row['controller']];digest=protocol['policy_sha256'][path];labels[digest]=path
            key=(row['seed'],row['scenario'],row['disturbance'])
            pairs[digest].append((key,baseline[key],row))
    results={}
    for digest,items in pairs.items():
        keys=[row[0] for row in items]
        if len(set(keys))!=len(keys):raise ValueError('Repeated policy/scenario pairs cannot count twice')
        seeds=sorted(set((k[0],k[1]) for k in keys));rng=np.random.default_rng(941)
        sample=rng.integers(0,len(seeds),(10000,len(seeds)))
        metrics={}
        for metric in ('ego_time','ego_variation','ego_cost','other_time'):
            diffs=np.array([np.mean([base[metric]-row[metric] for key,base,row in items if key[:2]==seed]) for seed in seeds])
            b=float(np.mean([base[metric] for _,base,_ in items]));r=float(np.mean([row[metric] for _,_,row in items]))
            metrics[metric]=dict(baseline=b,policy=r,reduction_percent=100*(b-r)/b,
                                 mean_improvement=float(diffs.mean()),ci95=np.quantile(diffs[sample].mean(-1),[.025,.975]).tolist())
        results[digest]=dict(path=labels[digest],independent_initials=len(seeds),policy_runs=len(items),metrics=metrics,
                             completed=sum(row['completed'] for _,_,row in items),
                             collisions=sum(row['collision'] for _,_,row in items),aborts=sum(row['aborted'] for _,_,row in items))
    summary=dict(purpose='development validation only; no formal prerequisite verdict',sumo_runs=total_runs,results=results)
    Path(out).write_text(json.dumps(summary,indent=2)+'\n')
    for value in results.values():print(json.dumps(value),flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folders',nargs='+');p.add_argument('--out',required=True);a=p.parse_args()
    summarize(a.folders,a.out)
