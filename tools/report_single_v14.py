"""Single-ego prerequisite verdict from the frozen complete test, never a subset."""
import argparse,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'artifacts/v14/single'
METRICS=('ego_cost','ego_time','ego_variation','other_time')


def mean(rows):
    return {**{key:float(np.mean([r[key] for r in rows])) for key in METRICS},
            'n':len(rows),'completed':sum(r['completed'] for r in rows),
            'collisions':sum(r['collision'] for r in rows),'aborts':sum(r['aborted'] for r in rows),
            'contract_violations':sum(r['contract_violations'] for r in rows),
            'outside_steps':sum(r['outside_steps'] for r in rows),'uncertified_steps':sum(r['uncertified_steps'] for r in rows),
            'controller_p95_ms':float(np.percentile([v for r in rows for v in r['controller_ms']],95))}


def summarize(p,rows):
    lookup={(r['seed'],r['disturbance'],r['controller']):r for r in rows}
    assert len(lookup)==len(rows)==p['expected_runs'],'Incomplete or repeated execution accounting'
    for item in p['initials']:
        for mode in p['disturbances']:
            for method in p['methods']:assert (item['seed'],mode,method) in lookup
    methods={m:mean([r for r in rows if r['controller']==m]) for m in p['methods']}
    categories={name:mean([r for r in rows if r['controller'].startswith(name+'_')]) for name in ('learned','last')}
    base=methods['rollout']
    gains={name:{key:100*(base[key]-value[key])/base[key] for key in METRICS} for name,value in categories.items()}
    rng=np.random.default_rng(p['bootstrap']['seed']);n=len(p['initials'])
    ids=rng.integers(0,n,(p['bootstrap']['draws'],n));contrasts={}
    for name in ('learned','last'):
        contrasts[name]={}
        for key in METRICS:
            clusters=[]
            for item in p['initials']:
                diffs=[lookup[item['seed'],mode,'rollout'][key]-lookup[item['seed'],mode,f'{name}_{seed}'][key]
                       for mode in p['disturbances'] for seed in p['training_seeds']]
                clusters.append(np.mean(diffs))
            values=np.array(clusters)
            contrasts[name][key]=dict(mean_improvement=float(values.mean()),
                                      ci95=np.quantile(values[ids].mean(-1),[.025,.975]).tolist())
    group_results={}
    for group in ('low_conflict','high_conflict'):
        br=[r for r in rows if r['controller']=='rollout' and r['group']==group]
        lr=[r for r in rows if r['controller'].startswith('learned_') and r['group']==group]
        if not br:
            group_results[group]=dict(initials=0,baseline=None,learned=None)
            continue
        group_results[group]=dict(initials=len(br)//len(p['disturbances']),baseline=mean(br),learned=mean(lr))
    c=p['criteria'];g=gains['learned'];intervals=contrasts['learned']
    checks=dict(ego_cost_gain=g['ego_cost']>=c['ego_cost_gain_percent_min'],
                ego_time_gain=g['ego_time']>=c['ego_time_gain_percent_min'],
                ego_variation_gain=g['ego_variation']>=c['ego_variation_gain_percent_min'],
                every_seed_cost_improves=all(methods[f'learned_{s}']['ego_cost']<base['ego_cost'] for s in p['training_seeds']),
                every_seed_time_improves=all(methods[f'learned_{s}']['ego_time']<base['ego_time'] for s in p['training_seeds']),
                cost_interval_positive=intervals['ego_cost']['ci95'][0]>0.,
                time_interval_positive=intervals['ego_time']['ci95'][0]>0.,
                all_complete_no_safety_events=all(r['completed'] and not any(r[k] for k in
                    ('collision','aborted','contract_violations','outside_steps','uncertified_steps')) for r in rows),
                zero_online_performance_rollout=all(r['learned_tail_fraction']==0. for r in rows if r['controller'].startswith('learned_')))
    passed=all(checks.values())
    return dict(executions=len(rows),independent_initials=n,methods=methods,seed_average=categories,
                gain_percent=gains,paired_initial_cluster_intervals=contrasts,conflict_groups=group_results,
                acceptance=dict(checks=checks,passed=sum(checks.values()),total=len(checks),all_passed=passed),
                individual_ego_verified=passed,marl_expansion_allowed=passed,
                interpretation='Standalone single-ego networks; source/weight/certificate/replay verification is additionally required before MARL')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',default=str(D/'study'));p.add_argument('--out',default=str(D/'summary.json'))
    a=p.parse_args();protocol=json.loads((D/'protocol.json').read_text());rows=json.loads((Path(a.study)/'episodes.json').read_text())
    result=summarize(protocol,rows);Path(a.out).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
