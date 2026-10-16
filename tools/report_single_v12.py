"""Account for every SUMO run; retain failed acceptance criteria verbatim."""
import argparse,json
from pathlib import Path
import numpy as np


def summarize(protocol,rows):
    if len(rows)!=protocol['expected_runs']:raise ValueError('Incomplete study')
    methods=protocol['methods'];seeds=protocol['training_seeds'];initials=[v['seed'] for v in protocol['initials']]
    keys=[(r['seed'],r['disturbance'],r['controller']) for r in rows]
    if len(set(keys))!=len(keys):raise ValueError('Duplicate execution')
    lookup={key:r for key,r in zip(keys,rows)};modes=protocol['disturbances']
    for initial in initials:
        for mode in modes:
            for method in methods:
                if (initial,mode,method) not in lookup:raise ValueError('Missing matched execution')
    def value(r,key):
        return r['mean_capped_finish_s']+.05*r['variation'] if key=='objective' else r[key]
    def means(selected):
        return {key:float(np.mean([value(r,key) for r in selected])) for key in ('mean_capped_finish_s','variation','ego_finish_s','ego_variation','objective','joint_acceptance')}
    aggregates={}
    for method in methods:
        rr=[r for r in rows if r['controller']==method]
        timing=[v for r in rr for v in r['controller_ms']]
        aggregates[method]={**means(rr),'n':len(rr),'completed':sum(r['completed'] for r in rr),
            'collisions':sum(r['collision'] for r in rr),'violations':sum(r['contract_violations']>0 for r in rr),
            'aborted':sum(r['aborted'] for r in rr),'controller_p95_ms':float(np.quantile(timing,.95)),
            'setup_p95_ms':float(np.quantile([r['setup_ms'] for r in rr],.95))}
    average={prefix:means([r for r in rows if r['controller'].startswith(prefix+'_')]) for prefix in ('bc','rl','gate')}
    rng=np.random.default_rng(protocol['bootstrap']['seed'])
    indices=rng.integers(0,len(initials),(protocol['bootstrap']['draws'],len(initials)))
    contrasts={}
    for prefix,reference in (('rl','rollout'),('gate','rollout'),('rl','rule'),('rl','bc')):
        for key in ('objective','mean_capped_finish_s','variation'):
            deltas=[]
            for initial in initials:
                differences=[]
                for mode in modes:
                    for seed in seeds:
                        ref=f'bc_{seed}' if reference=='bc' else reference
                        differences.append(value(lookup[(initial,mode,ref)],key)-value(lookup[(initial,mode,f'{prefix}_{seed}')],key))
                deltas.append(float(np.mean(differences)))
            d=np.array(deltas);ci=np.quantile(d[indices].mean(-1),[.025,.975])
            contrasts[f'{prefix}_vs_{reference}_{key}']={'gain':float(d.mean()),'ci95':ci.tolist(),
                'better_clusters':int(np.sum(d>1e-9)),'tied_clusters':int(np.sum(np.abs(d)<=1e-9)),'worse_clusters':int(np.sum(d< -1e-9))}
    gate_rows=[r for r in rows if r['controller'].startswith('gate_')]
    diagnostics={}
    for group in ('all','low_conflict','high_conflict'):
        selected=[r for r in gate_rows if group=='all' or r['group']==group]
        accepted=[];harmful=[];always_harm=[];missed=[];coverage=[];learning_gains=[];gate_gains=[]
        for r in selected:
            seed=int(r['controller'].split('_')[1]);base=lookup[(r['seed'],r['disturbance'],'rollout')]
            rl=lookup[(r['seed'],r['disturbance'],f'rl_{seed}')]
            chosen=rl if r['accepted'] else base
            for key in ('mean_capped_finish_s','variation','ego_finish_s'):
                if abs(value(r,key)-value(chosen,key))>1e-8:raise ValueError('Gated execution does not match selected controller')
            delta=value(rl,'objective')-value(base,'objective');gd=value(r,'objective')-value(base,'objective')
            accepted.append(r['accepted']);harmful.append(r['accepted'] and delta>1e-9);always_harm.append(delta>1e-9)
            missed.append(not r['accepted'] and delta< -1e-9)
            coverage.append(delta<=r['upper_delta_cost']+1e-9)
            learning_gains.append(-delta);gate_gains.append(-gd)
        n=len(selected);a=sum(accepted)
        diagnostics[group]={'n':n,'accepted':a,'acceptance_rate':a/n,'harmful_acceptances':sum(harmful),
            'harmful_acceptance_rate':sum(harmful)/n,'conditional_harm_rate':sum(harmful)/a if a else None,
            'always_learned_harmful_rate':sum(always_harm)/n,'missed_beneficial_rate':sum(missed)/n,
            'empirical_upper_coverage':float(np.mean(coverage)),
            'mean_learning_cost_gain':float(np.mean(learning_gains)),'mean_gate_cost_gain':float(np.mean(gate_gains)),
            'retained_net_gain_percent':100*float(np.mean(gate_gains))/float(np.mean(learning_gains)) if np.mean(learning_gains)>0 else None}
    base=aggregates['rollout'];rl=average['rl'];gate=average['gate'];diag=diagnostics['all']
    gain=lambda b,v:100*(b-v)/b
    performance={'rl_cost_gain_percent':gain(base['objective'],rl['objective']),
        'rl_variation_gain_percent':gain(base['variation'],rl['variation']),
        'rl_arrival_regression_percent':100*(rl['mean_capped_finish_s']-base['mean_capped_finish_s'])/base['mean_capped_finish_s'],
        'gate_cost_gain_percent':gain(base['objective'],gate['objective']),
        'gate_retained_gain_percent':diag['retained_net_gain_percent']}
    c=protocol['criteria'];target=[r for r in rows if r['controller'].startswith(('rl_','gate_'))]
    checks={
        'learned_cost_gain_vs_rollout':performance['rl_cost_gain_percent']>=c['learned_cost_gain_vs_rollout_percent_min'],
        'learned_variation_gain_vs_rollout':performance['rl_variation_gain_percent']>=c['learned_variation_gain_vs_rollout_percent_min'],
        'learned_arrival_noninferiority':performance['rl_arrival_regression_percent']<=c['learned_arrival_regression_vs_rollout_percent_max'],
        'every_training_seed_improves_cost':all(aggregates[f'rl_{s}']['objective']<base['objective'] for s in seeds),
        'all_rl_and_gate_completed':all(r['completed'] for r in target),
        'no_rl_or_gate_safety_events':all(not r['collision'] and not r['contract_violations'] and not r['aborted'] and not r['uncertified_steps'] for r in target),
        'gated_cost_gain':performance['gate_cost_gain_percent']>=c['gated_cost_gain_vs_rollout_percent_min'],
        'gate_retains_learning_gain':diag['retained_net_gain_percent'] is not None and diag['retained_net_gain_percent']>=c['gate_retained_cost_gain_percent_min'],
        'harmful_acceptance_rate':diag['harmful_acceptance_rate']<=c['harmful_acceptance_rate_max'],
        'real_admissions_and_fallbacks':0<diag['accepted']<diag['n']}
    return {'executions':len(rows),'independent_initials':len(initials),'methods':aggregates,'seed_average':average,
        'contrasts':contrasts,'gate_diagnostics':diagnostics,'performance':performance,
        'acceptance':{'passed':sum(checks.values()),'total':len(checks),'checks':checks,'all_passed':all(checks.values())}}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',default='artifacts/v12/single/protocol.json')
    p.add_argument('--study',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    result=summarize(json.loads(Path(a.protocol).read_text()),json.loads((Path(a.study)/'episodes.json').read_text()))
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'performance':result['performance'],'acceptance':result['acceptance'],'gate':result['gate_diagnostics']}),flush=True)
