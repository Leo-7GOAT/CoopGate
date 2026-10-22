"""Prespecified ego-specific statistics; no selection of trained seeds."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/v13/ego'


def mean(rows):
    keys=('ego_time','ego_variation','ego_cost','other_time','mean_capped_finish_s','learned_tail_fraction')
    return {**{k:float(np.mean([r[k] for r in rows])) for k in keys},'n':len(rows),
        'completed':sum(r['completed'] for r in rows),'collisions':sum(r['collision'] for r in rows),
        'violations':sum(r['contract_violations'] for r in rows),'outside_steps':sum(r['outside_steps'] for r in rows),
        'uncertified_steps':sum(r['uncertified_steps'] for r in rows),'aborted':sum(r['aborted'] for r in rows),
        'controller_p95_ms':float(np.percentile([v for r in rows for v in r['controller_ms']],95)),
        'setup_p95_ms':float(np.percentile([r['setup_ms'] for r in rows],95))}


def summarize(protocol,rows,calibration):
    seeds=protocol['training_seeds'];methods={m:mean([r for r in rows if r['controller']==m]) for m in protocol['methods']}
    averages={mode:mean([r for r in rows if r['controller'].startswith(mode+'_')]) for mode in ('raw','untrained','augmented','gate')}
    lookup={(r['seed'],r['disturbance'],r['controller']):r for r in rows};base=methods['rollout']
    assert len(lookup)==len(rows)==protocol['expected_runs']
    for item in protocol['initials']:
        for mode in protocol['disturbances']:
            for method in protocol['methods']:assert (item['seed'],mode,method) in lookup
    paired=[]
    for item in protocol['initials']:
        for mode in protocol['disturbances']:
            b=lookup[item['seed'],mode,'rollout']
            for s in seeds:
                a=lookup[item['seed'],mode,f'augmented_{s}'];g=lookup[item['seed'],mode,f'gate_{s}']
                delta=np.array([a['ego_cost']-b['ego_cost'],a['ego_time']-b['ego_time']]);pred=np.array(g['predicted'])
                accepted=g['accepted'];expected=a if accepted else b
                for key in ('ego_cost','ego_time','ego_variation','other_time'):
                    assert np.isclose(g[key],expected[key],atol=1e-8),(item,mode,s,key)
                assert not g['fault_fallback']
                paired.append({'seed':item['seed'],'policy':s,'group':g['group'],'delta_cost':float(delta[0]),
                    'delta_time':float(delta[1]),'predicted':pred.tolist(),'accepted':accepted,
                    'covered':bool(np.all(delta<=np.array(g['upper'])+1e-9)),
                    'harm':bool(np.any(delta>1e-9)),'baseline_cost':b['ego_cost']})
    diagnostics={};ablations=[]
    for group in ('all','low_conflict','high_conflict'):
        pp=[r for r in paired if group=='all' or r['group']==group];n=len(pp);accepted=sum(r['accepted'] for r in pp)
        harm=sum(r['accepted'] and r['harm'] for r in pp);gain=-sum(r['delta_cost'] for r in pp)
        retained=-sum(r['delta_cost'] for r in pp if r['accepted'])
        diagnostics[group]={'n':n,'accepted':accepted,'acceptance_rate':accepted/n,'harmful_admissions':harm,
            'joint_harm_rate':harm/n,'conditional_harm_rate':harm/accepted if accepted else None,
            'always_augmented_harm_rate':sum(r['harm'] for r in pp)/n,
            'empirical_joint_coverage':sum(r['covered'] for r in pp)/n,
            'mean_augmented_gain':gain/n,'mean_gated_gain':retained/n,
            'retained_gain_percent':100*retained/gain if gain>0 else None}
        for variant in ('uncalibrated_cost','calibrated_cost_only','joint_cost_time'):
            admitted=[]
            for r in pp:
                q=calibration[str(r['policy'])]['groups'][r['group']]['q_cost_only']
                if variant=='joint_cost_time':take=r['accepted']
                elif variant=='calibrated_cost_only':take=r['predicted'][0]+q<=-.05
                else:take=r['predicted'][0]<=-.05
                admitted.append(bool(take))
            ac=sum(admitted);harm=sum(take and r['harm'] for take,r in zip(admitted,pp))
            ablations.append({'variant':variant,'group':group,'n':n,'accepted':ac,'acceptance_rate':ac/n,
                'joint_harmful_admissions':harm,'joint_harm_rate':harm/n,
                'mean_retained_cost_gain':-sum(r['delta_cost'] for take,r in zip(admitted,pp) if take)/n,
                'mean_retained_time_gain':-sum(r['delta_time'] for take,r in zip(admitted,pp) if take)/n})
    contrasts={};rng=np.random.default_rng(protocol['bootstrap']['seed']);n=len(protocol['initials'])
    ids=rng.integers(0,n,(protocol['bootstrap']['draws'],n))
    for controller in ('raw','augmented','gate'):
        contrasts[controller]={}
        for key in ('ego_time','ego_cost','ego_variation','other_time'):
            clusters=[]
            for item in protocol['initials']:
                differences=[]
                for mode in protocol['disturbances']:
                    b=lookup[item['seed'],mode,'rollout']
                    differences.extend(b[key]-lookup[item['seed'],mode,f'{controller}_{s}'][key] for s in seeds)
                clusters.append(np.mean(differences))
            values=np.array(clusters);ci=np.quantile(values[ids].mean(-1),[.025,.975])
            contrasts[controller][key]={'mean_improvement':float(values.mean()),'ci95':ci.tolist()}
    performance={}
    for mode in averages:
        performance[mode]={key+'_gain_percent':100*(base[key]-averages[mode][key])/base[key] for key in ('ego_time','ego_variation','ego_cost','other_time')}
    extra_gain=100*(averages['untrained']['ego_cost']-averages['augmented']['ego_cost'])/averages['untrained']['ego_cost']
    alltarget=[r for r in rows if r['controller'].startswith(('augmented_','gate_'))]
    c=protocol['criteria'];ga=diagnostics['all'];pa=performance['augmented'];pg=performance['gate']
    checks={'augmented_cost_gain':pa['ego_cost_gain_percent']>=c['augmented_cost_gain_percent_min'],
        'augmented_variation_gain':pa['ego_variation_gain_percent']>=c['augmented_variation_gain_percent_min'],
        'augmented_time_gain':pa['ego_time_gain_percent']>=c['augmented_time_gain_percent_min'],
        'every_trained_seed_improves':all(methods[f'augmented_{s}']['ego_cost']<base['ego_cost'] for s in seeds),
        'learning_gain_vs_untrained':extra_gain>=c['augmented_gain_vs_untrained_percent_min'],
        'gated_cost_gain':pg['ego_cost_gain_percent']>=c['gated_cost_gain_percent_min'],
        'gate_retains_gain':ga['retained_gain_percent'] is not None and ga['retained_gain_percent']>=c['gate_retains_cost_gain_percent_min'],
        'gated_arrival_not_worse':pg['ego_time_gain_percent']>=-c['gated_time_regression_percent_max'],
        'harmful_admission':ga['joint_harm_rate']<=c['joint_harmful_admission_rate_max'],
        'completed_and_no_safety_events':all(r['completed'] and not r['collision'] and not r['contract_violations']
            and not r['aborted'] and not r['outside_steps'] and not r['uncertified_steps'] for r in alltarget),
        'real_admissions_and_fallbacks':0<ga['accepted']<ga['n']}
    individual=(performance['raw']['ego_cost_gain_percent']>0 and performance['raw']['ego_time_gain_percent']>=0
        and contrasts['raw']['ego_cost']['ci95'][0]>0)
    return {'executions':len(rows),'independent_initials':n,'methods':methods,'seed_average':averages,'performance':performance,
        'learning_gain_vs_untrained_percent':extra_gain,'gate_diagnostics':diagnostics,'counterfactual_gate_ablations':ablations,
        'paired_initial_cluster_intervals':contrasts,'acceptance':{'checks':checks,'passed':sum(checks.values()),
            'total':len(checks),'all_passed':all(checks.values())},'individual_ego_verified':individual,
        'marl_expansion_allowed':individual,'interpretation':'learning-augmented controller is separate from standalone neural policy'}


if __name__=='__main__':
    p=json.loads((D/'protocol.json').read_text());rows=json.loads((D/'study/episodes.json').read_text())
    s=summarize(p,rows,json.loads((D/'calibration/calibration.json').read_text()))
    (D/'summary.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s,indent=2))
