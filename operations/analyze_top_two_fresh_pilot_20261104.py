"""Full paired description only after the fresh pilot's actual native exit0."""
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import numpy as np
from coopgate.episode_store_v21 import observations
import experiments.top_two_candidates.fresh_pilot as p


def run():
    e=p.e;owner=e.read(p.OUT/'owner_native_exit.json')
    if owner['actual_native_exit_code']!=0 or owner['report_sha256']!=e.sha(p.OUT/'report.json'):
        raise ValueError('Actual successful completed pilot owner required')
    plan=p.guard();report=e.read(p.OUT/'report.json')
    if not report['completed'] or report['actual_sumo_executions']!=2240:
        raise ValueError('Original whole2240-run scope required')
    out=e.BASE/'development/local_top_two_fresh_analysis_20261104';out.mkdir(exist_ok=False)
    records_seen=0;paired=[];totals={};groups={}
    for shard in range(4):
        for block in e.read(p.OUT/f'shard{shard}/report.json')['blocks']:
            db=p.OUT/f'shard{shard}'/block['block']/'observations.sqlite'
            if e.sha(db)!=block['database_sha256']:raise ValueError('Reviewed observations changed')
            records={}
            for _,record,trace in observations(db,block['metadata'],block['keys']):
                records[record['variant'],record['method']]=record;records_seen+=1
            if len(records)!=7:raise ValueError('Complete seven-method pairing required')
            case=block['metadata']['case'];base=p.episode_outcome(records['baseline','baseline']['row'])
            for seed in ('4703','4704','4705'):
                outcomes={}
                for variant in ('history','top_history'):
                    value=p.episode_outcome(records[variant,seed]['row'])
                    harm=bool(value['failed'] or base['failed'] or np.any(value['costs']>base['costs']+1e-9)
                              or np.any(value['arrivals']>base['arrivals']+1e-9))
                    gain=100*(base['team_cost']-value['team_cost'])/base['team_cost']
                    outcomes[variant]=dict(adopted=value['adopted'],harm=harm,failed=value['failed'],gain=gain)
                row=dict(case=case,seed=seed,**outcomes);paired.append(row)
                for group in ('all','plant:'+case['plant'],'scene:'+case['scenario'],'mode:'+case['disturbance']):
                    q=groups.setdefault(group,{}).setdefault(seed,dict(n=0,old_adopt=0,new_adopt=0,
                        old_harm=0,new_harm=0,old_harmful_adopted=0,new_harmful_adopted=0,
                        harm_removed=0,harm_introduced=0,old_gain_sum=0.,new_gain_sum=0.))
                    old=outcomes['history'];new=outcomes['top_history'];q['n']+=1
                    q['old_adopt']+=int(old['adopted']);q['new_adopt']+=int(new['adopted'])
                    q['old_harm']+=int(old['harm']);q['new_harm']+=int(new['harm'])
                    q['old_harmful_adopted']+=int(old['harm'] and old['adopted'])
                    q['new_harmful_adopted']+=int(new['harm'] and new['adopted'])
                    q['harm_removed']+=int(old['harm'] and not new['harm'])
                    q['harm_introduced']+=int(not old['harm'] and new['harm'])
                    q['old_gain_sum']+=old['gain'];q['new_gain_sum']+=new['gain']
    if records_seen!=2240 or len(paired)!=960:raise ValueError('Missing actual observations')
    for group,models in groups.items():
        for seed,q in models.items():
            q['old_mean_gain_percent']=q.pop('old_gain_sum')/q['n']
            q['new_mean_gain_percent']=q.pop('new_gain_sum')/q['n']
            q['gain_change_pp']=q['new_mean_gain_percent']-q['old_mean_gain_percent']
            if group=='all':
                for variant,kind in [('history','old'),('top_history','new')]:
                    original=report['results']['fresh:'+variant+':'+seed]
                    if (original['initials']!=q['n'] or original['adopted']!=q[kind+'_adopt']
                        or original['harmful_adopted']!=q[kind+'_harmful_adopted']
                        or not math.isclose(original['mean_paired_percentage_gain'],q[kind+'_mean_gain_percent'],abs_tol=1e-12,rel_tol=0)):
                        raise ValueError('Independent full statistics differ from original report')
    result=dict(completed=True,source_report_sha256=e.sha(p.OUT/'report.json'),
        actual_sumo=2240,fresh_development_initials=320,groups=groups,
        all_root_statistics_independently_match=True,new_statistical_inference=False,
        new_error_allocated=False,old_development_screen_still_failed=True,
        subgroup_descriptions_are_not_subgroup_certificates=True)
    e.write(out/'paired_outcomes.json',paired);e.write(out/'report.json',result)
    lines=['# Complete fresh balanced top two pilot','',
        '320new development initials,2240actual SUMO episodes,all three fixed actor seeds.',
        'All outcomes remain. This is development evidence, not a formal certificate.',
        'The earlier reused-data promotion screen remains failed.','',
        '| Model | Adoption old/new | Harm old/new | Gain old | Gain new | Difference pp | Harm removed/introduced |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for seed,q in groups['all'].items():
        lines.append(f"| {seed} | {q['old_adopt']}/{q['new_adopt']} | {q['old_harm']}/{q['new_harm']} | {q['old_mean_gain_percent']:.6f}% | {q['new_mean_gain_percent']:.6f}% | {q['gain_change_pp']:.6f} | {q['harm_removed']}/{q['harm_introduced']} |")
    lines+=['','All original root statistics were independently reconstructed from complete sealed',
        'blocks. Full plant/scene/disturbance descriptions are retained in report.json;',
        'they are not separate risk certificates. Callback timing is in the original',
        'report and includes every actual method. No communication-delay metric or',
        'hard-real-time guarantee. No formal6or statistical error was allocated.']
    with (out/'results.md').open('x',encoding='utf-8',newline='\n') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(completed=True,all=groups['all'])),flush=True)


if __name__=='__main__':run()
