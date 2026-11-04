"""Summarize only terminal-complete bounded-candidate development evidence."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from coopgate.episode_store_v21 import observations
import experiments.top_two_candidates.run as experiment


def run():
    e=experiment.e;source=experiment.OUT
    experiment.owned(source);plan=experiment.guard();r=e.read(source/'report.json')
    if (not r['completed'] or r['additional_actual_sumo_executions']!=594 or
        r['effective_paired_executions']!=1600 or r['new_independent_initials']!=0 or
        any(c['actual_native_exit_code'] for c in r['children'])):
        raise ValueError('Original complete scope and actual native exits required')
    out=e.BASE/'development/local_top_two_analysis_20261104';out.mkdir(exist_ok=False)
    usage={};rows=[]
    for shard in range(4):
        for block in e.read(source/f'shard{shard}/report.json')['blocks']:
            db=ROOT/block['database']
            if e.sha(db)!=block['database_sha256']:raise ValueError('Reviewed bytes changed')
            for key,record,trace in observations(db,block['metadata'],block['keys']):
                name=record['variant']+':'+record['method']
                u=usage.setdefault(name,dict(episodes=0,rank_two_adopted_episodes=0,
                    rank_two_control_steps=0,total_learning_steps=0,total_steps=0))
                selected=[t for t in trace if t['actual_learning_used'] and t['composition']>=100]
                u['episodes']+=1;u['rank_two_adopted_episodes']+=int(bool(selected))
                u['rank_two_control_steps']+=len(selected);u['total_steps']+=len(trace)
                u['total_learning_steps']+=sum(t['actual_learning_used'] for t in trace)
    if len(usage)!=6 or any(u['episodes']!=100 for u in usage.values()):raise ValueError('All600target observations required')
    text=['# Complete bounded local top two candidate development','',
        'All100original cases and three fixed MAPPO actor seeds remain. The600candidate',
        'observations comprise594new actual SUMO runs plus6completed preflights.',
        'The1000completed reference observations are reused, giving1600effective paired',
        'observations and ZERO new independent initials. The previous fresh80are now',
        'reused development data. No statistical error or formal6is allocated.','',
        '| Source cohort | Variant | Model | Adoption | Harmful adopted | Mean gain | P95 ms | Max ms | >500ms |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for stage,label in [('fresh','Prior80'),('regression','Known20')]:
        for variant in ('history','joint','reserve','top_history','top_reserve'):
            for seed in experiment.SEEDS:
                value=r['results'][stage+':'+variant+':'+seed]
                rows.append(dict(source_stage=stage,variant=variant,seed=seed,**value))
                text.append(f"| {label} | {variant} | {seed} | {value['adopted']}/{value['initials']} | {value['harmful_adopted']} | {value['mean_paired_percentage_gain']:.6f}% | {value['callback_p95_ms']:.3f} | {value['callback_max_ms']:.3f} | {value['callback_over_500ms']} |")
    text += ['', 'The complete development promotion screen was fixed before collection:',
        'all three prior80mean gains >=history+0.05percentage points, no adoption',
        'loss, no failures, no harm on prior80, <=14known model-initial harms and',
        'no newly introduced known harm versus history. It is not a risk certificate.','']
    for variant,value in r['development_promotion'].items():
        text.append(variant+': '+('PASSES development screen.' if value['passed'] else 'FAILS development screen.'))
        text.append('```json\n'+json.dumps(value,indent=2)+'\n```')
    text += ['', 'Actual selected rank-two commands (not merely generated candidates):','```json',
        json.dumps(usage,indent=2),'```','',
        'Every new episode has complete controller and physics replay. Successful',
        f"zero-adoption fallback trajectories exactly equal baseline in{r['zero_adoption_exact_physical_fallbacks']}cases.",
        'All earlier failed owners and unfavorable methods remain. Different batches',
        'do not establish causal timing speedups. Scores are finite forecast margins;',
        'actor logits are preferences, not safety probabilities. No communication',
        'latency metric, general raw-MAPPO superiority or hard-real-time claim follows.']
    e.write(out/'report.json',dict(completed=True,source_report_sha256=e.sha(source/'report.json'),
        source_owner_sha256=e.sha(source/'owner_native_exit.json'),source_plan_sha256=e.sha(source/'plan.json'),
        effective_observations=1600,additional_actual_sumo=594,new_independent_initials=0,
        results=rows,rank_two_execution=usage,development_promotion=r['development_promotion'],
        new_sumo_runs=0,new_error_allocated=False))
    with (out/'results.md').open('x',encoding='utf-8',newline='\n') as f:f.write('\n'.join(text)+'\n')
    print(json.dumps(dict(completed=True,promotion=r['development_promotion'],usage=usage)),flush=True)


if __name__=='__main__':run()
