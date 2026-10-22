"""Replay the six published V10 regressions; diagnostic, not new test evidence."""
import csv
import json
from pathlib import Path
import numpy as np
from coopgate.disturbance_v9 import DisturbedMergeEnv
from coopgate.experiment_v11 import prepare
from coopgate.learning_v6 import load_models
from coopgate.merge_v4 import MergeConfig

ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/v11'


def main():
    old=list(csv.DictReader((ROOT/'artifacts/v10/study/episodes.csv').open()))
    methods=('v9_learned_pair','v10_rollout','smooth_learned');modes=('uniform','bias','pulse')
    lookup={(r['scenario'],int(r['seed']),r['disturbance'],r['method']):r for r in old}
    keys=sorted({(r['scenario'],int(r['seed'])) for r in old})
    regressions=[(s,seed) for s,seed in keys if np.mean([float(lookup[s,seed,d,'rollout_learned']['mean_capped_finish_s'])-
        float(lookup[s,seed,d,'v9_learned_pair']['mean_capped_finish_s']) for d in modes])>1e-8]
    assert len(regressions)==6
    models=load_models(ROOT/'artifacts/v6',[201,211,223]);c=MergeConfig();rows=[]
    with DisturbedMergeEnv(c,'constant') as env:
        for scenario,seed in regressions:
            for mode in modes:
                for method in methods:
                    env.reset_disturbed(seed,scenario,mode,.5)
                    env.gate,_=prepare(method,models,env.x,env.prev,c,.5,seed)
                    while True:
                        _,_,done,truncated,metrics=env.step([4]*3)
                        if done or truncated:break
                    commands=np.array([[0.,0.,0.]]+[t['command'] for t in env.trace])
                    rows.append({**metrics,'method':method,'total_command_variation':float(np.abs(np.diff(commands,axis=0)).sum())})
                    if method!='smooth_learned':
                        reference=lookup[scenario,seed,mode,'rollout_learned' if method=='v10_rollout' else method]
                        assert abs(metrics['mean_capped_finish_s']-float(reference['mean_capped_finish_s']))<1e-6
            print(f'Replayed old adverse scene {scenario}/{seed}',flush=True)
    clusters=[]
    for scenario,seed in regressions:
        means={m:float(np.mean([r['mean_capped_finish_s'] for r in rows if r['seed']==seed and r['method']==m])) for m in methods}
        clusters.append({'scenario':scenario,'seed':seed,'mean_cost_s':means,
            'new_worse_than_v9':means['smooth_learned']>means['v9_learned_pair']+1e-8,
            'new_worse_than_v10':means['smooth_learned']>means['v10_rollout']+1e-8})
    result={'description':'Post-study diagnostic replay of all six initial-state clusters where published V10 was worse than V9 under the three perturbed modes. Selected adverse historical cases; not independent generalization evidence. Some were also used in development.',
        'executions':len(rows),'clusters':clusters,'remaining_worse_than_v9':sum(r['new_worse_than_v9'] for r in clusters),
        'remaining_worse_than_v10':sum(r['new_worse_than_v10'] for r in clusters),'episodes':rows}
    (D/'historical_regression_audit.json').write_text(json.dumps(result,indent=2)+'\n',newline='\n')
    print(json.dumps({k:v for k,v in result.items() if k!='episodes'},indent=2))


if __name__=='__main__':main()
