"""Disjoint SUMO calibration; one random error mode per independent initial."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from coopgate.disturbance_v9 import DisturbedMergeEnv
from coopgate.marl_v12 import MARLGate,load_policy
from coopgate.confidence_v12 import predict_costs,conformal_upper
from coopgate.single_v12 import SingleRolloutGate
from experiment_single_v12 import run_episode


def run(out,models,n=256):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);rng=np.random.default_rng(20261016122)
    seeds=rng.choice(np.arange(22000000,23000000),n,replace=False).tolist()
    protocols=[{'seed':int(seed),'regime':str(rng.choice(['nominal','dense'])),
        'mode':str(rng.choice(['clean','uniform','bias','pulse']))} for i,seed in enumerate(seeds)]
    (out/'protocol.json').write_text(json.dumps({'episodes':protocols,'alpha':.1,'minimum_gain_s':.05,
        'baseline':'single_rollout','objective':'mean arrival seconds + 0.05 * total command variation',
        'models':{str(k):hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in models.items()},
        'calibration_unit':'one independent initial and one randomly assigned error mode'},indent=2)+'\n')
    policies={k:load_policy(v)[0] for k,v in models.items()};rows=[];initials=[]
    with DisturbedMergeEnv() as env:
        for i,item in enumerate(protocols):
            factories={'baseline':lambda x:SingleRolloutGate()}
            factories.update({str(k):(lambda x,m=m:MARLGate(m,mode='single')) for k,m in policies.items()})
            for method in rng.permutation(list(factories)):
                row,_,initial=run_episode(env,item['seed'],item['regime'],item['mode'],factories[method])
                rows.append({**row,'policy':str(method)})
                if method=='baseline':initials.append((item['seed'],initial))
            if (i+1)%16==0:print(json.dumps({'calibration_initials':i+1,'sumo_runs':len(rows)}),flush=True)
    by_seed={seed:x for seed,x in initials};x=np.array([by_seed[item['seed']] for item in protocols])
    bt,bv=predict_costs(x);predicted_base=bt+.05*bv;calibrations={};residual_rows=[]
    for k,m in policies.items():
        lt,lv=predict_costs(x,m);predictions=lt+.05*lv-predicted_base
        groups={g:[] for g in ('low_conflict','high_conflict')}
        for i,item in enumerate(protocols):
            baseline=next(r for r in rows if r['seed']==item['seed'] and r['policy']=='baseline')
            learned=next(r for r in rows if r['seed']==item['seed'] and r['policy']==str(k))
            actual_delta=learned['mean_capped_finish_s']-baseline['mean_capped_finish_s']+.05*(learned['variation']-baseline['variation'])
            residual=actual_delta-predictions[i]
            groups[baseline['group']].append(residual)
            residual_rows.append({'seed':item['seed'],'policy':k,'group':baseline['group'],
                'predicted_delta':float(predictions[i]),'actual_delta':actual_delta,
                'residual':float(residual)})
        calibrations[str(k)]={'alpha':.1,'minimum_gain_s':.05,'baseline':'single_rollout',
            'groups':{g:{'n':len(r),'q':conformal_upper(r,.1)} for g,r in groups.items()}}
    for name,value in [('episodes',rows),('calibration',calibrations),('residuals',residual_rows)]:
        (out/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(calibrations),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--models',default='runs/v12_single_release')
    p.add_argument('--n',type=int,default=256);a=p.parse_args()
    run(a.out,{s:Path(a.models)/f'ppo_{s}/policy.pt' for s in [411,421,431]},a.n)
