"""Independent SUMO calibration of simultaneous cost/time residuals."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from coopgate.disturbance_v9 import DisturbedMergeEnv
from coopgate.ego_v13 import EgoRolloutGate
from coopgate.learned_rollout_v13 import LearnedRolloutGate
from coopgate.ego_confidence_v13 import predict_ego
from coopgate.confidence_v12 import conformal_upper,conflict_group
from experiment_ego_v13 import run_episode

ROOT=Path(__file__).resolve().parents[1]


def calibrate(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);seeds=[701,711,721]
    paths={s:ROOT/f'runs/v13_ego_release/dqn_{s}/policy.pt' for s in seeds}
    for path in paths.values():
        if not path.with_name('metadata.json').exists():raise RuntimeError('Release training incomplete')
    rng=np.random.default_rng(202610160213);episodes=[{'seed':int(seed),'regime':str(rng.choice(['nominal','dense'])),
        'mode':str(rng.choice(['clean','uniform','bias','pulse']))} for seed in rng.choice(np.arange(31000000,32000000),256,replace=False)]
    protocol={'episodes':episodes,'alpha':.1,'residual':'max(delta_cost-predicted_cost, delta_time-predicted_time)',
        'cost_margin_seconds':.05,'time_noninferiority_margin_seconds':0.,'expected_sumo_runs':1024,
        'models':{str(s):hashlib.sha256(p.read_bytes()).hexdigest() for s,p in paths.items()}}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    learned={s:LearnedRolloutGate(path) for s,path in paths.items()};rows=[];residuals=[];start=time.perf_counter()
    with DisturbedMergeEnv() as env:
        for i,item in enumerate(episodes):
            b,_,initial=run_episode(env,item['seed'],item['regime'],item['mode'],lambda x:EgoRolloutGate())
            rows.append({**b,'controller':'rollout'})
            for s in seeds:
                a,_,_=run_episode(env,item['seed'],item['regime'],item['mode'],lambda x,s=s:LearnedRolloutGate(paths[s]))
                rows.append({**a,'controller':f'augmented_{s}'})
                predicted=predict_ego(initial,learned[s]);delta=np.array([a['ego_cost']-b['ego_cost'],a['ego_time']-b['ego_time']])
                residuals.append({**item,'policy':s,'group':conflict_group(initial),'predicted':predicted.tolist(),
                    'delta':delta.tolist(),'residual':(delta-predicted).tolist(),'score':float(np.max(delta-predicted))})
            if (i+1)%16==0:print(json.dumps({'calibration_initials':i+1,'sumo_runs':len(rows),'elapsed_s':time.perf_counter()-start}),flush=True)
    calibration={}
    for s in seeds:
        groups={}
        for group in ('low_conflict','high_conflict'):
            rr=[r for r in residuals if r['policy']==s and r['group']==group]
            groups[group]={'n':len(rr),'q':conformal_upper([r['score'] for r in rr],.1),
                'q_cost_only':conformal_upper([r['residual'][0] for r in rr],.1)}
        calibration[str(s)]={'alpha':.1,'groups':groups}
    for name,value in [('calibration',calibration),('episodes',rows),('residuals',residuals)]:
        (out/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({'calibration':calibration,'sumo_runs':len(rows)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='runs/v13_ego_calibration');a=p.parse_args();calibrate(a.out)
