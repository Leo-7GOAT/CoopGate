"""V11 development on previously published V10 scenes, not a fresh test."""
import argparse
import json
from pathlib import Path
import numpy as np
from coopgate.disturbance_v9 import DisturbedMergeEnv
from coopgate.merge_v4 import MergeConfig
from coopgate.rollout_v10 import RolloutGate
from coopgate.smooth_v11 import SmoothGate
from coopgate.learning_v6 import load_models, predict

ROOT=Path(__file__).resolve().parents[1]


def run(count, methods, out):
    models=load_models(ROOT/'artifacts/v6',[201,211,223])
    old=json.loads((ROOT/'artifacts/v10/protocol.json').read_text())
    rows=[]
    with DisturbedMergeEnv(MergeConfig(),'constant') as env:
        for regime,seeds in old['test_seeds'].items():
            for seed in seeds[:count]:
                for mode in old['disturbances']:
                    for method in methods:
                        env.reset_disturbed(seed,regime,mode,.5)
                        p=int(predict(models,env.x[None])[0][0])
                        if method=='v10':
                            gate=RolloutGate(env.x,env.prev,env.c,.5,'pair',p)
                        else:
                            gate=SmoothGate(env.x,env.prev,env.c,.5,'pair',p,
                                quantum=1.25 if 'coarse' in method else .625,
                                ternary='ternary' in method, smooth='timeonly' not in method,
                                retain_binary='only' not in method, family_rollouts='families' in method)
                        env.gate=gate
                        while True:
                            _,_,done,truncated,metrics=env.step([4]*3)
                            if done or truncated:break
                        commands=np.array([[0.,0.,0.]]+[r['command'] for r in env.trace])
                        rows.append({**metrics,'method':method,
                            'total_variation':float(np.abs(np.diff(commands,axis=0)).sum()),
                            'cycle_p95_ms':float(np.percentile([r['gate_ms'] for r in env.trace],95))})
                out.write_text(json.dumps(rows,indent=2)+'\n',newline='\n')
                print(json.dumps({'regime':regime,'seed':seed,'rows':len(rows)}),flush=True)
    for method in methods:
        group=[r for r in rows if r['method']==method and r['disturbance']!='clean']
        print(json.dumps({'method':method,'n':len(group),
            'cost':float(np.mean([r['mean_capped_finish_s'] for r in group])),
            'variation':float(np.mean([r['total_variation'] for r in group])),
            'jerk':float(np.mean([r['mean_jerk'] for r in group])),
            'complete':sum(r['completed'] for r in group),'abort':sum(r['aborted'] for r in group),
            'p95':float(np.percentile([r['cycle_p95_ms'] for r in group],95))}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=8)
    p.add_argument('--methods',nargs='+',default=['v10','lex','lex_coarse','lex_ternary','timeonly_ternary'])
    p.add_argument('--out',type=Path,default=Path('runs/v11_development.json'))
    a=p.parse_args();run(a.count,a.methods,a.out)
