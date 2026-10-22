"""Development-only audit of why frozen V2 proposals are rejected."""
from pathlib import Path
from itertools import product
import json
from collections import Counter
import numpy as np
from coopgate.config import Config
from coopgate.controllers import backup_intents, Intent
from coopgate.paired import ClosedLoopPredictor
from coopgate.mappo import load_policy
from coopgate.sim import SumoEnv

ROOT=Path(__file__).resolve().parents[1]
cfg=Config(commit_lane_changes=True,continuous_network=True,gate_budget_ms=10000)
counts=Counter(); episodes=[]
for policy_seed in (31,47,59):
    model,_=load_policy(ROOT/f'artifacts/v2_final/seed{policy_seed}/policy.pt')
    with SumoEnv(cfg,'paired') as env:
        original=None
        for k in range(3):
            obs=env.reset(61000+k,('nominal','dense','braking')[k])
            original=env.gate
            class Audit:
                def select(self,states,learned,context):
                    counts['decisions']+=1
                    backup=backup_intents(states,cfg);learned=list(learned)
                    for i in range(3):
                        if context['lane_remaining'][i]>1e-6:
                            backup[i]=Intent(backup[i].speed,int(context['lane_targets'][i]))
                            learned[i]=Intent(learned[i].speed,int(context['lane_targets'][i]))
                    predictor=ClosedLoopPredictor(cfg)
                    hypotheses=(-2.5,0,1.5)
                    base=[predictor.rollout(states,backup,context,a) for a in hypotheses]
                    counts['backup_geometry_failed']+=min(f.margin for f in base)<0
                    counts['backup_reserve_failed']+=min(f.reserve for f in base)<0
                    for bits in product((False,True),repeat=3):
                        if not any(bits):continue
                        counts['candidates']+=1
                        proposed=[learned[i] if bits[i] else backup[i] for i in range(3)]
                        pred=[predictor.rollout(states,proposed,context,a) for a in hypotheses]
                        geometric=min(f.margin for f in pred)>=0
                        reserve=min(f.reserve for f in pred)>=0
                        gain=min(f.value-b.value for f,b in zip(pred,base))>=.5
                        counts['geometry_failed']+=not geometric
                        counts['reserve_failed']+=not reserve
                        counts['gain_failed']+=not gain
                        counts['all_pass']+=geometric and reserve and gain
                        counts['geometry_only_blocker']+=not geometric and reserve and gain
                        counts['reserve_only_blocker']+=geometric and not reserve and gain
                        counts['gain_only_blocker']+=geometric and reserve and not gain
                    return original.select(states,learned,context)
            env.gate=Audit()
            while True:
                actions,_,_=model.act(obs,True)
                obs,_,t,u,metrics=env.step(actions)
                if t or u:break
            episodes.append({'policy_seed':policy_seed,**metrics})
            print(json.dumps({'policy':policy_seed,'scene':k,'done':True}),flush=True)
out=ROOT/'artifacts/v3/development';out.mkdir(parents=True,exist_ok=True)
(out/'v2_rejection_audit.json').write_text(json.dumps({'counts':{k:int(v) for k,v in counts.items()},'episodes':episodes,
    'scope':'nine development executions, three new scene seeds shared by three frozen V2 policies; not an independent performance test'},indent=2),newline='\n')
print(dict(counts))
