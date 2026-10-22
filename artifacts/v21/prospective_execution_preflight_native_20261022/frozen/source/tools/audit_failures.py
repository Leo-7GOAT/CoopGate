from pathlib import Path
import json
from coopgate.sim import SumoEnv
from coopgate.mappo import load_policy

out=Path('artifacts/v2/audit');out.mkdir(parents=True,exist_ok=True)
model,_=load_policy('artifacts/seed19/policy.pt')
report=[]
for seed,scenario in ((20002,'nominal'),(22003,'braking')):
    with SumoEnv(method='joint_robust') as e:
        obs=e.reset(seed,scenario)
        while True:
            a,_,_=model.act(obs,True)
            obs,_,t,u,info=e.step(a)
            if t or u:break
        entry={'metrics':info,'collided_ids':sorted(e.collided_ids),'last_decisions':e.trace[-8:]}
        report.append(entry)
(out/'v1_failures.json').write_text(json.dumps(report,indent=2),encoding='utf-8',newline='\n')
print(json.dumps([{'seed':r['metrics']['seed'],'ids':r['collided_ids'],'last_modes':r['last_decisions'][-1]['modes'],'last_emergency':r['last_decisions'][-1]['emergency']} for r in report]))
