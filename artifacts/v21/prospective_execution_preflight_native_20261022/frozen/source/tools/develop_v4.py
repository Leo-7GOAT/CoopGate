"""Small development-only audit; never used as the held-out test set."""
from pathlib import Path
import csv,json,time
from coopgate.merge_v4 import MergeEnv

OUT=Path('runs/v4_development');OUT.mkdir(parents=True,exist_ok=True)
rows=[]
for method in ('constant','eta','search','individual','no_memory'):
    start=time.perf_counter()
    with MergeEnv(method=method) as e:
        for seed in range(100,104):
            e.reset(seed,'nominal' if seed%2==0 else 'dense')
            while True:
                _,_,t,u,m=e.step([4,4,4])
                if t or u:break
            rows.append(m)
    print(json.dumps({'method':method,'seconds':time.perf_counter()-start,'rows':rows[-4:]}),flush=True)
with (OUT/'episodes.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
