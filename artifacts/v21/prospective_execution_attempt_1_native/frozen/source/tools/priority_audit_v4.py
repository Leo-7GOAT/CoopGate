"""Post-study development diagnostic, NOT a held-out learned-policy result.

Compare three static tie priorities among exactly the same feasible modes.
The per-scene hindsight minimum is an unattainable comparison oracle, not a
deployable controller or evidence that an RL policy can learn its selection.
"""
from pathlib import Path
import json,csv,time
import numpy as np
from coopgate.merge_v4 import MergeGate,MergeEnv,backup,admissible,certify

class PriorityGate(MergeGate):
    def __init__(self,c,order):
        super().__init__(c,'constant');self.weights=np.zeros(3)
        self.weights[list(order)]=[4,2,1]
    def select(self,x,prev,proposal):
        start=time.perf_counter();c=self.c;b=backup(prev,c);p=admissible(np.full(3,c.accel),prev,c)
        candidates=np.where(self.bits,p,b);valid=certify(x,prev,candidates,c)
        # Rank is lexicographically secondary, never trades primary intervention.
        cost=np.sum((candidates-p)**2,axis=1)
        ids=np.flatnonzero(valid)
        k=min(ids,key=lambda i:(float(cost[i]),-float(np.dot(candidates[i],self.weights)))) if len(ids) else 0
        chosen=candidates[k]
        return chosen,{'in_set':bool(certify(x,prev,b,c)[0]),'certified':bool(certify(x,prev,chosen,c)[0]),
                       'intervention':float(np.mean(np.abs(chosen-p)>1e-5)),'gate_ms':(time.perf_counter()-start)*1000}


def main():
    out=Path('artifacts/v4/priority_audit');out.mkdir(parents=True,exist_ok=True)
    if (out/'episodes.csv').exists():raise RuntimeError('Refusing overwrite')
    rows=[]
    for name,order in [('original',None),('012',(0,1,2)),('021',(0,2,1)),('102',(1,0,2))]:
        with MergeEnv(method='constant') as e:
            for scenario in ('nominal','dense'):
                for seed in range(200,210):
                    e.reset(seed,scenario)
                    if order is not None:e.gate=PriorityGate(e.c,order)
                    while True:
                        _,_,t,u,m=e.step([4]*3)
                        if t or u:break
                    rows.append({**m,'priority':name})
        print(json.dumps({'priority':name,'episodes':20}),flush=True)
    with (out/'episodes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    deltas=[]
    for scenario in ('nominal','dense'):
        for seed in range(200,210):
            g=[r for r in rows if r['seed']==seed and r['scenario']==scenario]
            base=next(r for r in g if r['priority']=='original');best=min(g,key=lambda r:r['mean_capped_finish_s'])
            deltas.append({'seed':seed,'scenario':scenario,'seconds_saved_oracle':base['mean_capped_finish_s']-best['mean_capped_finish_s'],
                           'best_priority':best['priority'],'original_s':base['mean_capped_finish_s'],'oracle_s':best['mean_capped_finish_s']})
    result={'scope':'80 additional development executions, 20 matched cases. Post-hoc hindsight lower bound among static controllers, not a learned policy test.',
            'mean_seconds_saved':float(np.mean([r['seconds_saved_oracle'] for r in deltas])),
            'improved_cases':sum(r['seconds_saved_oracle']>1e-6 for r in deltas),'cases':deltas,
            'collisions':sum(r['collision'] for r in rows),'violating_episodes':sum(r['contract_violations']>0 for r in rows)}
    (out/'summary.json').write_text(json.dumps(result,indent=2),newline='\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
