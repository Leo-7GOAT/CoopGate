"""Supplementary memory ablation and isolated soft-budget engineering timing."""
from pathlib import Path
import argparse,json,csv,time
import numpy as np
from coopgate.config import Config
from coopgate.mappo import load_policy
from coopgate.sim import SumoEnv

root=Path(__file__).resolve().parents[1]
out=root/'artifacts/v2_final'


def main(seed,timing):
    protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    cfg=Config(**protocol['config'])
    if timing:cfg.gate_budget_ms=150.
    model,_=load_policy(root/f'artifacts/v2_final/seed{seed}/policy.pt')
    name='timing' if timing else f'memory_seed{seed}'
    if (out/f'{name}.csv').exists():
        raise RuntimeError('Existing results will not be overwritten; choose a new --out directory.')
    rows=[];latencies={};gates={}
    for split in (['test'] if timing else ['test','shift']):
        scenes=protocol[f'{split}_scenes'][:5] if timing else protocol[f'{split}_scenes']
        for method in (['traditional','independent','paired'] if timing else ['paired_no_memory']):
            loop=[];gate=[]
            with SumoEnv(cfg,method) as env:
                for scene in scenes:
                    obs=env.reset(**scene)
                    while True:
                        start=time.perf_counter()
                        a,_,_=model.act(obs,True)
                        obs,_,t,u,metrics=env.step(a)
                        loop.append((time.perf_counter()-start)*1000)
                        gate.append(env.gate_ms[-1])
                        if t or u:break
                    rows.append({'training_seed':seed,'split':split,**metrics})
            def percentiles(values):
                return {'steps':len(values),'mean_ms':float(np.mean(values)),
                        'p95_ms':float(np.percentile(values,95)), 'max_ms':float(np.max(values)),
                        'over_150ms':int(np.sum(np.array(values)>150)),
                        'over_500ms':int(np.sum(np.array(values)>500))}
            latencies[method]=percentiles(loop);gates[method]=percentiles(gate)
    name='timing' if timing else f'memory_seed{seed}'
    with (out/f'{name}.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    if timing:
        (out/'timing.json').write_text(json.dumps({'config':cfg.dict(),'loop':latencies,'gate':gates,
            'scope':'single local process after primary experiments; soft budget, not hard realtime'},indent=2),newline='\n')
    print(json.dumps({'complete':name,'episodes':len(rows)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=31);p.add_argument('--timing',action='store_true')
    p.add_argument('--out',default='runs/v2_reproduction')
    a=p.parse_args();out=root/a.out
    out.mkdir(parents=True,exist_ok=True)
    if not (out/'protocol.json').exists():
        import shutil
        shutil.copy2(root/'artifacts/v2_final/protocol.json',out/'protocol.json')
    main(a.seed,a.timing)
