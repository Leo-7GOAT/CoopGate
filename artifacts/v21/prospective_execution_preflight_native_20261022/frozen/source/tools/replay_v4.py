from pathlib import Path
import argparse,json,time
import numpy as np
from coopgate.merge_v4 import MergeEnv,MergeConfig
from coopgate.learning_v4 import load
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/v4'


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[127,139,151],default=127)
    p.add_argument('--gui',action='store_true');p.add_argument('--timing',action='store_true')
    p.add_argument('--out',type=Path,default=ROOT/'runs/v4_replay');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'results.json').exists():raise RuntimeError('Refusing overwrite')
    protocol=json.loads((D/'protocol.json').read_text());cfg=MergeConfig(**protocol['config']);model,_=load(D/f'seed{a.seed}/policy.pt')
    result={}
    for method in (('constant','eta','search','joint_rl') if a.timing else ('joint_rl',)):
        durations=[];metrics=[]
        with MergeEnv(cfg,method,a.gui) as e:
            for index,scene in enumerate(protocol['splits']['nominal'][:5] if a.timing else protocol['splits']['dense'][:1]):
                obs=e.reset(**scene)
                if a.gui:
                    for i,v in enumerate(e.ids):e.conn.vehicle.setColor(v,[(20,150,220),(230,100,30),(140,60,190)][i])
                while True:
                    if a.gui and len(e.trace)==6:e.conn.gui.screenshot('View #0',str((a.out/'sumo_gui.png').resolve()),width=1400,height=700)
                    start=time.perf_counter();act=model.act(obs,True)[0];obs,_,t,u,m=e.step(act);durations.append((time.perf_counter()-start)*1000)
                    if t or u:break
                metrics.append(m)
                if not a.timing:(a.out/'trace.json').write_text(json.dumps(e.trace,indent=2),newline='\n')
        result[method]={'steps':len(durations),'mean_ms':float(np.mean(durations)),'p95_ms':float(np.percentile(durations,95)),
                        'max_ms':max(durations),'over_500ms':sum(t>500 for t in durations),'episodes':metrics}
    (a.out/'results.json').write_text(json.dumps(result,indent=2),newline='\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
