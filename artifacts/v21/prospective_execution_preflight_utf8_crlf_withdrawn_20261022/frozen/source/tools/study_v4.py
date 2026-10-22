"""Frozen V4 pilot: matched scenes and final checkpoints, no test selection."""
from pathlib import Path
import argparse,csv,json,hashlib,shutil
import numpy as np
from coopgate.merge_v4 import MergeConfig,MergeEnv
from coopgate.learning_v4 import load
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'artifacts/v4'
SEEDS=[127,139,151]
METHODS=['constant','eta','search','bc','raw_rl','no_memory','individual','joint_rl']


def write(p,x):p.write_text(json.dumps(x,indent=2),newline='\n')


def prepare():
    DATA.mkdir(parents=True,exist_ok=True)
    if (DATA/'protocol.json').exists():raise RuntimeError('Already frozen')
    rng=np.random.default_rng(2026101304)
    protocol={'config':MergeConfig().dict(),'training_seeds':SEEDS,'train_steps':6000,'bc_episodes':20,
        'methods':METHODS,'splits':{scenario:[{'seed':int(rng.integers(10000000,1000000000)),'scenario':scenario} for _ in range(20)] for scenario in ('nominal','dense')},
        'selector':'final checkpoint, no held-out checkpoint or hyperparameter selection',
        'training_distribution':'nominal only; dense has higher speeds and closer arrivals',
        'primary':'Mean goal-arrival time per car capped at 25s; collisions score 25s for all cars. Report completion and safety alongside time.',
        'scope':'Three fully controlled fixed-route cars, two incoming edges and one outgoing edge, no internal junction links, exact sampled speed adapter.',
        'search_baseline':'All 125 joint actions from the same five action values, exact backup certificate, two-second constant-command progress surrogate.',
        'statistics':'Per-training-seed paired scenes; descriptive bootstrap only. No risk certificate from this small pilot.',
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/coopgate').glob('*.py')}}
    write(DATA/'protocol.json',protocol)


def run(seed,out,checkpoint):
    protocol=json.loads((DATA/'protocol.json').read_text());cfg=MergeConfig(**protocol['config']);out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'episodes.csv').exists():raise RuntimeError('Refusing overwrite')
    for p in ('policy.pt','bc.pt','config.json','training.csv'):shutil.copy2(Path(checkpoint).parent/p,out/p)
    model,_=load(checkpoint);bc,_=load(Path(checkpoint).parent/'bc.pt');rows=[]
    for split,scenes in protocol['splits'].items():
        for method in protocol['methods']:
            with MergeEnv(cfg,'joint_rl' if method=='bc' else method) as env:
                for i,scene in enumerate(scenes):
                    obs=env.reset(**scene)
                    while True:
                        a=(bc if method=='bc' else model).act(obs,True)[0]
                        obs,_,t,u,metrics=env.step(a)
                        if t or u:break
                    rows.append({**metrics,'method':method,'training_seed':seed,'split':split})
                    if i==0:write(out/f'trace_{split}_{method}.json',env.trace)
            with (out/'episodes.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
            print(json.dumps({'seed':seed,'split':split,'method':method,'n':20}),flush=True)
    write(out/'manifest.json',{'episodes':len(rows),'protocol_sha256':hashlib.sha256((DATA/'protocol.json').read_bytes()).hexdigest(),
        'checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--seed',type=int,choices=SEEDS)
    p.add_argument('--out');p.add_argument('--checkpoint');a=p.parse_args()
    if a.prepare:prepare()
    elif a.seed is None:p.error('--seed required')
    else:run(a.seed,a.out or ROOT/f'runs/v4_reproduction/seed{a.seed}',a.checkpoint or DATA/f'seed{a.seed}/policy.pt')
