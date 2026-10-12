"""Frozen independent V3 study. No controller selection on evaluation outcomes."""
from pathlib import Path
import argparse,csv,json,hashlib,shutil,time
import numpy as np
from coopgate.recovery import RecoveryConfig
from coopgate.platoon import PlatoonEnv
from coopgate.learning_v3 import load
from coopgate.certification import certify

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v3/final'
SEEDS=[83,97,109]
METHODS=['raw_mpc','raw_rl','joint_mpc','individual','independent_front','no_memory','joint_bc','joint_constant','joint_rl']


def write(path,data):path.write_text(json.dumps(data,indent=2),newline='\n')


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():raise RuntimeError('Already frozen')
    rng=np.random.default_rng(2026101503)
    def scenes(n,shift=False):
        return [{'seed':int(rng.integers(10000000,1000000000)),'scenario':'double_brake' if shift else str(rng.choice(['cruise','moderate','hard_brake']))} for _ in range(n)]
    write(OUT/'protocol.json',{'config':RecoveryConfig().dict(),'training_seeds':SEEDS,'steps':12000,'bc_episodes':30,
        'methods':METHODS,'test_scenes':scenes(30),'shift_scenes':scenes(30,True),'certification_scenes':scenes(100),
        'certification_controllers':['joint_mpc','joint_rl83','joint_rl97','joint_rl109'],
        'endpoints':['collision','gap_violation'],'family_size':8,'target':.05,'delta':.05,
        'scope':'Fixed-order single-lane SUMO benchmark with exact sampled acceleration adapter; distinct task from V2 merging',
        'primary_measure':'mean controlled distance in fixed 25-second episodes, zero after a collision; report per training seed',
        'training_selector':'final checkpoint, no held-out selection','timing':'primary runs may be concurrent; engineering timing separately'})


def worker(seed,out,checkpoint):
    protocol=json.loads((OUT/'protocol.json').read_text());cfg=RecoveryConfig(**protocol['config'])
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'episodes.csv').exists():raise RuntimeError('Refusing overwrite')
    model,_=load(checkpoint);bc,_=load(Path(checkpoint).parent/'bc.pt')
    for name in ('policy.pt','bc.pt','training.csv','config.json'):shutil.copy2(Path(checkpoint).parent/name,out/name)
    rows=[];start=time.perf_counter()
    settings=[('test',protocol['test_scenes'],METHODS),('shift',protocol['shift_scenes'],METHODS),
              ('certification',protocol['certification_scenes'],['joint_rl']+(['joint_mpc'] if seed==83 else []))]
    for split,scenes,methods in settings:
        for method in methods:
            with PlatoonEnv(cfg,'joint_rl' if method in ('joint_bc','joint_constant') else method) as env:
                for i,scene in enumerate(scenes):
                    obs=env.reset(**scene)
                    while True:
                        a=np.full(3,4) if method=='joint_constant' else (bc if method=='joint_bc' else model).act(obs,True)[0]
                        obs,_,t,u,metrics=env.step(a)
                        if t or u:break
                    rows.append({'training_seed':seed,'split':split,**metrics,'method':method})
                    if i==0 and split!='certification':write(out/f'trace_{split}_{method}.json',env.trace)
            with (out/'episodes.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
            print(json.dumps({'seed':seed,'split':split,'method':method,'episodes':len(scenes)}),flush=True)
    cert=[]
    for method in settings[-1][2]:
        group=[r for r in rows if r['split']=='certification' and r['method']==method]
        for event in ('collision','gap_violation'):
            failures=sum(r['collision'] if event=='collision' else r['gap_violation_substeps']>0 for r in group)
            cert.append({'seed':seed,'method':method,'endpoint':event,**certify(failures,len(group),family_size=8)})
    write(out/'certification.json',cert)
    write(out/'manifest.json',{'protocol_sha256':hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest(),
        'checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),'episodes':len(rows),'wall_s':time.perf_counter()-start,
        'source_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/coopgate').glob('*.py')}})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--seed',type=int,choices=SEEDS)
    p.add_argument('--out');p.add_argument('--checkpoint');a=p.parse_args()
    if a.prepare:prepare()
    elif a.seed is None:p.error('--seed required')
    else:worker(a.seed,a.out or OUT/f'seed{a.seed}',a.checkpoint or ROOT/f'runs/v3_train{a.seed}/policy.pt')
