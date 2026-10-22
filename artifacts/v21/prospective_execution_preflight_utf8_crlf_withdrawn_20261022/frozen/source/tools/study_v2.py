"""Frozen paired evaluation, independent episode certification, and OOD stress."""
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
from coopgate.config import Config
from coopgate.sim import SumoEnv
from coopgate.mappo import load_policy
from coopgate.experiment import summarize
from coopgate.certification import certify

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts'/'v2_final'
METHODS=['traditional','mappo','independent','joint_robust','paired_no_gain','paired_no_reserve','paired']
SEEDS=[31,47,59]


def write_json(path,data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8',newline='\n')


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'protocol.json'
    if path.exists():
        raise RuntimeError('Protocol already frozen; do not overwrite after inspecting results.')
    rng=np.random.default_rng(20261014)
    def scenes(n):
        return [{'seed':int(rng.integers(100000,1000000000)),
                 'scenario':str(rng.choice(['nominal','dense','braking']))} for _ in range(n)]
    calibration=scenes(100)
    testing=scenes(15)
    shifted=[{'seed':int(rng.integers(100000,1000000000)),'scenario':'shift_braking'} for _ in range(15)]
    config=Config(commit_lane_changes=True,continuous_network=True,gate_budget_ms=10000).dict()
    write_json(path,{'study':'v2 fixed pilot','training_seeds':SEEDS,'training_steps':30000,
                     'bc_episodes_per_seed':90,'bc_epochs':20,'online_bc_weight':.1,
                     'methods':METHODS,'config':config,
                     'certification_scenes':calibration,'test_scenes':testing,'shift_scenes':shifted,
                     'certification_controllers':['traditional','paired31','paired47','paired59'],
                     'risk_endpoints':['collision','not_success'],'family_size':8,'delta':.05,'risk_target':.05,
                     'execution':'full deterministic mode enumeration; timing audit separately uses deployment budget',
                     'scope':'IID uniform mixture of three SUMO templates; no road-safety or OOD guarantee'})
    print(path)


def worker(seed, checkpoint=None):
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    cfg=Config(**protocol['config'])
    checkpoint=Path(checkpoint) if checkpoint else ROOT/'artifacts/v2_final'/f'seed{seed}'/'policy.pt'
    model,_=load_policy(checkpoint)
    out=OUT/f'seed{seed}';out.mkdir(exist_ok=True)
    if (out/'episodes.csv').exists():
        raise RuntimeError('Existing results will not be overwritten; choose a new --out directory.')
    import shutil
    for file in ('policy.pt','training.csv','config.json','bc.json'):
        shutil.copy2(checkpoint.parent/file,out/file)
    rows=[]
    sets=[('test',protocol['test_scenes'],METHODS),
          ('shift',protocol['shift_scenes'],METHODS),
          ('certification',protocol['certification_scenes'],['paired']+(['traditional'] if seed==31 else []))]
    start=time.perf_counter()
    for split,scenes,methods in sets:
        for method in methods:
            with SumoEnv(cfg,method) as env:
                for k,scene in enumerate(scenes):
                    obs=env.reset(**scene)
                    while True:
                        a,_,_=model.act(obs,True)
                        obs,_,t,u,metrics=env.step(a)
                        if t or u:break
                    row={'training_seed':seed,'split':split,**metrics}
                    rows.append(row)
                    if k==0 and split!='certification':
                        write_json(out/f'trace_{split}_{method}.json',env.trace)
                    if (k+1)%25==0:
                        print(json.dumps({'seed':seed,'split':split,'method':method,'episodes':k+1}),flush=True)
            # Persist after every controller so interruption never loses completed groups.
            with (out/'episodes.csv').open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
            print(json.dumps({'seed':seed,'split':split,'method':method,'finished':len(scenes)}),flush=True)
    write_json(out/'manifest.json',{'protocol_sha256':hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest(),
                                  'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                                  'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src'/'coopgate').glob('*.py')},
                                  'episodes':len(rows),'wall_seconds':time.perf_counter()-start})
    write_json(out/'summary.json',{split:summarize([r for r in rows if r['split']==split]) for split,_,_ in sets})
    cert=[]
    for method in sets[-1][2]:
        group=[r for r in rows if r['split']=='certification' and r['method']==method]
        for endpoint in ('collision','not_success'):
            failures=sum(r['collision'] if endpoint=='collision' else not r['success'] for r in group)
            cert.append({'training_seed':seed,'method':method,'endpoint':endpoint,
                         **certify(failures,len(group),family_size=protocol['family_size'])})
    write_json(out/'certification.json',cert)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--seed',type=int)
    p.add_argument('--out',default='runs/v2_reproduction')
    p.add_argument('--checkpoint')
    a=p.parse_args()
    OUT=ROOT/a.out
    OUT.mkdir(parents=True,exist_ok=True)
    if a.prepare:prepare()
    else:
        if a.seed not in SEEDS:p.error('--seed must be 31, 47, or 59')
        if not (OUT/'protocol.json').exists():
            import shutil
            shutil.copy2(ROOT/'artifacts/v2_final/protocol.json',OUT/'protocol.json')
        worker(a.seed,a.checkpoint)
