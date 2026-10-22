"""Execute the frozen seven-method SUMO study without adaptive selection."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from coopgate.disturbance_v9 import DisturbedMergeEnv
from coopgate.ego_v13 import EgoRolloutGate,EgoPolicyGate
from coopgate.policy_iteration_v14 import load_policy
from experiment_ego_v13 import run_episode

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'artifacts/v14/single'


def factories(protocol=None):
    torch.set_num_threads(1)
    p=protocol or json.loads((D/'protocol.json').read_text())
    fs={'rollout':lambda x:EgoRolloutGate()}
    for seed in p['training_seeds']:
        for mode,file in (('learned','policy.pt'),('last','last.pt')):
            model,_=load_policy(D/f'seed_{seed}'/file)
            fs[f'{mode}_{seed}']=lambda x,m=model:EgoPolicyGate(m)
    return fs


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    protocol_path=D/'protocol.json';p=json.loads(protocol_path.read_text())
    for name,digest in p['input_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    fs=factories(p);warm=np.array([[-80.,12.],[-85.,13.],[-120.,11.]])
    for f in fs.values():f(warm).select(warm,np.zeros(3))
    rows=[];initials=[];started=time.perf_counter()
    with DisturbedMergeEnv() as env:
        for index,item in enumerate(p['initials']):
            for mode_index,mode in enumerate(p['disturbances']):
                order=np.random.default_rng(item['seed']+mode_index+1414).permutation(p['methods'])
                for method in order:
                    row,trace,initial=run_episode(env,item['seed'],item['regime'],mode,fs[method],index in p['audit_initial_indices'])
                    rows.append({**row,'controller':str(method)})
                    if mode_index==0 and method=='rollout':initials.append({**item,'state':initial.tolist()})
                    if trace is not None:(out/f'trace_{index}_{mode}_{method}.json').write_text(json.dumps(trace,indent=2)+'\n')
            if (index+1)%8==0:
                (out/'episodes.json').write_text(json.dumps(rows,indent=2)+'\n')
                print(json.dumps(dict(initials=index+1,sumo_runs=len(rows),elapsed_s=time.perf_counter()-started)),flush=True)
    (out/'episodes.json').write_text(json.dumps(rows,indent=2)+'\n')
    (out/'initials.json').write_text(json.dumps(initials,indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps(dict(protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        sumo_runs=len(rows),single_worker=True,elapsed_s=time.perf_counter()-started),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='runs/v14_single_study');a=p.parse_args();run(a.out)
