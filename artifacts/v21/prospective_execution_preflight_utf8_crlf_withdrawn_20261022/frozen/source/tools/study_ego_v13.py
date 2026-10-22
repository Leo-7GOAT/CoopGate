"""Paired actual SUMO experiments after the V13 freeze."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from coopgate.disturbance_v9 import DisturbedMergeEnv
from coopgate.marl_v12 import MARLGate
from coopgate.ego_v13 import EgoRolloutGate,EgoPolicyGate,load_ego
from coopgate.learned_rollout_v13 import LearnedRolloutGate
from coopgate.ego_confidence_v13 import EgoConfidenceGate
from experiment_ego_v13 import run_episode
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/v13/ego'


def factories():
    p=json.loads((D/'protocol.json').read_text());cal=json.loads((D/'calibration/calibration.json').read_text())
    fs={'rule':lambda x:MARLGate(None,mode='constant'),'rollout':lambda x:EgoRolloutGate()}
    for s in p['training_seeds']:
        path=D/f'seed_{s}/policy.pt';model,_=load_ego(path)
        fs[f'raw_{s}']=lambda x,m=model:EgoPolicyGate(m)
        fs[f'untrained_{s}']=lambda x,s=s:LearnedRolloutGate(D/f'seed_{s}/initial.pt')
        fs[f'augmented_{s}']=lambda x,s=s:LearnedRolloutGate(D/f'seed_{s}/policy.pt')
        fs[f'gate_{s}']=lambda x,s=s:EgoConfidenceGate(D/f'seed_{s}/policy.pt',x,cal[str(s)])
    return fs


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);protocol_path=D/'protocol.json';p=json.loads(protocol_path.read_text())
    for path,digest in p['source_sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    fs=factories();warm=np.array([[-80.,12.],[-85.,13.],[-120.,11.]])
    for factory in fs.values():factory(warm).select(warm,np.zeros(3))
    rows=[];initials=[];start=time.perf_counter()
    with DisturbedMergeEnv() as env:
        for index,item in enumerate(p['initials']):
            for mode_index,mode in enumerate(p['disturbances']):
                order=np.random.default_rng(item['seed']+mode_index+1317).permutation(p['methods'])
                for method in order:
                    row,trace,initial=run_episode(env,item['seed'],item['regime'],mode,fs[method],index<2)
                    rows.append({**row,'controller':str(method)})
                    if mode_index==0 and method=='rule':initials.append({**item,'state':initial.tolist()})
                    if trace is not None:(out/f'trace_{index}_{mode}_{method}.json').write_text(json.dumps(trace,indent=2)+'\n')
            if (index+1)%8==0:
                (out/'episodes.json').write_text(json.dumps(rows,indent=2)+'\n')
                print(json.dumps({'test_initials':index+1,'sumo_runs':len(rows),'elapsed_s':time.perf_counter()-start}),flush=True)
    (out/'episodes.json').write_text(json.dumps(rows,indent=2)+'\n');(out/'initials.json').write_text(json.dumps(initials,indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps({'protocol_sha256':hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        'sumo_runs':len(rows),'single_worker':True,'elapsed_s':time.perf_counter()-start},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='runs/v13_ego_study');a=p.parse_args();run(a.out)
