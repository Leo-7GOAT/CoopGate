"""Frozen, paired SUMO study for single-learner gating before MARL expansion."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from coopgate.disturbance_v9 import DisturbedMergeEnv
from coopgate.marl_v12 import MARLGate,load_policy
from coopgate.single_v12 import SingleRolloutGate
from coopgate.confidence_v12 import EpisodeConfidenceGate
from experiment_single_v12 import run_episode

ROOT=Path(__file__).resolve().parents[1]


def run(protocol_path,out):
    protocol_path=Path(protocol_path);protocol=json.loads(protocol_path.read_text());out=Path(out);out.mkdir(parents=True,exist_ok=False)
    for path,digest in protocol['source_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    policies={};bc={}
    for seed in protocol['training_seeds']:
        policies[seed]=load_policy(ROOT/protocol['policies'][str(seed)])[0]
        bc[seed]=load_policy(ROOT/protocol['bc_policies'][str(seed)])[0]
    calibration=json.loads((ROOT/protocol['calibration']).read_text())
    factories={'rule':lambda x:MARLGate(None,mode='constant'),'rollout':lambda x:SingleRolloutGate()}
    for seed in policies:
        factories[f'rl_{seed}']=lambda x,s=seed:MARLGate(policies[s],mode='single')
        factories[f'bc_{seed}']=lambda x,s=seed:MARLGate(bc[s],mode='single')
        factories[f'gate_{seed}']=lambda x,s=seed:EpisodeConfidenceGate(policies[s],x,calibration[str(s)])
    assert set(factories)==set(protocol['methods'])
    warm=np.array([[-80.,12.],[-85.,13.],[-120.,11.]])
    for factory in factories.values():factory(warm).select(warm,np.zeros(3))
    rows=[];initials=[];start=time.perf_counter()
    with DisturbedMergeEnv() as env:
        for index,item in enumerate(protocol['initials']):
            for mode_index,mode in enumerate(protocol['disturbances']):
                order=np.random.default_rng(item['seed']+1217+mode_index).permutation(protocol['methods'])
                for method in order:
                    row,trace,initial=run_episode(env,item['seed'],item['regime'],mode,factories[method],index<2)
                    rows.append({**row,'controller':str(method)})
                    if mode_index==0 and method=='rule':initials.append({'seed':item['seed'],'regime':item['regime'],'state':initial.tolist()})
                    if trace is not None:(out/f'trace_{index}_{mode}_{method}.json').write_text(json.dumps(trace,indent=2)+'\n')
            if (index+1)%8==0:
                (out/'episodes.json').write_text(json.dumps(rows,indent=2)+'\n')
                print(json.dumps({'independent_initials':index+1,'sumo_runs':len(rows),'elapsed_s':time.perf_counter()-start}),flush=True)
    (out/'episodes.json').write_text(json.dumps(rows,indent=2)+'\n')
    (out/'initials.json').write_text(json.dumps(initials,indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps({'protocol_sha256':hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        'executions':len(rows),'single_worker':True,'elapsed_s':time.perf_counter()-start},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',default='artifacts/v12/single/protocol.json');p.add_argument('--out',required=True)
    a=p.parse_args();run(a.protocol,a.out)
