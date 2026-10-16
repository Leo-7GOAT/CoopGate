"""Fixed stress-bank development forecasts, without using episode error tapes."""
import argparse,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
import torch
from numba import set_num_threads
from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.stress_prediction_v17 import predict_stress,fixed_stress_bank

ROOT=Path(__file__).resolve().parents[1]


def run(study,out,threads=3):
    study,out=Path(study),Path(out);torch.set_num_threads(1);set_num_threads(threads)
    p=json.loads((study/'protocol.json').read_text());rows=json.loads((study/'episodes.json').read_text())
    if len(rows)!=p['expected_runs'] or len(p['models'])!=1:raise ValueError('One complete fixed-model development study required')
    name=next(iter(p['models']));record=p['models'][name];path=ROOT/record['path']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Policy changed')
    model,_=load_coupled(path);out.mkdir(parents=True,exist_ok=False)
    names,tapes=fixed_stress_bank()
    protocol=dict(purpose='Finite stress-bank DEVELOPMENT prediction; no performance risk certificate',
        model=name,model_sha256=record['sha256'],scenario_names=list(names),
        study_protocol_sha256=hashlib.sha256((study/'protocol.json').read_bytes()).hexdigest(),
        uses_realized_error_tape=False,bank_depends_on_episode_seed=False,exhaustive_bounded_error_claim=False,
        threads=threads,timing='Batched development throughput, not an online latency benchmark')
    files=list((ROOT/'src/coopgate').glob('*.py'))+[Path(__file__)]
    protocol['source_sha256']={str(f.resolve().relative_to(ROOT)).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');np.save(out/'fixed_bank.npy',tapes)
    for source in protocol['source_sha256']:
        target=out/'source'/source;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/source,target)
    predict_stress(np.array([[[-80.,12.],[-85.,13.],[-120.,11.]]]),model)
    records=[];baselines=[];learned=[];counts=np.zeros(3,dtype=np.int64);started=time.perf_counter()
    for first in range(0,p['n_initials'],16):
        indices=range(first,min(first+16,p['n_initials']));seeds=[p['seed_base']+i for i in indices]
        initials=np.array([next(r['initial'] for r in rows if r['seed']==seed) for seed in seeds])
        upper,details=predict_stress(initials,model);baselines.append(details['baseline']);learned.append(details['learned'])
        counts+=np.array([details['baseline_transitions'],details['learned_transitions'],details['baseline_internal_transitions']])
        for k,seed in enumerate(seeds):records.append(dict(seed=seed,controller=name,predicted_difference=upper[k].tolist()))
        print(json.dumps(dict(initials=len(records),stress_trajectories=2*len(records)*len(names),elapsed_s=time.perf_counter()-started)),flush=True)
    base=np.concatenate(baselines);learn=np.concatenate(learned)
    np.savez_compressed(out/'scenario_values.npz',baseline=base,learned=learn)
    (out/'difference_predictions.json').write_text(json.dumps(records,indent=2)+'\n')
    clean_error=[]
    for i in range(p['n_initials']):
        seed=p['seed_base']+i
        for method,values in [('team_planner',base),(name,learn)]:
            r=next(r for r in rows if r['seed']==seed and r['disturbance']=='clean' and r['controller']==method)
            clean_error.append(float(np.abs(values[i,0]-np.column_stack((r['per_vehicle_cost'],r['per_vehicle_finish']))).max()))
    summary=dict(initials=len(records),stress_trajectories=2*len(records)*len(names),max_clean_discrepancy=max(clean_error),
        baseline_transitions=int(counts[0]),learned_transitions=int(counts[1]),baseline_internal_transitions=int(counts[2]),
        elapsed_s=time.perf_counter()-started,backend='model forecasts, not additional SUMO runs')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study');p.add_argument('--out',required=True);p.add_argument('--threads',type=int,default=3)
    a=p.parse_args();run(a.study,a.out,a.threads)
