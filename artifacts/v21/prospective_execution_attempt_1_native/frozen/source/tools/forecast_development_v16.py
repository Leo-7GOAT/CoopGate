"""Check model-only predictions against saved clean runs; no gate selection."""
import argparse,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
import torch
from coopgate.profile_prediction_v16 import predict_profile
from validate_cooperative_v16 import factories

ROOT=Path(__file__).resolve().parents[1]


def run(study,out):
    torch.set_num_threads(1);study,out=Path(study),Path(out)
    p=json.loads((study/'protocol.json').read_text());rows=json.loads((study/'episodes.json').read_text())
    if len(rows)!=p['expected_runs']:raise ValueError('Require the entire completed development study')
    for r in p['models'].values():
        if hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()!=r['sha256']:raise ValueError('Model changed since study')
    fs,_=factories([ROOT/r['path'] for r in p['models'].values()]);out.mkdir(parents=True,exist_ok=False)
    selected=['team_planner']+list(p['models']);records=[]
    protocol=dict(purpose='Development forecast reconstruction only; no risk certificate',
        study_protocol_sha256=hashlib.sha256((study/'protocol.json').read_bytes()).hexdigest(),
        methods=selected,metrics=['cost','arrival_time'],future_disturbances_observed=False)
    files=list((ROOT/'src/coopgate').glob('*.py'))+[Path(__file__),ROOT/'tools/validate_cooperative_v16.py']
    protocol['source_sha256']={str(f.resolve().relative_to(ROOT)).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for name in protocol['source_sha256']:
        target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    source=ROOT/'src/coopgate/profile_prediction_v16.py';(out/'prediction_source.py').write_bytes(source.read_bytes())
    for i in range(p['n_initials']):
        seed=p['seed_base']+i
        for name in selected:
            clean=next(r for r in rows if r['seed']==seed and r['disturbance']=='clean' and r['controller']==name)
            start=time.perf_counter();prediction,details=predict_profile(clean['initial'],fs[name]);elapsed=time.perf_counter()-start
            actual=np.column_stack((clean['per_vehicle_cost'],clean['per_vehicle_finish']))
            records.append(dict(seed=seed,controller=name,predicted=prediction.tolist(),actual_clean=actual.tolist(),
                max_absolute_clean_discrepancy=float(np.abs(prediction-actual).max()),milliseconds=elapsed*1000,**details))
        if (i+1)%16==0:print(json.dumps(dict(initials=i+1,forecasts=len(records))),flush=True)
    (out/'predictions.json').write_text(json.dumps(records,indent=2)+'\n')
    summary=dict(forecasts=len(records),max_clean_discrepancy=max(r['max_absolute_clean_discrepancy'] for r in records),
        nonmatching_clean_forecasts=sum(r['max_absolute_clean_discrepancy']>1e-8 for r in records),
        forecast_transitions=sum(r['forecast_transitions'] for r in records),
        baseline_internal_planning_transitions=sum(r['baseline_planning_internal_transitions'] for r in records),
        p95_ms_by_method={name:float(np.percentile([r['milliseconds'] for r in records if r['controller']==name],95)) for name in selected})
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study');p.add_argument('--out',required=True);a=p.parse_args();run(a.study,a.out)
