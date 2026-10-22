"""Explore gate opportunities on DEVELOPMENT data; issue no risk certificate.

Thresholds inspected here require an independent frozen risk-validation study.
The nominal forecast never receives the realized disturbance tape.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from coopgate.joint_admission_v15 import calibrate_joint


def run(study,forecast,out,model_name=None):
    study,forecast,out=Path(study),Path(forecast),Path(out)
    p=json.loads((study/'protocol.json').read_text());rows=json.loads((study/'episodes.json').read_text())
    direct=(forecast/'difference_predictions.json').exists()
    predictions=json.loads((forecast/('difference_predictions.json' if direct else 'predictions.json')).read_text())
    forecast_protocol=json.loads((forecast/'protocol.json').read_text())
    if forecast_protocol['study_protocol_sha256']!=hashlib.sha256((study/'protocol.json').read_bytes()).hexdigest():
        raise ValueError('Forecasts belong to a different study')
    if len(rows)!=p['expected_runs']:raise ValueError('A complete candidate study is required')
    if model_name is None and len(p['models'])!=1:raise ValueError('Specify the frozen candidate explicitly')
    name=model_name or next(iter(p['models']))
    if name not in p['models']:raise ValueError('Unknown candidate model')
    if direct and forecast_protocol.get('model')!=name:raise ValueError('Stress predictions belong to a different model')
    n=p['n_initials'];modes=p['modes']
    lookup={(r['seed'],r['disturbance'],r['controller']):r for r in rows}
    forecasts={(r['seed'],r['controller']):np.array(r['predicted_difference' if direct else 'predicted']) for r in predictions}
    seeds=list(range(p['seed_base'],p['seed_base']+n));actual=[];predicted=[];groups=[];regimes=[]
    for seed in seeds:
        def metrics(row):return np.column_stack((row['per_vehicle_cost'],row['per_vehicle_finish']))
        actual.append([metrics(lookup[seed,m,name])-metrics(lookup[seed,m,'team_planner']) for m in modes])
        predicted.append(forecasts[seed,name] if direct else forecasts[seed,name]-forecasts[seed,'team_planner'])
        groups.append(lookup[seed,modes[0],name]['group']);regimes.append(lookup[seed,modes[0],name]['scenario'])
    actual=np.array(actual);predicted=np.array(predicted);groups=np.array(groups);regimes=np.array(regimes)
    baseline_cost=float(np.mean([r['team_cost'] for r in rows if r['controller']=='team_planner']))
    masks={'always_learn':np.ones(n,bool)}
    for threshold in (.05,.1,.2,.5):
        masks[f'nominal_all_individuals_gain_{threshold}']=(np.all(predicted<=1e-9,axis=(1,2)) & (predicted[:,:,0].mean(1)<=-threshold))
        masks[f'nominal_team_only_gain_{threshold}']=predicted[:,:,0].mean(1)<=-threshold
    # Exploratory split only: independent formal data is still required.
    calibration=np.zeros(n,bool);test=np.zeros(n,bool);conformal=np.zeros(n,bool);bounds={}
    profile_id=(p['models'][name]['sha256'],)
    for regime in sorted(set(regimes)):
        ids=np.flatnonzero(regimes==regime);train=ids[:len(ids)//2];held=ids[len(ids)//2:]
        calibration[train]=True;test[held]=True
        bound=calibrate_joint(actual[train,:,None],predicted[train,None],profile_id,alpha=.1)
        bounds[regime]=dict(q=bound.q,calibration_initials=len(train),heldout_initials=len(held))
        for i in held:conformal[i]=bound.select(predicted[i,None],profile_id,minimum_team_gain=.05)[0] is not None
    masks['conformal_joint_split']=conformal
    result={}
    for label,mask in masks.items():
        eligible=test if label=='conformal_joint_split' else np.ones(n,bool)
        summaries={}
        for group in ['all']+sorted(set(groups)):
            subset=eligible if group=='all' else eligible&(groups==group)
            accepted=mask&subset;harmful=np.any(actual>1e-9,axis=(1,2,3))
            difference=np.where(mask[:,None,None,None],actual,0.)[subset]
            gain=-float(difference[:,:,:,0].mean()) if subset.any() else None
            summaries[group]=dict(initials=int(subset.sum()),admitted=int(accepted.sum()),
                harmful_admitted_initials=int((harmful&accepted).sum()),
                conditional_empirical_harm=float(harmful[accepted].mean()) if accepted.any() else None,
                per_vehicle_gated_cost_change=difference[:,:,:,0].mean((0,1)).tolist() if subset.any() else None,
                mean_team_gated_gain=gain,gain_percent=100*gain/baseline_cost if gain is not None else None)
        result[label]=summaries
    out.mkdir(parents=True,exist_ok=False)
    record=dict(purpose='Development hypothesis generation only; no certified risk or formal test claim',
        model=name,model_sha256=p['models'][name]['sha256'],conditional_risk_unit='Worst over four modes, three vehicles and cost/time per initial',
        nominal_forecast_source=str(forecast),conformal_split_bounds=bounds,results=result)
    (out/'diagnosis.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study');p.add_argument('forecast');p.add_argument('--out',required=True)
    p.add_argument('--model')
    a=p.parse_args();run(a.study,a.forecast,a.out,a.model)
