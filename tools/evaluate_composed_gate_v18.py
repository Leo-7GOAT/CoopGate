"""Actual SUMO mechanism study of isolated vs complete nominal admission.

Neither nominal gate is statistically certified. All isolated counterfactuals
and the actually composed controller execute as different complete profiles.
No column-wise recombination substitutes for a closed-loop run.
"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from coopgate.composed_gate_v18 import ComposedProfile, nominal_independent_decision
from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.profile_prediction_v16 import predict_profile
from coopgate.disturbance_v9 import DisturbedMergeEnv
from develop_v12 import initial
from experiment_ego_v13 import run_episode
from verify_development_v14 import trace_audit

ROOT = Path(__file__).resolve().parents[1]


def run(path, out, n=64, seed_base=61400001):
    if n < 4 or n % 2:
        raise ValueError('Even nonempty regime split required')
    torch.set_num_threads(1)
    path, out = Path(path), Path(out)
    model, meta = load_coupled(path)
    if meta['updates'] != meta['target_updates']:
        raise ValueError('Incomplete joint training')
    out.mkdir(parents=True, exist_ok=False)
    modes = ['clean', 'uniform', 'bias', 'pulse']
    masks = {'baseline': np.zeros(3, bool), 'full': np.ones(3, bool)}
    masks.update({f'isolated_{i}': np.arange(3) == i for i in range(3)})
    methods = list(masks)+['independent_nominal', 'joint_nominal']
    audit_indices = [0, 1, n//2, n//2+1]
    files = list((ROOT/'src/coopgate').glob('*.py'))+[Path(__file__), ROOT/'tools/experiment_ego_v13.py',
            ROOT/'tools/develop_v12.py', ROOT/'tools/verify_development_v14.py']
    protocol = dict(purpose='Nominal gate composition DEVELOPMENT ablation, not calibrated risk validation',
        model_path=str(path).replace('\\', '/'), model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        n_initials=n, seed_base=seed_base, methods=methods, modes=modes, audit_initial_indices=audit_indices,
        expected_sumo_runs=n*len(modes)*len(methods), minimum_gain=.05,
        independent='Each vehicle checks its own cost/time under an isolated learned-component profile, assuming other components use the baseline',
        joint='Check all cost/time components and mean cost gain under the full joint learned profile',
        statistical_risk_claim=False, uses_realized_disturbance_tape_for_decisions=False,
        source_sha256={str(f.resolve().relative_to(ROOT)).replace('\\', '/'):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    for source in protocol['source_sha256']:
        target=out/'source'/source;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/source,target)
    rows, decisions = [], []
    started = time.perf_counter()
    clean_error = []
    with DisturbedMergeEnv() as env:
        for i in range(n):
            seed = seed_base+i
            regime = 'nominal' if i<n//2 else 'dense'
            env.reset_disturbed(seed, regime, 'clean', .5)
            x = env.x.copy()
            np.testing.assert_allclose(x, initial(seed, regime), atol=1e-8, rtol=0)
            predictions = {}
            prediction_start = time.perf_counter()
            for name, mask in masks.items():
                predictions[name], _ = predict_profile(x, lambda x, m=mask:ComposedProfile(model, m))
            isolated = np.array([predictions[f'isolated_{j}'][j]-predictions['baseline'][j] for j in range(3)])
            full_difference = predictions['full']-predictions['baseline']
            selected = dict(masks)
            selected['independent_nominal'] = nominal_independent_decision(isolated, .05)
            admit = bool(np.all(full_difference<=1e-9) and full_difference[:, 0].mean()<=-.05)
            selected['joint_nominal'] = np.full(3, admit)
            decisions.append(dict(seed=seed, regime=regime, initial=x.tolist(),
                predictions={k:v.tolist() for k,v in predictions.items()},
                isolated_own_difference=isolated.tolist(), full_difference=full_difference.tolist(),
                selected_masks={k:v.tolist() for k,v in selected.items()},
                prediction_ms=(time.perf_counter()-prediction_start)*1000.))
            for mi, mode in enumerate(modes):
                for name in np.random.default_rng(seed+mi+1818).permutation(methods):
                    mask=selected[name]
                    result, trace, actual_initial = run_episode(env, seed, regime, mode,
                        lambda x,m=mask:ComposedProfile(model,m), i in audit_indices)
                    np.testing.assert_allclose(actual_initial,x,atol=1e-8,rtol=0)
                    costs=np.array(result['per_vehicle_finish'])+.05*np.array(result['per_vehicle_variation'])
                    result.update(controller=str(name),initial=x.tolist(),per_vehicle_cost=costs.tolist(),
                        team_cost=float(costs.mean()),component_learning_mask=mask.tolist())
                    rows.append(result)
                    if trace is not None:
                        trace_audit(trace,result)
                        (out/f'trace_{i}_{mode}_{name}.json').write_text(json.dumps(trace,indent=2)+'\n')
                    if mode=='clean' and name in predictions:
                        actual=np.column_stack((costs,result['per_vehicle_finish']))
                        clean_error.append(float(np.abs(actual-predictions[name]).max()))
            (out/'episodes.json').write_text(json.dumps(rows,indent=2)+'\n')
            (out/'decisions.json').write_text(json.dumps(decisions,indent=2)+'\n')
            if (i+1)%4==0:
                print(json.dumps(dict(initials=i+1,sumo_runs=len(rows),elapsed_s=time.perf_counter()-started)),flush=True)
    if len(rows)!=protocol['expected_sumo_runs']:
        raise RuntimeError('Incomplete comparison')
    lookup={(r['seed'],r['disturbance'],r['controller']):r for r in rows}
    summary={}
    for name in methods:
        data=[r for r in rows if r['controller']==name]
        values=np.array([[np.column_stack((lookup[seed,mode,name]['per_vehicle_cost'],lookup[seed,mode,name]['per_vehicle_finish']))
            -np.column_stack((lookup[seed,mode,'baseline']['per_vehicle_cost'],lookup[seed,mode,'baseline']['per_vehicle_finish']))
            for mode in modes] for seed in range(seed_base,seed_base+n)])
        selected=np.array([d['selected_masks'][name] for d in decisions]);accepted=selected.any(1)
        harmed=np.any(values>1e-9,axis=(1,2,3))
        # Refusal is a complete baseline profile and must exactly reproduce its paired costs.
        np.testing.assert_allclose(values[~accepted],0,atol=1e-9,rtol=0)
        summary[name]=dict(runs=len(data),completed=sum(r['completed'] for r in data),
            collisions=sum(r['collision'] for r in data),contract_violations=sum(r['contract_violations'] for r in data),
            aborted=sum(r['aborted'] for r in data),admitted_initials=int(accepted.sum()),
            harmful_initials=int(harmed.sum()),conditional_empirical_harm=float(harmed[accepted].mean()) if accepted.any() else None,
            team_cost=float(np.mean([r['team_cost'] for r in data])),
            per_vehicle_cost_difference=values[:,:,:,0].mean((0,1)).tolist(),
            nominal_prediction_count=sum(name in d['predictions'] for d in decisions))
    result=dict(purpose=protocol['purpose'],actual_sumo_runs=len(rows),results=summary,
        nominal_predictions=len(clean_error),maximum_clean_forecast_error=max(clean_error),elapsed_s=time.perf_counter()-started)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('path');p.add_argument('--out',required=True)
    p.add_argument('--n',type=int,default=64);p.add_argument('--seed-base',type=int,default=61400001)
    a=p.parse_args();run(a.path,a.out,a.n,a.seed_base)
