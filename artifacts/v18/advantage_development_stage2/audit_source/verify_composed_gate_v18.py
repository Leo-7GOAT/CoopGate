"""Reconstruct nominal gate choices, real composed commands and paired runs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.composed_gate_v18 import ComposedProfile, nominal_independent_decision
from coopgate.profile_prediction_v16 import predict_profile
from coopgate.disturbance_v9 import DisturbedMergeEnv
from experiment_ego_v13 import run_episode
from verify_development_v14 import trace_audit
from verify_single_v12 import equivalent

ROOT=Path(__file__).resolve().parents[1]


def verify(directory, replay=False):
    torch.set_num_threads(1)
    directory=Path(directory)
    p=json.loads((directory/'protocol.json').read_text())
    for source,digest in p['source_sha256'].items():
        assert hashlib.sha256((directory/'source'/source).read_bytes()).hexdigest()==digest,source
        assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==digest,source
    path=ROOT/p['model_path']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==p['model_sha256']
    model,_=load_coupled(path)
    rows=json.loads((directory/'episodes.json').read_text())
    ds=json.loads((directory/'decisions.json').read_text())
    lookup={(r['seed'],r['disturbance'],r['controller']):r for r in rows}
    seeds=range(p['seed_base'],p['seed_base']+p['n_initials'])
    assert len(rows)==len(lookup)==p['expected_sumo_runs']
    assert set(lookup)=={(s,m,c) for s in seeds for m in p['modes'] for c in p['methods']}
    assert [d['seed'] for d in ds]==list(seeds)
    controls=predictions=traces=0
    for i,d in enumerate(ds):
        seed=d['seed'];base=np.array(d['predictions']['baseline'])
        own=np.array([np.array(d['predictions'][f'isolated_{j}'])[j]-base[j] for j in range(3)])
        np.testing.assert_array_equal(own,d['isolated_own_difference'])
        np.testing.assert_array_equal(nominal_independent_decision(own,p['minimum_gain']),d['selected_masks']['independent_nominal'])
        difference=np.array(d['predictions']['full'])-base
        np.testing.assert_array_equal(difference,d['full_difference'])
        admit=np.all(difference<=1e-9) and difference[:,0].mean()<=-p['minimum_gain']
        assert d['selected_masks']['joint_nominal']==[bool(admit)]*3
        for name in p['methods']:
            mask=np.array(d['selected_masks'][name],bool)
            for mode in p['modes']:
                row=lookup[seed,mode,name]
                np.testing.assert_allclose(row['initial'],d['initial'],atol=1e-8,rtol=0)
                assert row['component_learning_mask']==mask.tolist()
                if not mask.any():
                    for key in ('per_vehicle_finish','per_vehicle_variation','completed','aborted','collision'):
                        equivalent(row[key],lookup[seed,mode,'baseline'][key])
                if mask.all():
                    for key in ('per_vehicle_finish','per_vehicle_variation','completed','aborted','collision'):
                        equivalent(row[key],lookup[seed,mode,'full'][key])
                if mode=='clean' and name in d['predictions']:
                    np.testing.assert_allclose(d['predictions'][name],np.column_stack((row['per_vehicle_cost'],row['per_vehicle_finish'])),atol=1e-8,rtol=0)
                if i in p['audit_initial_indices']:
                    trace=json.loads((directory/f'trace_{i}_{mode}_{name}.json').read_text());trace_audit(trace,row)
                    controller=ComposedProfile(model,mask);x=np.array(d['initial']);previous=np.zeros(3)
                    for item in trace:
                        command,_=controller.select(x,previous)
                        np.testing.assert_allclose(command,item['command'],atol=1e-9,rtol=0)
                        x=np.array(item['x']);previous=np.array(item['command']);controls+=1
                    traces+=1
            if i in p['audit_initial_indices'] and name in d['predictions']:
                prediction,_=predict_profile(d['initial'],lambda x,m=mask:ComposedProfile(model,m))
                np.testing.assert_allclose(prediction,d['predictions'][name],atol=1e-8,rtol=0);predictions+=1
    replays=0
    if replay:
        with DisturbedMergeEnv() as env:
            for i in (0,p['n_initials']//2):
                d=ds[i]
                for mode in p['modes']:
                    for name in p['methods']:
                        mask=np.array(d['selected_masks'][name],bool)
                        actual,trace,_=run_episode(env,d['seed'],d['regime'],mode,lambda x,m=mask:ComposedProfile(model,m),True)
                        row=lookup[d['seed'],mode,name]
                        for key in ('per_vehicle_finish','per_vehicle_variation','completed','collision','aborted','contract_violations'):
                            equivalent(actual[key],row[key])
                        saved=json.loads((directory/f'trace_{i}_{mode}_{name}.json').read_text())
                        np.testing.assert_allclose([r['command'] for r in trace],[r['command'] for r in saved],atol=1e-9,rtol=0);replays+=1
    result=dict(actual_sumo_runs=len(rows),decision_initials=len(ds),audited_traces=traces,
        reconstructed_controls=controls,recomputed_forecasts=predictions,additional_sumo_replays=replays,
        episodes_sha256=hashlib.sha256((directory/'episodes.json').read_bytes()).hexdigest(),
        statistical_admission_certified=False)
    print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--replay',action='store_true');p.add_argument('--out')
    a=p.parse_args();r=verify(a.directory,a.replay)
    if a.out:Path(a.out).write_text(json.dumps(r,indent=2)+'\n')
