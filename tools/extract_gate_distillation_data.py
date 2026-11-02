"""Offline teacher data for a prospective FAST gate; MAPPO actors unchanged.

Labels are supervised arbitration labels,not PPO transitions or new trials.
Inputs contain only online-observable state,proposals,baseline and observer
quantities. Source episodes remain grouped by independent initial.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from coopgate.ego_v13 import ego_observations
from coopgate.fast_v8 import parameters
from coopgate.mappo_ctde import local_kinematic_features
from coopgate.marl_v12 import central_state
from coopgate.merge_v4 import MergeConfig
from coopgate.recovery_joint_gate_v35 import compositions

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/mappo_ctde_20261101/development'
WEIGHTS=(1.,.75,.5,.25)


def extract():
    torch.set_num_threads(1)
    out=ROOT/'artifacts/mappo_ctde_20261101/gate_distillation_data_20261102';out.mkdir(exist_ok=False)
    sources=[BASE/f'parameter_scenario_grid_20261102_shard{i}' for i in range(4)]
    sources.append(BASE/'parameter_scenario_regression_20261102')
    xdata=[];labels=[];groups=[];identities=[];source_hashes={};strict=0;config=MergeConfig();packed=parameters(config)
    for directory in sources:
        if not json.loads((directory/'verification.json').read_text())['all_match']:raise ValueError('Reviewed complete teachers required')
        data=(directory/'episodes.jsonl').read_bytes()
        complete=json.loads((directory/'collection_complete.json').read_text())
        digest=hashlib.sha256(data).hexdigest()
        if digest!=complete['episodes_sha256']:raise ValueError('Teacher data changed')
        source_hashes[directory.name]=digest
        for line in data.splitlines():
            record=json.loads(line);row=record['row']
            if row['rule']!='parameter_scenario_confidence':continue
            state=np.array(row['initial']);previous=np.zeros(3)
            for step,item in enumerate(record['trace']):
                proposal=np.array(item['actor_proposal']);base=np.array(item['baseline_command'])
                lo=np.array(item['actuator_memory_lower']);hi=np.array(item['actuator_memory_upper'])
                bounds=np.array(item['actuator_bounds']);normalized=bounds.copy()
                normalized[:,2:4]*=1000.;normalized[:,6]*=10.
                obs=torch.as_tensor(ego_observations(state,previous,step,config))
                derived=local_kinematic_features(obs)[:,18:].numpy().ravel()
                feature=np.concatenate((central_state(state,previous,step,config),proposal/5,base/5,
                    lo/5,hi/5,np.array(item['reconstructed_actuator'])/5,normalized.ravel(),derived))
                label=0;decoded=base
                if item['actual_learning_used']:
                    code=item['composition'];weight=item['repair_weight'];label=1+(code-1)*4+WEIGHTS.index(weight)
                    candidates,_=compositions(state,previous,proposal,base,lo,hi,bounds,packed)
                    matches=[command for c,w,command,_ in candidates if c==code and w==weight]
                    if len(matches)!=1:raise ValueError('Teacher class has no unique online-decodable candidate')
                    decoded=matches[0]
                np.testing.assert_allclose(decoded,item['command'],atol=1e-9,rtol=0)
                strict+=int(np.array_equal(decoded,item['command']))
                xdata.append(feature);labels.append(label);groups.append([row['seed'],row['scenario']])
                identities.append(dict(case=record['key'],step=step))
                previous=np.array(item['command']);state=np.array(item['x'])
    features=np.asarray(xdata,np.float32);y=np.asarray(labels,np.int64)
    unique=sorted({(seed,scene) for seed,scene in groups});group=np.array([unique.index((seed,scene)) for seed,scene in groups],np.int64)
    if features.shape[1]!=73 or not np.isfinite(features).all():raise ValueError('Online-only73-feature schema violated')
    np.savez_compressed(out/'data.npz',features=features,labels=y,initial_group=group)
    metadata=dict(rows=len(y),features=73,independent_initial_groups=len(unique),groups=unique,
        label_counts=np.bincount(y,minlength=29).tolist(),strict_decodes=strict,
        all_decodes_within_original_adoption_tolerance=True,source_episodes_sha256=source_hashes,
        data_sha256=hashlib.sha256((out/'data.npz').read_bytes()).hexdigest(),
        feature_contract='global13;actor/baseline commands6;memory interval6;midpoint memory3;declared bounds21;deterministic local kinematic transforms24. NO true plant,seed,future,teacher score/forecast/outcome inputs.',
        supervision='Teacher composition/repair label0..28;not PPO or new simulator runs',
        split_requirement='Group by independent initial across ALL models/modes/plants. Never random-split frames.',
        actual_student_trained=False,extra_sumo_runs=0,new_risk_certificate=False)
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (out/'row_identity.json').write_text(json.dumps(identities)+'\n')
    print(json.dumps({k:metadata[k] for k in ('rows','features','independent_initial_groups','strict_decodes','actual_student_trained')}),flush=True)


if __name__=='__main__':extract()
