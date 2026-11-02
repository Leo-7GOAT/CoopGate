"""Independent grouped-imitation audit; this is NOT a traffic/risk certificate."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from coopgate.fast_gate_model import FastGateModel,online_features
from coopgate.merge_v4 import MergeConfig

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/mappo_ctde_20261101'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    torch.set_num_threads(1)
    out=BASE/'fast_gate_training_20261102';data_dir=BASE/'gate_distillation_data_20261102'
    plan=json.loads((out/'plan.json').read_text());complete=json.loads((out/'complete.json').read_text())
    report=json.loads((out/'report.json').read_text());meta=json.loads((data_dir/'metadata.json').read_text())
    if sha(data_dir/'data.npz')!=plan['teacher_data_sha256']:raise ValueError('Data changed')
    for name,digest in plan['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen training source changed')
    for name,digest in complete['model_sha256'].items():
        if sha(out/name)!=digest:raise ValueError('Model changed')
    data=np.load(data_dir/'data.npz');features=data['features'];labels=data['labels'];groups=data['initial_group']
    rows=json.loads((data_dir/'row_identity.json').read_text())
    lookup={(tuple(r['case']),r['step']):i for i,r in enumerate(rows)}
    if len(lookup)!=len(rows):raise ValueError('Duplicate teacher row')
    checked=set();config=MergeConfig()
    for name,digest in meta['source_episodes_sha256'].items():
        path=BASE/'development'/name/'episodes.jsonl'
        if sha(path)!=digest:raise ValueError('Teacher episode source changed')
        for line in path.read_text().splitlines():
            r=json.loads(line)
            if r['row']['rule']!='parameter_scenario_confidence':continue
            state=np.array(r['row']['initial']);previous=np.zeros(3)
            for step,item in enumerate(r['trace']):
                i=lookup[(tuple(r['key']),step)]
                f=online_features(state,previous,step,np.array(item['actor_proposal']),np.array(item['baseline_command']),
                    np.array(item['actuator_memory_lower']),np.array(item['actuator_memory_upper']),
                    np.array(item['reconstructed_actuator']),np.array(item['actuator_bounds']),config)
                if f.tobytes()!=features[i].tobytes():raise ValueError('Runtime feature bytes differ')
                checked.add(i);state=np.array(item['x']);previous=np.array(item['command'])
    if len(checked)!=len(rows):raise ValueError('Missing feature rows')
    recomputed=[]
    for old in report['folds']:
        group=old['held_out_initial'];train=groups!=group;test=groups==group
        saved=torch.load(out/f'fold_{group}.pt',map_location='cpu',weights_only=True)
        model=FastGateModel();model.load_state_dict(saved['model']);model.eval()
        x=torch.from_numpy(features[train]);y=torch.from_numpy(labels[test])
        torch.testing.assert_close(model.mean,x.mean(0),rtol=0,atol=0)
        torch.testing.assert_close(model.scale,x.std(0).clamp(min=.05),rtol=0,atol=0)
        np.testing.assert_array_equal(model.classes.numpy(),np.bincount(labels[train],minlength=29)>0)
        with torch.no_grad():score,pred=model(torch.from_numpy(features[test])).softmax(-1).max(-1)
        item=dict(held_out_initial=int(group),thresholds={})
        for threshold in plan['thresholds']:
            selected=(pred!=0)&(score>=threshold);correct=(pred==y)&selected
            expected=old['thresholds'][str(threshold)]
            if int(selected.sum())!=expected['selected']:raise ValueError('Fold selected count differs')
            precision=float(correct.sum()/selected.sum()) if selected.any() else None
            if precision!=expected['teacher_class_precision']:raise ValueError('Fold precision differs')
            item['thresholds'][str(threshold)]=dict(selected=int(selected.sum()),correct=int(correct.sum()))
        recomputed.append(item)
    proof=dict(feature_rows_byte_matched=len(checked),models_sha_verified=len(complete['model_sha256']),
        fold_normalization_train_only_verified=True,fold_results=recomputed,
        independent_initial_groups=len(set(groups)),supervised_not_PPO=True,
        passed_positive_generalization_prerequisite=False,runtime_promotion=False,
        actual_sumo_executions_added=0,risk_certificate=False,
        native_training_exit_receipt_available=False,
        meaning='Completed artifacts independently verified; original native training exit was not recovered. No retrospectively fabricated receipt.',
        source_sha256=sha(Path(__file__)))
    with (out/'independent_audit.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(proof,f,indent=2);f.write('\n')
    print(json.dumps(proof),flush=True)


if __name__=='__main__':audit()
