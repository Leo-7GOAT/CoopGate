"""Fixed-budget grouped validation and supervised final ensemble training."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from coopgate.fast_gate_model import FastGateModel

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'artifacts/mappo_ctde_20261101/gate_distillation_data_20261102'
OUT=ROOT/'artifacts/mappo_ctde_20261101/fast_gate_training_20261102'


def fit(x,y,seed):
    torch.manual_seed(seed);model=FastGateModel()
    model.mean.copy_(x.mean(0));model.scale.copy_(x.std(0).clamp(min=.05))
    counts=torch.bincount(y,minlength=29);model.classes.copy_(counts>0)
    weights=(counts[0].float()/counts.clamp(min=1)).sqrt().clamp(max=10)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    for _ in range(100):
        for ids in torch.randperm(len(y)).split(256):
            loss=torch.nn.functional.cross_entropy(model(x[ids]),y[ids],weight=weights)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
    model.eval();return model


def train():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(1)
    meta=json.loads((DATA/'metadata.json').read_text());digest=hashlib.sha256((DATA/'data.npz').read_bytes()).hexdigest()
    if digest!=meta['data_sha256']:raise ValueError('Teacher data changed')
    data=np.load(DATA/'data.npz');x=torch.from_numpy(data['features']);y=torch.from_numpy(data['labels']);groups=data['initial_group']
    plan=dict(algorithm='supervised_arbitration_distillation_not_PPO',teacher_data_sha256=digest,
        epochs=100,minibatch=256,lr=.0003,weight_decay=.0001,width=128,
        class_weight='sqrt(n_baseline/n_class),cap10;unseen classes masked',normalization='TRAIN-ONLY mean/std,floor.05',
        cross_validation='Leave ONE independent initial group out across ALL models/modes/plants',
        final_seeds=list(range(22011,22016)),thresholds=[.9,.95,.99],risk_certificate=False,
        source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in
            ('src/coopgate/fast_gate_model.py','tools/train_fast_gate.py','docs/fast_gate_20261102_protocol.md')})
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');folds=[]
    for g in sorted(set(groups)):
        train_ids=np.where(groups!=g)[0];test_ids=np.where(groups==g)[0]
        model=fit(x[train_ids],y[train_ids],21000+int(g))
        with torch.no_grad():p=model(x[test_ids]).softmax(-1);score,pred=p.max(-1)
        target=y[test_ids];positives=target!=0;item=dict(held_out_initial=int(g),train_rows=len(train_ids),test_rows=len(test_ids),
            teacher_positive_decisions=int(positives.sum()),overall_accuracy=float((pred==target).float().mean()),thresholds={})
        for threshold in plan['thresholds']:
            selected=(pred!=0)&(score>=threshold);correct=(pred==target)&selected
            item['thresholds'][str(threshold)]=dict(selected=int(selected.sum()),teacher_class_precision=float(correct.sum()/selected.sum()) if selected.any() else None,
                exact_positive_recall=float(correct.sum()/positives.sum()) if positives.any() else None)
        folds.append(item);torch.save(dict(model=model.state_dict(),held_out_initial=int(g)),OUT/f'fold_{g}.pt')
        print(json.dumps(item),flush=True)
    for seed in plan['final_seeds']:
        model=fit(x,y,seed);torch.save(dict(algorithm=plan['algorithm'],model=model.state_dict(),seed=seed,
            teacher_data_sha256=digest,training_initial_groups=len(set(groups)),risk_certificate=False),OUT/f'ensemble_{seed}.pt')
    report=dict(folds=folds,final_models=5,independent_initial_groups=len(set(groups)),teacher_rows=len(y),
        inference='Only SIX independent initials. Classification is NOT no-harm calibration or closed-loop traffic efficacy.',risk_certificate=False)
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (OUT/'complete.json').write_text(json.dumps(dict(final_models=5,model_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.pt')}),indent=2)+'\n')


if __name__=='__main__':train()
