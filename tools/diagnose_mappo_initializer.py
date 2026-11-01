"""Training-set phase/class diagnostics, distinct from held-out traffic efficacy."""
import argparse
import json
from pathlib import Path

import torch

from coopgate.mappo_ctde import MAPPOCTDE,PPOConfig


def diagnose(directory):
    directory=Path(directory);torch.set_num_threads(1)
    data=torch.load(directory/'dataset.pt',map_location='cpu',weights_only=True)
    source=torch.load(directory/'initializer.pt',map_location='cpu',weights_only=True)
    if source['algorithm']!='planner_bc_initializer_20261101':
        raise ValueError('Explicit BC initialization required')
    model=MAPPOCTDE(PPOConfig(**source['ppo_config']))
    model.actor.load_state_dict(source['model']['actor'])
    obs,mask,label=data['obs'],data['mask'],data['labels']
    with torch.no_grad():
        dist=model.distribution(obs,mask);prediction=dist.logits.argmax(-1)
    initial=obs[:,5]==0
    groups={}
    for name,ids in [('initial',initial),('after_initial',~initial)]:
        groups[name]=dict(samples=int(ids.sum()),
            accuracy=float((prediction[ids]==label[ids]).float().mean()) if ids.any() else None,
            labels=torch.bincount(label[ids],minlength=13).tolist(),
            predictions=torch.bincount(prediction[ids],minlength=13).tolist())
    report=dict(groups=groups,meaning='Training action classification only; actions can have different or equivalent physical effects',
                additional_simulator_runs=0,held_out_efficacy_claim=False)
    (directory/'phase_diagnosis.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory')
    diagnose(parser.parse_args().directory)
