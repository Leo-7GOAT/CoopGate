"""Train/validate offline high-level bandit policies; never open calibration/test."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from coopgate.learning_v6 import Selector, features


ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'artifacts/v6'


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    protocol = json.loads((D/'protocol.json').read_text()); cfg = protocol['training']
    if args.seed not in cfg['seeds']:
        raise ValueError('Seed not frozen in protocol')
    out = D/f'seed{args.seed}'; out.mkdir(exist_ok=False)
    train = np.load(D/'data/train.npz'); val = np.load(D/'data/validation.npz')
    torch.set_num_threads(1); torch.manual_seed(args.seed)
    x = torch.tensor(features(train['states'])); vx = torch.tensor(features(val['states']))
    costs = train['costs']-train['costs'][:, :1]
    target = torch.tensor(costs, dtype=torch.float32)
    model = Selector(); optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    rows = []; best = float('inf')
    for epoch in range(1, cfg['epochs']+1):
        model.train()
        for ids in torch.randperm(len(x)).split(cfg['batch_size']):
            logits, estimates = model(x[ids]); probabilities = logits.softmax(-1)
            value_loss = nn.functional.mse_loss(estimates-estimates[:, :1], target[ids])
            if epoch <= cfg['warmup_epochs']:
                soft_target = (-target[ids]/cfg['teacher_temperature']).softmax(-1)
                actor_loss = -(soft_target*logits.log_softmax(-1)).sum(-1).mean()
            else:
                # Exact full-information policy-gradient objective for one action
                # per episode; every action cost is from the frozen simulator model.
                entropy = -(probabilities*logits.log_softmax(-1)).sum(-1).mean()
                actor_loss = (probabilities*target[ids]).sum(-1).mean()-cfg['entropy_weight']*entropy
            loss = actor_loss+cfg['value_weight']*value_loss
            optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.); optimizer.step()
        if epoch == cfg['warmup_epochs'] or epoch % cfg['validation_interval'] == 0:
            model.eval()
            with torch.no_grad():
                action = model(vx)[0].argmax(-1).numpy()
            mean_cost = float(val['costs'][np.arange(len(action)), action].mean())
            row = {'epoch': epoch, 'validation_arrival_s': mean_cost, 'actor_loss': float(actor_loss.detach()), 'value_loss': float(value_loss.detach())}
            rows.append(row)
            data = {'benchmark': 'merge_v6_bandit', 'seed': args.seed, 'epoch': epoch,
                    'validation_arrival_s': mean_cost, 'model': model.state_dict()}
            if epoch == cfg['warmup_epochs']:
                torch.save(data, out/'imitation.pt')
            if epoch >= cfg['warmup_epochs'] and mean_cost < best-1e-10:
                best = mean_cost; torch.save(data, out/'policy.pt')
            if epoch % 30 == 0:
                print(json.dumps({'seed': args.seed, **row}), flush=True)
    with (out/'training.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator='\n'); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({'seed': args.seed, 'best_validation_arrival_s': best}), flush=True)


if __name__ == '__main__':
    main()
