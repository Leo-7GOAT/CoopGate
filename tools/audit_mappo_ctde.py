"""Independently reconstruct saved on-policy batches and PPO gradients."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from coopgate.mappo_ctde import MAPPOCTDE, PPOConfig, losses, load_mappo


def audit(directory):
    directory=Path(directory)
    torch.set_num_threads(1)
    meta=json.loads((directory/'protocol.json').read_text(encoding='utf-8'))
    complete=json.loads((directory/'complete.json').read_text(encoding='utf-8'))
    for name, expected in meta['source_sha256'].items():
        if hashlib.sha256((directory/'source'/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Training snapshot hash mismatch: '+name)
    model, final=load_mappo(directory/'policy.pt')
    if complete['policy_sha256'] != hashlib.sha256((directory/'policy.pt').read_bytes()).hexdigest():
        raise ValueError('Final policy hash mismatch')
    if final['steps'] != meta['target_steps'] or complete['steps'] != meta['target_steps']:
        raise ValueError('Incomplete training')
    snapshots=sorted(directory.glob('rollout_before_update_*.pt'))
    batches=[]
    for path in snapshots:
        # These are this project's locally generated trusted diagnostic tensors,
        # including numpy buffers, not an external untrusted checkpoint.
        saved=torch.load(path, map_location='cpu', weights_only=False)
        cfg=PPOConfig(**saved['metadata']['ppo_config'])
        before=MAPPOCTDE(cfg, saved['metadata']['critic_kind'])
        before.load_state_dict(saved['before_model'])
        b=saved['batch']
        with torch.no_grad():
            lp=before.distribution(b['obs'], b['mask']).log_prob(b['actions'])
            value=before.value(b['obs'], b['state'])
        torch.testing.assert_close(lp, b['old_logp'], rtol=0, atol=2e-5)
        torch.testing.assert_close(value, b['old_value'], rtol=0, atol=1e-5)
        # Reconstruct GAE explicitly, without calling the training helper.
        r, v, nv, terminal = [saved[k] for k in ('rewards', 'values', 'next_values', 'boundaries')]
        result=np.zeros_like(v)
        for t in range(len(r)):
            multiplier=1.
            for j in range(t, len(r)):
                delta=r[j, :, None]+cfg.gamma*(~terminal[j, :, None])*nv[j]-v[j]
                result[t] += multiplier*delta
                multiplier=multiplier*cfg.gamma*cfg.gae_lambda*(~terminal[j, :, None])
        active=saved['active']
        standardized=(result-result[active].mean())/(result[active].std()+1e-8)
        np.testing.assert_allclose(standardized.reshape(-1, 3), b['advantage'].numpy(), atol=3e-5, rtol=1e-5)
        np.testing.assert_allclose((result+v).reshape(-1, 3), b['returns'].numpy(), atol=3e-5, rtol=1e-5)
        loss, detail=losses(before, b, cfg)
        loss.backward()
        actor_norm=sum(float(p.grad.square().sum()) for p in before.actor.parameters())**.5
        critic_norm=sum(float(p.grad.square().sum()) for p in before.critic.parameters())**.5
        if not actor_norm > 0 or not critic_norm > 0:
            raise ValueError('Actor or centralized critic has no gradient')
        role_gradients=[]
        for role in range(3):
            before.zero_grad()
            selected=b['active'][:, role]
            if not selected.any():
                role_gradients.append(None)
                continue
            dist=before.distribution(b['obs'], b['mask'])
            role_loss=-(dist.log_prob(b['actions'])[:, role]*b['advantage'][:, role])[selected].mean()
            role_loss.backward()
            role_gradients.append(sum(float(p.grad.square().sum()) for p in before.actor.parameters())**.5)
        if meta['stage'] == 'joint' and not all(g is not None and g > 0 for g in role_gradients):
            raise ValueError('A joint-training role supplies no actor gradient')
        batches.append(dict(path=path.name, steps=saved['metadata']['steps'],
            sampled_action_logp_max_error=float((lp-b['old_logp']).abs().max()),
            actor_gradient_l2=actor_norm, critic_gradient_l2=critic_norm,
            per_role_actor_gradient_l2=role_gradients,
            active_role_samples=b['active'].sum(0).tolist(),
            independent_gae_max_error=float(np.max(np.abs(standardized.reshape(-1, 3)-b['advantage'].numpy())))))
    if len(batches) != 2:
        raise ValueError('Expected first and final independently auditable rollouts')
    report=dict(algorithm=meta['algorithm'], stage=meta['stage'], steps=final['steps'],
        final_budget_complete=True, on_policy_proposals_verified=True,
        independent_gae_verified=True, snapshots=batches,
        policy_sha256=complete['policy_sha256'],
        meaning='Algorithm integrity only; not evidence of superiority or safety')
    (directory/'audit.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report), flush=True)
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('directory')
    audit(parser.parse_args().directory)
