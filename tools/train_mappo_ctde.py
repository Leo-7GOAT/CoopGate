"""Fresh on-policy CTDE training; outputs cannot overwrite an earlier run."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
from dataclasses import asdict

import numpy as np
import torch

from coopgate.mappo_ctde import MAPPOCTDE, PPOConfig, PPOVector, gae, losses, load_mappo

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train(out, steps, seed, stage, critic_kind='central', initial=None, config=None, expert_initial=None,
          continue_single=None, reset_pool=None,legacy_initial=None,rear_coast=False):
    cfg = config or PPOConfig()
    if steps <= 0 or steps % cfg.n_envs:
        raise ValueError('Step budget must be a positive multiple of vector width')
    if initial and stage != 'joint':
        raise ValueError('Only the joint stage may warm start from new single PPO')
    if expert_initial and (initial or stage != 'single'):
        raise ValueError('Expert initializer is a separate single-stage development path')
    if continue_single and (initial or expert_initial or stage!='single'):
        raise ValueError('Single continuation is a separate failed-prerequisite development path')
    if legacy_initial and (initial or expert_initial or continue_single or stage!='single'):
        raise ValueError('Legacy initialization is a separately disclosed single-stage experiment')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = MAPPOCTDE(cfg, critic_kind)
    init_meta = None
    if initial:
        source, init_meta = load_mappo(initial)
        if init_meta['stage'] != 'single' or init_meta['steps'] != init_meta['target_steps']:
            raise ValueError('Only completed new single PPO weights may initialize joint actor')
        model.actor.load_state_dict(source.actor.state_dict())
    if expert_initial:
        source=torch.load(expert_initial, map_location='cpu', weights_only=True)
        if source['algorithm'] != 'planner_bc_initializer_20261101' or source['ppo_config'] != asdict(cfg):
            raise ValueError('Expected a separately declared matching planner-BC initializer')
        model.actor.load_state_dict(source['model']['actor'])
    if continue_single:
        source, previous_meta=load_mappo(continue_single)
        if (previous_meta['stage']!='single' or previous_meta['steps']!=previous_meta['target_steps']
                or previous_meta['ppo_config']!=asdict(cfg)):
            raise ValueError('Single continuation requires completed matching on-policy single PPO')
        model.load_state_dict(source.state_dict())
    if legacy_initial:
        source=torch.load(legacy_initial,map_location='cpu',weights_only=True)
        if source['algorithm']!='legacy_pi_actor_initializer_20261101' or source['ppo_config']!=asdict(cfg):
            raise ValueError('Explicit matching legacy-PI initializer required; it is not a PPO result')
        model.actor.load_state_dict(source['model']['actor'])
    if rear_coast:
        if stage!='joint' or not initial or not cfg.role_actors:
            raise ValueError('Rear courtesy prior is a disclosed joint-stage initialization only')
        from coopgate.marl_v12 import LEVELS
        with torch.no_grad():
            model.actor.roles[2][-1].weight.zero_()
            model.actor.roles[2][-1].bias.copy_(torch.tensor(-.08*np.abs(LEVELS),dtype=torch.float32))
    initial_actor = {k:v.detach().clone() for k,v in model.actor.state_dict().items()}
    if cfg.ego_actor_lr>0:
        if not cfg.role_actors:
            raise ValueError('A separate ego learning rate requires independent role actors')
        actor_groups=[dict(params=actor.parameters(),lr=cfg.ego_actor_lr if i==1 else cfg.actor_lr)
                      for i,actor in enumerate(model.actor.roles)]
    else:
        actor_groups=([dict(params=model.actor.initial.parameters(),lr=cfg.initial_actor_lr),
                       dict(params=model.actor.flow.parameters(),lr=cfg.actor_lr)] if cfg.phase_actor
                      else [dict(params=model.actor.parameters(),lr=cfg.actor_lr)])
    optimizer = torch.optim.Adam(actor_groups+[dict(params=model.critic.parameters(),lr=cfg.critic_lr)],eps=1e-5)
    # Keep rollout/minibatch RNG independent of how many random numbers the
    # chosen critic architecture or initializer consumed during construction.
    proposal_rng_seed=seed+451000000
    torch.manual_seed(proposal_rng_seed)
    source_paths = list((ROOT/'src/coopgate').glob('*.py')) + [Path(__file__), ROOT/'docs/mappo_ctde_20261101_protocol.md']
    source_hashes = {p.relative_to(ROOT).as_posix():digest(p) for p in source_paths}
    inherited_legacy=(init_meta or {}).get('legacy_initializer') or (init_meta or {}).get('inherited_legacy_initializer')
    meta = dict(algorithm='mappo_ctde_20261101' if critic_kind == 'central' else 'ippo_local_20261101',
        stage=stage, critic_kind=critic_kind, seed=seed, proposal_rng_seed=proposal_rng_seed,
        target_steps=steps, ppo_config=asdict(cfg),
        initial_checkpoint=None if initial is None else dict(path=str(initial), sha256=digest(initial)),
        expert_initializer=None if expert_initial is None else dict(path=str(expert_initial), sha256=digest(expert_initial)),
        continued_single_checkpoint=None if continue_single is None else dict(path=str(continue_single), sha256=digest(continue_single)),
        legacy_initializer=None if legacy_initial is None else dict(path=str(legacy_initial),sha256=digest(legacy_initial),
            ancestry='Model-based policy-iteration ego actor; old performance/certificates are not new PPO evidence'),
        inherited_legacy_initializer=inherited_legacy,
        rear_coast_initialization=bool(rear_coast),
        reset_pool=None if reset_pool is None else dict(path=str(reset_pool), sha256=digest(reset_pool), probability=.5,
            training_metrics='suffix cost on 50% reset states; not comparable to original-start costs'),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        source_sha256=source_hashes, training_backend='exact vector simulator, not SUMO process invocations',
        likelihood_action='sampled pre-shield categorical proposal',
        actor_information='local 18 observed features only', critic_information='global 13 public state features' if critic_kind=='central' else 'local 18 observed features',
        reward='ramp cost' if stage=='single' else 'mean per-car cost',
        time_limit='terminal finite 25s horizon; no terminal bootstrap',
        primary_checkpoint='final fixed step budget', no_teacher_actions=expert_initial is None and continue_single is None and legacy_initial is None and inherited_legacy is None,
        no_teacher_actions_in_ppo_updates=True, no_policy_iteration=legacy_initial is None and inherited_legacy is None,
        no_policy_iteration_in_ppo_updates=True)
    (out/'protocol.json').write_text(json.dumps(meta, indent=2)+'\n', encoding='utf-8')
    for name in source_hashes:
        target=out/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    torch.save(dict(**meta, steps=0, model=model.state_dict()), out/'initial.pt')
    if reset_pool:
        from coopgate.mappo_reset_curriculum import ResetCurriculumVector
        env=ResetCurriculumVector(reset_pool,cfg.n_envs,seed+450000000,stage)
    else:
        env = PPOVector(cfg.n_envs, seed+450000000, stage)
    obs, state, mask = env.inputs()
    history, episodes, samples = [], [], np.zeros(3, np.int64)
    total = updates = 0
    started = time.perf_counter()
    while total < steps:
        records = []
        for _ in range(min(cfg.rollout, (steps-total)//cfg.n_envs)):
            ob, st, ma = torch.as_tensor(obs), torch.as_tensor(state), torch.as_tensor(mask)
            with torch.no_grad():
                dist = model.distribution(ob, ma)
                sampled = dist.sample()
                logp = dist.log_prob(sampled)
                val = model.value(ob, st)
            executed_indices = env.execute(sampled.numpy(), mask)
            following, reward, boundary, active, finished = env.step(executed_indices)
            with torch.no_grad():
                nextval = model.value(torch.as_tensor(following[0]), torch.as_tensor(following[1]))
            records.append((obs, state, mask, sampled.numpy(), logp.numpy(), val.numpy(),
                            reward, nextval.numpy(), boundary, active))
            episodes.extend(finished)
            samples += active.sum(0)
            # Parent returns final pre-reset inputs for bootstrap, but the next
            # sampled action MUST use the newly reset env's current inputs.
            obs, state, mask = env.inputs()
            total += cfg.n_envs
        ob, st, ma, actions, old_logp, values, rewards, nv, boundary, active = map(np.asarray, zip(*records))
        advantage, returns = gae(rewards, values, nv, boundary, boundary, cfg.gamma, cfg.gae_lambda)
        if not active.any():
            raise RuntimeError('No active samples in collected rollout')
        advantage = (advantage-advantage[active].mean())/(advantage[active].std()+1e-8)
        data = dict(obs=ob, state=st, mask=ma, actions=actions, old_logp=old_logp,
                    old_value=values, advantage=advantage, returns=returns, active=active)
        size = len(records)*cfg.n_envs
        batch = {k:torch.as_tensor(v.reshape((size,)+v.shape[2:])) for k,v in data.items()}
        # Verifiable before-update on-policy identity, not an assumption about
        # replay data or about action commands after projection.
        with torch.no_grad():
            recomputed = model.distribution(batch['obs'], batch['mask']).log_prob(batch['actions'])
            error = float((recomputed-batch['old_logp']).abs().max())
        if error > 2e-5:
            raise RuntimeError(f'Rollout not on-policy: logp error {error}')
        if updates == 0 or total == steps:
            torch.save(dict(batch=batch, rewards=rewards, values=values, next_values=nv,
                            boundaries=boundary, active=active,
                            before_model=model.state_dict(), metadata=dict(**meta, steps=total)),
                       out/f'rollout_before_update_{total}.pt')
        metrics = []
        for _ in range(cfg.epochs):
            for ids in torch.randperm(size).split(cfg.minibatch):
                mb = {k:v[ids] for k,v in batch.items()}
                if not mb['active'].any():
                    continue
                loss, detail = losses(model, mb, cfg)
                optimizer.zero_grad()
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                if not torch.isfinite(norm):
                    raise FloatingPointError('Nonfinite PPO gradient')
                optimizer.step()
                metrics.append({k:float(v.detach()) for k,v in detail.items()})
        updates += 1
        recent = episodes[-100:]
        change = sum(float((v-initial_actor[k]).square().sum()) for k,v in model.actor.state_dict().items())
        row = dict(joint_steps=total, updates=updates, episodes=env.episodes,
            recent_team_cost=float(np.mean([np.mean(e['cost']) for e in recent])) if recent else None,
            recent_ramp_cost=float(np.mean([e['cost'][1] for e in recent])) if recent else None,
            recent_completion=float(np.mean([e['complete'] for e in recent])) if recent else None,
            onpolicy_logp_max_error=error, actor_delta_squared_l2=change,
            actor_samples_0=int(samples[0]), actor_samples_1=int(samples[1]), actor_samples_2=int(samples[2]),
            **{k:float(np.mean([m[k] for m in metrics])) for k in metrics[0]},
            elapsed_s=time.perf_counter()-started)
        history.append(row)
        with (out/'training.csv').open('w', newline='', encoding='utf-8') as f:
            writer=csv.DictWriter(f, fieldnames=list(row)); writer.writeheader(); writer.writerows(history)
        checkpoint=dict(**meta, steps=total, model=model.state_dict(), optimizer=optimizer.state_dict(),
                        updates=updates, actor_samples=samples.tolist(), actor_delta_squared_l2=change)
        torch.save(checkpoint, out/'latest.pt')
        if total % 65536 == 0 or total == steps:
            torch.save(checkpoint, out/f'step_{total}.pt')
            print(json.dumps(row), flush=True)
    torch.save(checkpoint, out/'policy.pt')
    (out/'episodes.json').write_text(json.dumps(episodes)+'\n', encoding='utf-8')
    (out/'complete.json').write_text(json.dumps(dict(steps=total, target_steps=steps,
        elapsed_s=time.perf_counter()-started, policy_sha256=digest(out/'policy.pt'),
        actor_samples=samples.tolist(), actor_delta_squared_l2=change), indent=2)+'\n', encoding='utf-8')
    return checkpoint


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--stage', choices=['single','joint'], required=True)
    parser.add_argument('--critic', choices=['central','local'], default='central')
    parser.add_argument('--initial')
    parser.add_argument('--kinematic', action='store_true')
    parser.add_argument('--entropy', type=float, default=.01)
    parser.add_argument('--gae-lambda', type=float, default=.95)
    parser.add_argument('--expert-initial')
    parser.add_argument('--continue-single')
    parser.add_argument('--reset-pool')
    parser.add_argument('--accelerated',action='store_true')
    parser.add_argument('--actor-lr',type=float,default=3e-4)
    parser.add_argument('--phase-actor',action='store_true')
    parser.add_argument('--initial-actor-lr',type=float,default=3e-4)
    parser.add_argument('--startup-width',type=int,default=0)
    parser.add_argument('--startup-relu',action='store_true')
    parser.add_argument('--legacy-initial')
    parser.add_argument('--role-actors',action='store_true');parser.add_argument('--actor-relu',action='store_true')
    parser.add_argument('--logit-scale',type=float,default=1.)
    parser.add_argument('--width',type=int,default=128)
    parser.add_argument('--rear-coast',action='store_true')
    parser.add_argument('--ego-actor-lr',type=float,default=0.)
    args=parser.parse_args()
    config=PPOConfig(width=args.width,kinematic_features=args.kinematic or args.accelerated, entropy=args.entropy,
                     gae_lambda=args.gae_lambda,accelerated_features=args.accelerated,actor_lr=args.actor_lr,
                     phase_actor=args.phase_actor,initial_actor_lr=args.initial_actor_lr,
                     startup_width=args.startup_width,startup_relu=args.startup_relu,
                     role_actors=args.role_actors,actor_relu=args.actor_relu,actor_logit_scale=args.logit_scale,
                     ego_actor_lr=args.ego_actor_lr)
    train(args.out, args.steps, args.seed, args.stage, args.critic, args.initial, config, args.expert_initial,
          args.continue_single,args.reset_pool,args.legacy_initial,args.rear_coast)
