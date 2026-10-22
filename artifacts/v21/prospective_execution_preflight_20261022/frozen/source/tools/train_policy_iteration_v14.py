"""Development experiment: distill iteratively improved sequential policies.

The full source and exact data budget are recorded. No evaluation score enters
data generation, gradient updates, or final checkpoint choice.
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from coopgate.ego_v13 import EgoVector
from coopgate.learned_rollout_v13 import LearnedRolloutGate
from coopgate.policy_iteration_v14 import IteratedPolicy, batch_returns

ROOT = Path(__file__).resolve().parents[1]


def fit(model, data, rng, epochs, batch_size=512):
    obs, costs = data
    mask = np.isfinite(costs)
    regret = np.where(mask, costs - np.min(costs, axis=1, keepdims=True), 0.)
    # Targets preserve all equal-cost actions. A small tie preference selects
    # the largest proposal, matching the conventional rule's tie convention.
    tie = np.arange(costs.shape[1])[None] * .0001
    targets = np.exp(np.clip((-regret + tie) / .04, -50., 0.)) * mask
    targets /= targets.sum(-1, keepdims=True)
    ob = torch.tensor(obs)
    ma = torch.tensor(mask)
    re = torch.tensor(regret, dtype=torch.float32)
    ta = torch.tensor(targets, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    for _ in range(epochs):
        for ids in np.array_split(rng.permutation(len(obs)), max(1, (len(obs) + batch_size - 1) // batch_size)):
            logits = model(ob[ids]).masked_fill(~ma[ids], -1e9)
            lp = logits.log_softmax(-1)
            ce = -(ta[ids] * lp).sum(-1)
            expected_regret = (logits.softmax(-1) * re[ids]).sum(-1)
            loss = (ce + 2. * expected_regret).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            opt.step()
    with torch.no_grad():
        logits = model(ob).masked_fill(~ma, -1e9)
        selected = logits.argmax(-1)
        regret_value = re.gather(1, selected[:, None]).mean().item()
    return float(loss.detach()), regret_value


def train(out, seed=801, iterations=4, states_per_iteration=8192, epochs=30,
          teacher='artifacts/v13/ego/seed_701/policy.pt', rule_only=False, width=128,
          warmstart=None, latest_only=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = IteratedPolicy(width)
    if warmstart:
        from coopgate.policy_iteration_v14 import load_policy
        model, warm_metadata = load_policy(warmstart)
        if warm_metadata['width']!=width:raise ValueError('Warmstart width mismatch')
    reference = LearnedRolloutGate(teacher)
    meta = dict(algorithm='rollout_policy_iteration_ego_v14', seed=seed, width=width,
                iterations=iterations, states_per_iteration=states_per_iteration,
                epochs=epochs, teacher=str(teacher), teacher_sha256=hashlib.sha256(Path(teacher).read_bytes()).hexdigest(),
                rule_only=rule_only, learned_vehicles=[1], objective='ego time + 0.05 ego command variation',
                warmstart=str(warmstart) if warmstart else None, latest_only=latest_only,
                warmstart_sha256=hashlib.sha256(Path(warmstart).read_bytes()).hexdigest() if warmstart else None,
                inference='local neural actor followed by the shared physical safety filter; no performance rollout',
                data='on-policy visited states with 15% admissible action exploration and nominal complete-action training rollouts',
                checkpoint_selection='fixed final iteration; development scores do not affect selection')
    sources = ['src/coopgate/policy_iteration_v14.py', 'tools/train_policy_iteration_v14.py']
    meta['source_sha256'] = {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
    for name in sources:
        target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    (out/'protocol.json').write_text(json.dumps(meta, indent=2)+'\n')
    torch.save({**meta, 'iteration':0, 'model':model.state_dict()}, out/'initial.pt')
    history = []
    stored_obs, stored_costs = [], []
    total_transitions = visited = 0
    started = time.perf_counter()
    for iteration in range(iterations):
        env = EgoVector(32, 34000000 + seed * 10 + iteration, .05)
        if iteration == 0 and not warmstart:
            weights, kind = reference.weights, reference.kind
        else:
            weights, kind = model.compiled_weights(), 2
        ob_round, costs_round = [], []
        round_transitions = 0
        for j in range(0, states_per_iteration, env.n):
            obs, _, masks = env.inputs()
            obs, mask = obs[:, 0], masks[:, 0]
            costs, count = batch_returns(env.x, env.previous, env.steps, env.packed,
                                         weights, kind, not rule_only)
            ob_round.append(obs.copy())
            costs_round.append(costs)
            round_transitions += int(count)
            # Dataset aggregation covers the current deployed apprentice;
            # first-round collection also uses random-initial apprentice.
            selected = model.proposal(obs, mask)
            random = (rng.random(mask.shape) - np.where(mask, 0., 2.)).argmax(-1)
            selected = np.where(rng.random(env.n) < .15, random, selected)
            env.step(selected[:, None])
        new_obs, new_costs = np.concatenate(ob_round), np.concatenate(costs_round)
        np.savez_compressed(out/f'data_{iteration+1}.npz', obs=new_obs, costs=new_costs)
        if latest_only:stored_obs,stored_costs=[],[]
        stored_obs.append(new_obs)
        stored_costs.append(new_costs)
        # Retain earlier visited states, while newly generated returns reflect
        # the current policy. Old policy-return targets remain explicitly old.
        loss, regret = fit(model, (np.concatenate(stored_obs), np.concatenate(stored_costs)), rng, epochs)
        total_transitions += round_transitions
        visited += len(new_obs)
        row = dict(iteration=iteration+1, visited_states=visited,
                   evaluated_model_transitions=total_transitions,
                   imitation_loss=loss, training_action_regret_s=regret,
                   elapsed_s=time.perf_counter()-started)
        history.append(row)
        print(json.dumps(row), flush=True)
        checkpoint = {**meta, **row, 'model':model.state_dict()}
        torch.save(checkpoint, out/f'iteration_{iteration+1}.pt')
        torch.save(checkpoint, out/'policy.pt')
        with (out/'training.csv').open('w', newline='') as file:
            writer=csv.DictWriter(file, fieldnames=row, lineterminator='\n')
            writer.writeheader()
            writer.writerows(history)
    (out/'metadata.json').write_text(json.dumps({**meta, **history[-1]}, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--seed', type=int, default=801)
    parser.add_argument('--iterations', type=int, default=4)
    parser.add_argument('--states', type=int, default=8192)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--teacher', default='artifacts/v13/ego/seed_701/policy.pt')
    parser.add_argument('--rule-only', action='store_true')
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--warmstart')
    parser.add_argument('--latest-only', action='store_true')
    args=parser.parse_args()
    train(args.out, args.seed, args.iterations, args.states, args.epochs,
          args.teacher, args.rule_only, args.width, args.warmstart, args.latest_only)
