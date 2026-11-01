"""Complete paired development evaluation, exact physics or actual SUMO.

This does not reuse a PI certificate, select a lucky checkpoint, or silently
discard incomplete episodes. Original strong planners remain in the comparison.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from scipy.stats import t

from coopgate.cooperative_v15 import JointPolicyGate
from coopgate.disturbance_v9 import DisturbedMergeEnv, disturbance_tape
from coopgate.ego_v13 import EgoRolloutGate, ego_observations
from coopgate.fast_v8 import parameters
from coopgate.joint_planner_v15 import JointRolloutGate
from coopgate.mappo_ctde import MAPPOCTDE, PPOConfig, load_mappo
from coopgate.marl_v12 import LEVELS, action_mask, batch_step, project_joint
from coopgate.merge_v4 import MergeConfig
from develop_v12 import initial
from experiment_ego_v13 import run_episode


class PPOGate(JointPolicyGate):
    def __init__(self, policy, stage):
        super().__init__(policy)
        self.stage=stage

    def select(self, state, previous, unused=None):
        if self.stage == 'joint':
            return super().select(state, previous, unused)
        started=time.perf_counter()
        mask=action_mask(previous)
        actions=np.where(mask, np.arange(len(LEVELS)), -1).max(-1)
        actions[1]=self.policy.proposal(ego_observations(state, previous, self.step)[1:2], mask[1:2])[0]
        proposal=LEVELS[actions].copy()
        cleared=state[:, 0] >= self.c.goal
        proposal[cleared]=np.minimum(self.c.accel, previous[cleared]+2.5)
        command, vertex, ok=project_joint(state, previous, proposal, self.packed, .5)
        if not ok:
            raise RuntimeError('Single PPO has no certified command')
        self.step += 1
        return command, dict(gate_ms=(time.perf_counter()-started)*1000.,
                             agent_actions=actions.tolist(), proposal=proposal.tolist(),
                             shield_vertex=vertex, certified=True, in_set=True,
                             intervention=float(np.mean(np.abs(command-proposal)>1e-8)))


class RuleGate:
    def __init__(self):
        self.packed=parameters(MergeConfig())

    def select(self, state, previous, unused=None):
        started=time.perf_counter()
        proposal=np.minimum(2.5, previous+2.5)
        command, _, ok=project_joint(state, previous, proposal, self.packed, .5)
        if not ok:
            raise RuntimeError('Rule has no certified command')
        return command, dict(gate_ms=(time.perf_counter()-started)*1000.,
                             certified=True, in_set=True,
                             intervention=float(np.mean(np.abs(command-proposal)>1e-8)))


def exact_episode(seed, scenario, mode, factory):
    c=MergeConfig(); packed=parameters(c)
    state=initial(seed, scenario); start=state.copy()
    previous=np.zeros(3); variation=np.zeros(3); finish=np.full(3, c.horizon)
    tape=disturbance_tape(seed, mode, .5, 250)
    controller=factory(); trace=[]; error=None; safe=True
    # JIT/network warmup happens in evaluate(), outside recorded callbacks.
    for step in range(50):
        if np.all(state[:, 0] >= c.goal):
            break
        active=state[:, 0] < c.goal
        started=time.perf_counter()
        try:
            command, info=controller.select(state.copy(), previous.copy())
        except (RuntimeError, ValueError) as exc:
            elapsed=(time.perf_counter()-started)*1000.
            error=dict(type=type(exc).__name__, message=str(exc), step=step, controller_ms=elapsed)
            break
        elapsed=(time.perf_counter()-started)*1000.
        # The command has already been physically projected; batch_step repeats
        # the same filter, which MUST be idempotent for a certified command.
        nxt, executed, valid=batch_step(state[None], previous[None], command[None],
                                       tape[None, step*5:step*5+5], packed, .5)
        if not np.array_equal(executed[0], command):
            raise RuntimeError('Certified command changed under the common filter')
        variation += active*np.abs(command-previous)
        arrived=active & (nxt[0, :, 0] >= c.goal)
        finish[arrived]=(step+1)*c.decision_dt
        trace.append(dict(step=step, before=state.tolist(), previous=previous.tolist(),
                          command=command.tolist(), after=nxt[0].tolist(),
                          controller_ms=elapsed, safe=bool(valid[0]),
                          proposal=info.get('proposal'), actions=info.get('agent_actions')))
        state=nxt[0]; previous=command; safe &= bool(valid[0])
        if not valid[0]:
            error=dict(type='PhysicalViolation', message='Unsafe physical transition', step=step)
            break
    complete=bool(np.all(state[:, 0] >= c.goal) and error is None and safe)
    cost=finish+.05*variation
    return dict(seed=seed, scenario=scenario, mode=mode, initial=start.tolist(),
                completed=complete, aborted=error is not None, error=error,
                per_vehicle_finish=finish.tolist(), per_vehicle_variation=variation.tolist(),
                per_vehicle_cost=cost.tolist(), team_cost=float(cost.mean()),
                controller_ms=[x['controller_ms'] for x in trace], trace=trace)


def evaluate(out, paths, n=128, seed_base=94500001, backend='exact', expert_initial=None):
    out=Path(out); out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    factories=dict(rule=RuleGate, ego_planner=EgoRolloutGate, team_planner=JointRolloutGate)
    checkpoints={}
    if expert_initial:
        source=torch.load(expert_initial, map_location='cpu', weights_only=True)
        if source['algorithm'] != 'planner_bc_initializer_20261101':
            raise ValueError('BC-only control must use an explicit non-PPO initializer')
        bc=MAPPOCTDE(PPOConfig(**source['ppo_config']))
        bc.actor.load_state_dict(source['model']['actor']); bc.eval()
        factories['bc_only']=lambda:PPOGate(bc, 'single')
        checkpoints['bc_only']=dict(path=str(expert_initial), sha256=hashlib.sha256(Path(expert_initial).read_bytes()).hexdigest(),
                                    stage='bc_only', steps=0)
    for path in paths:
        model, meta=load_mappo(path)
        if meta['steps'] != meta['target_steps']:
            raise ValueError('Primary comparison requires completed final checkpoint')
        name=f"{meta['stage']}_{meta['critic_kind']}_{meta['seed']}"
        if name in factories:
            raise ValueError('Duplicate trained method')
        factories[name]=lambda model=model, stage=meta['stage']:PPOGate(model, stage)
        checkpoints[name]=dict(path=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                               stage=meta['stage'], steps=meta['steps'])
    modes=('clean', 'uniform', 'bias', 'pulse')
    protocol=dict(purpose='development; not a risk certificate or untouched confirmation',
                  backend=backend, n_independent_initials=n, seed_base=seed_base,
                  modes=modes, methods=list(factories), checkpoints=checkpoints,
                  expected_executions=n*len(modes)*len(factories), no_outcome_filtering=True)
    root=Path(__file__).resolve().parents[1]
    inputs=list((root/'src/coopgate').glob('*.py'))+[Path(__file__)]
    protocol['source_sha256']={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n', encoding='utf-8')
    for factory in factories.values():
        factory().select(initial(94509999, 'nominal'), np.zeros(3))
    batch_step(initial(94509999, 'nominal')[None], np.zeros((1, 3)), np.zeros((1, 3)),
               np.zeros((1, 5, 3)), parameters(MergeConfig()), .5)
    rows=[]; started=time.perf_counter()
    env=DisturbedMergeEnv() if backend == 'sumo' else None
    if env:
        env.__enter__()
    try:
        with (out/'episodes.jsonl').open('x', encoding='utf-8') as file:
            for i in range(n):
                seed=seed_base+i
                scenario='nominal' if i<n//2 else 'dense'
                for mode in modes:
                    for name, factory in factories.items():
                        if backend == 'exact':
                            row=exact_episode(seed, scenario, mode, factory)
                        else:
                            result, trace, actual_initial=run_episode(env, seed, scenario, mode,
                                                        lambda _state, f=factory:f(), True)
                            row={**result, 'seed':seed, 'scenario':scenario, 'mode':mode,
                                 'initial':actual_initial.tolist(), 'trace':trace}
                            cost=np.array(row['per_vehicle_finish'])+.05*np.array(row['per_vehicle_variation'])
                            row.update(per_vehicle_cost=cost.tolist(), team_cost=float(cost.mean()))
                        row['method']=name
                        file.write(json.dumps(row)+'\n'); file.flush()
                        rows.append({k:v for k,v in row.items() if k != 'trace'})
                if (i+1)%16 == 0 or i+1 == n:
                    print(json.dumps(dict(initials_completed=i+1, executions=len(rows),
                                          elapsed_s=time.perf_counter()-started)), flush=True)
    finally:
        if env:
            env.__exit__(None, None, None)
    lookup={(r['seed'], r['mode'], r['method']):r for r in rows}
    if len(lookup) != protocol['expected_executions']:
        raise RuntimeError('Missing or duplicate comparisons')
    summary={}
    for method in factories:
        own=[r for r in rows if r['method'] == method]
        failures=sum(not r['completed'] for r in own)
        comparisons={}
        references=list(dict.fromkeys(['ego_planner','team_planner']+list(checkpoints)))
        for reference in references:
            paired=[lookup[r['seed'], r['mode'], reference] for r in own]
            if failures or any(not p['completed'] for p in paired):
                comparisons[reference]=dict(all_scope_gain_assessable=False)
                continue
            # Average the repeated modes WITHIN each independent initial first.
            deltas=np.array([np.array(p['per_vehicle_cost'])-r['per_vehicle_cost']
                             for r,p in zip(own, paired)]).reshape(n, len(modes), 3).mean(1)
            gain=(np.array([p['team_cost'] for p in paired])-np.array([r['team_cost'] for r in own]))
            gain=(100.*gain/np.array([p['team_cost'] for p in paired])).reshape(n, len(modes)).mean(1)
            team=deltas.mean(-1)
            lower=float(team.mean()-t.ppf(.95, n-1)*team.std(ddof=1)/np.sqrt(n)) if n>1 else None
            comparisons[reference]=dict(all_scope_gain_assessable=True,
                mean_paired_percentage_gain=float(gain.mean()), mean_team_gain_s=float(team.mean()),
                team_gain_one_sided_t_lower_s=lower,
                mean_per_vehicle_gain_s=deltas.mean(0).tolist(),
                per_vehicle_gain_one_sided_t_lower_s=(deltas.mean(0)-t.ppf(.95,n-1)*deltas.std(0,ddof=1)/np.sqrt(n)).tolist() if n>1 else None)
        timings=np.concatenate([r['controller_ms'] for r in own])
        summary[method]=dict(executions=len(own), failures=failures,
            mean_per_vehicle_cost=np.mean([r['per_vehicle_cost'] for r in own], axis=0).tolist(),
            controller_p95_ms=float(np.percentile(timings, 95)), controller_max_ms=float(timings.max()),
            comparisons=comparisons)
    report=dict(protocol=protocol, results=summary, executions=len(rows),
                elapsed_s=time.perf_counter()-started,
                intervals='Development t intervals on independent initial means; not finite-sample certificates')
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(executions=report['executions'],elapsed_s=report['elapsed_s'],
                          results=summary)),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--paths', nargs='+', required=True)
    parser.add_argument('--n', type=int, default=128)
    parser.add_argument('--seed-base', type=int, default=94500001)
    parser.add_argument('--backend', choices=['exact','sumo'], default='exact')
    parser.add_argument('--expert-initial')
    args=parser.parse_args()
    evaluate(args.out, args.paths, args.n, args.seed_base, args.backend, args.expert_initial)
