"""Bounded, post-hoc mechanism diagnosis; not a new effectiveness experiment."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from coopgate.disturbance_v9 import disturbance_tape
from coopgate.ego_v13 import EgoRolloutGate, ego_observations
from coopgate.fast_v8 import parameters
from coopgate.mappo_ctde import load_mappo
from coopgate.marl_v12 import action_mask, batch_step
from coopgate.merge_v4 import MergeConfig


def suffix(state, previous, step, tape, first=None):
    state=state.copy(); previous=previous.copy(); costs=np.zeros(3)
    gate=EgoRolloutGate(); gate.step=step
    for k in range(step,50):
        active=state[:,0]<60.
        if not active.any():
            break
        if k==step and first is not None:
            command=first.copy(); gate.step+=1
        else:
            command,_=gate.select(state,previous)
        nxt,actual,valid=batch_step(state[None],previous[None],command[None],
            tape[None,k*5:k*5+5],parameters(MergeConfig()),.5)
        if not valid[0] or not np.array_equal(actual[0],command):
            raise RuntimeError('Counterfactual suffix physically invalid')
        costs+=active*(.5+.05*np.abs(command-previous))
        state=nxt[0]; previous=command
    if np.any(state[:,0]<60.):
        raise RuntimeError('Counterfactual suffix incomplete')
    return costs


def diagnose(directory, checkpoint):
    directory=Path(directory)
    if not (directory/'report.json').exists():
        raise ValueError('Wait for complete evaluation')
    torch.set_num_threads(1)
    model,meta=load_mappo(checkpoint)
    name=f"{meta['stage']}_{meta['critic_kind']}_{meta['seed']}"
    rows=[json.loads(s) for s in (directory/'episodes.jsonl').read_text().splitlines()]
    references={(r['seed'],r['mode']):r for r in rows if r['method']=='ego_planner'}
    candidates=[r for r in rows if r['method']==name]
    candidates.sort(key=lambda r:r['per_vehicle_cost'][1]-references[r['seed'],r['mode']]['per_vehicle_cost'][1])
    # Outcome-based diagnostic selection is explicitly post-hoc. No inference
    # on this selected subset and no new independent sample count is reported.
    selected=candidates[:8]+candidates[-8:]
    output=[]
    for row in selected:
        record=dict(seed=row['seed'],mode=row['mode'],
                    full_episode_ramp_regret=row['per_vehicle_cost'][1]-references[row['seed'],row['mode']]['per_vehicle_cost'][1])
        for item in row['trace']:
            state=np.array(item['before']); previous=np.array(item['previous']); k=item['step']
            if state[1,0]>=60.:
                break
            baseline=EgoRolloutGate();baseline.step=k
            teacher,_=baseline.select(state,previous)
            command=np.array(item['command'])
            if np.array_equal(command,teacher):
                continue
            with torch.no_grad():
                dist=model.distribution(torch.as_tensor(ego_observations(state,previous,k)),
                    torch.as_tensor(action_mask(previous)))
                confidence=float(dist.probs[1].max())
            actual=disturbance_tape(row['seed'],row['mode'],.5,250)
            clean=np.zeros_like(actual)
            record.update(first_different_step=k, state=state.tolist(),previous=previous.tolist(),
                learned_command=command.tolist(),baseline_command=teacher.tolist(),
                actor_max_probability=confidence)
            try:
                record['realized_suffix_regret']=(suffix(state,previous,k,actual,command)-suffix(state,previous,k,actual)).tolist()
                record['nominal_suffix_regret']=(suffix(state,previous,k,clean,command)-suffix(state,previous,k,clean)).tolist()
            except (RuntimeError,ValueError) as exc:
                record['counterfactual_error']=str(exc)
            break
        output.append(record)
    report=dict(purpose='Post-hoc 8 best and 8 worst episode diagnostics only; no success selection',
        actual_sumo_runs=0, additional_independent_initials=0,
        realized_future_used='Offline counterfactual diagnosis ONLY; never an actor or gate input',
        actor_probability='Uncalibrated softmax, not safety confidence', cases=output)
    (directory/'mechanism_diagnosis.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(cases=len(output),first_command_differences=sum('first_different_step' in r for r in output))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');parser.add_argument('--checkpoint',required=True)
    args=parser.parse_args();diagnose(args.directory,args.checkpoint)
