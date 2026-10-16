"""Frozen-actor replacement diagnostics in the model, not additional SUMO runs.

Each replacement is evaluated from a fresh initial with the entire remaining
joint policy reacting to the changed trajectory. This is not a Nash test.
"""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
import torch
from coopgate.cooperative_v15 import load_joint
from coopgate.ppo_numerics_v15 import require_complete_training
from coopgate.ego_v13 import ego_observations
from coopgate.marl_v12 import LEVELS,action_mask,batch_step
from coopgate.merge_v4 import MergeConfig
from coopgate.fast_v8 import parameters
from coopgate.disturbance_v9 import disturbance_tape
from develop_v12 import initial


def rollout(model,reference,n=128,seed_base=59100001):
    c=MergeConfig();packed=parameters(c);modes=('clean','uniform','bias','pulse')
    records=[dict(seed=seed_base+i,regime='nominal' if i<n//2 else 'dense',disturbance=m) for i in range(n) for m in modes]
    x=np.array([initial(r['seed'],r['regime']) for r in records]);prev=np.zeros((len(x),3))
    tapes=np.array([disturbance_tape(r['seed'],r['disturbance'],.5,250) for r in records])
    arrival=np.zeros((len(x),3));variation=np.zeros_like(arrival);done=np.zeros(len(x),bool)
    proposed_change=np.zeros(3,int);executed_change=np.zeros(3,int);actor_samples=np.zeros(3,int)
    transitions=0
    for step in range(50):
        ids=np.flatnonzero(~done)
        if not len(ids):break
        state=x[ids];previous=prev[ids];active=state[:,:,0]<60
        obs=ego_observations(state,previous,step);mask=action_mask(previous)
        actions=model.proposal(obs,mask);original=reference.proposal(obs,mask)
        errors=tapes[ids,step*5:(step+1)*5]
        following,command,ok=batch_step(state,previous,LEVELS[actions],errors,packed,.5)
        _,reference_command,reference_ok=batch_step(state,previous,LEVELS[original],errors,packed,.5)
        if not ok.all() or not reference_ok.all():raise RuntimeError('Ablation physical contract failed')
        actor_samples+=active.sum(0);proposed_change+=((actions!=original)&active).sum(0)
        executed_change+=((np.abs(command-reference_command)>1e-9)&active).sum(0)
        arrival[ids]+=active*.5;variation[ids]+=active*np.abs(command-previous)
        x[ids]=following;prev[ids]=command;done[ids]=np.all(following[:,:,0]>=60,axis=1);transitions+=len(ids)
    for i,r in enumerate(records):
        r.update(cost=(arrival[i]+.05*variation[i]).tolist(),time=arrival[i].tolist(),
                 variation=variation[i].tolist(),complete=bool(done[i]))
    return dict(records=records,mean_cost=(arrival+.05*variation).mean(0).tolist(),
        completed=int(done.sum()),trajectories=len(records),model_transitions=transitions,
        active_actor_samples=actor_samples.tolist(),proposal_changes_vs_initial=proposed_change.tolist(),
        executed_changes_vs_initial_at_same_state=executed_change.tolist())


def run(paths,out,n=128,seed_base=59100001):
    torch.set_num_threads(1);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    if n<4 or n%2:raise ValueError('Even paired regime count required')
    results={};protocol=dict(purpose='Development model-only actor replacement; not SUMO or an equilibrium proof',
        n_initials=n,seed_base=seed_base,variants=['full','restore_0','restore_1','restore_2','all_initial'],models={})
    for path in paths:
        path=Path(path);model,meta=load_joint(path);require_complete_training(meta)
        reference,_=load_joint(path.parent/'initial.pt');name=('mappo' if meta['critic_kind']=='central' else 'ippo')+f'_{meta["seed"]}'
        protocol['models'][name]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            initial_sha256=hashlib.sha256((path.parent/'initial.pt').read_bytes()).hexdigest())
        results[name]={}
        for variant in protocol['variants']:
            current=copy.deepcopy(model)
            if variant=='all_initial':current=copy.deepcopy(reference)
            elif variant.startswith('restore_'):
                i=int(variant[-1]);current.actors[i]=copy.deepcopy(reference.actors[i])
            result=rollout(current,reference,n,seed_base);results[name][variant]=result
            print(json.dumps(dict(method=name,variant=variant,**{k:v for k,v in result.items() if k!='records'})),flush=True)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('paths',nargs='+');p.add_argument('--out',required=True)
    p.add_argument('--n',type=int,default=128);p.add_argument('--seed-base',type=int,default=59100001)
    a=p.parse_args();run(a.paths,a.out,a.n,a.seed_base)
