"""Inspect actual saved initializations; no efficacy or trained-policy claim."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from coopgate.mappo_ctde import load_mappo

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/mappo_ctde_20261101/development'


def audit(seed):
    if seed not in (4703,4704,4705):raise ValueError('Registered paired seeds only')
    torch.set_num_threads(1)
    left=BASE/f'joint_{seed}/initial.pt';right=BASE/f'ippo_{seed}/initial.pt'
    central,cm=load_mappo(left);local,lm=load_mappo(right)
    if (cm['steps']!=0 or lm['steps']!=0 or cm['critic_kind']!='central'
            or lm['critic_kind']!='local' or cm['seed']!=lm['seed']
            or cm['ppo_config']!=lm['ppo_config'] or cm['initial_checkpoint']!=lm['initial_checkpoint']
            or cm['proposal_rng_seed']!=lm['proposal_rng_seed']):
        raise ValueError('Initialization,configuration or proposal stream mismatch')
    a=central.actor.state_dict();b=local.actor.state_dict()
    if set(a)!=set(b) or not all(torch.equal(a[k],b[k]) for k in a):
        raise ValueError('Any paired actor initialization differs')
    report=dict(seed=seed,all_actor_initial_tensors_identical=True,ppo_config_identical=True,
        proposal_rng_seed_identical=True,checkpoint_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (left,right)},
        actor_parameters=sum(p.numel() for p in central.actor.parameters()),
        centralized_critic_parameters=sum(p.numel() for p in central.critic.parameters()),
        shared_local_critic_parameters=sum(p.numel() for p in local.critic.parameters()),
        critic_parameter_counts_equal=False,trained_steps_proved_by_this_audit=0,
        meaning='Initialization control only; shared local critic differs in information and parameter/compute count; no efficacy claim')
    (right.parent/'paired_initialization_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True);audit(parser.parse_args().seed)
