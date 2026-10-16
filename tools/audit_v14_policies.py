"""Compare the trained Torch deployment policy with compiled training rollouts."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from coopgate.ego_v13 import ego_observations
from coopgate.marl_v12 import LEVELS,action_mask,batch_step,project_joint
from coopgate.single_v12 import constant_proposal
from coopgate.learned_rollout_v13 import actor_action
from coopgate.fast_v8 import parameters
from coopgate.merge_v4 import MergeConfig
from coopgate.disturbance_v9 import disturbance_tape
from develop_v12 import initial


def audit(path,n=64):
    torch.set_num_threads(1);meta=torch.load(path,map_location='cpu',weights_only=True)
    kind=meta['algorithm']
    if kind=='rollout_policy_iteration_ego_v14':
        from coopgate.policy_iteration_v14 import load_policy
        candidate_function=None
    elif kind=='certificate_feature_policy_iteration_v14':
        from coopgate.certificate_policy_v14 import load_policy,candidates as candidate_function,choose_compiled
    elif kind=='kinematic_feature_policy_iteration_v14':
        from coopgate.kinematic_policy_v14 import load_policy,candidates as candidate_function,choose_compiled
    else:raise ValueError(kind)
    model,_=load_policy(path);weights=model.compiled_weights();c=MergeConfig();packed=parameters(c)
    x=np.array([initial(39000001+i,'nominal' if i<n//2 else 'dense') for i in range(n)])
    previous=np.zeros((n,3));observations_checked=action_differences=command_differences=0
    tapes=np.array([disturbance_tape(39000001+i,['clean','uniform','bias','pulse'][i%4],.5,250) for i in range(n)])
    for step in range(50):
        proposals=[]
        for i in range(n):
            if candidate_function is None:
                mask=action_mask(previous[i])[1:2]
                selected=int(model.proposal(ego_observations(x[i],previous[i],step)[1:2],mask)[0])
                compiled_action=actor_action(x[i],previous[i],step,weights,True,2)
                compiled_index=int(np.argmin(np.abs(LEVELS-compiled_action)))
                p=constant_proposal(previous[i],packed);p[1]=LEVELS[selected]
                cp=constant_proposal(previous[i],packed);cp[1]=compiled_action
                if x[i,1,0]>=c.goal:p[1]=cp[1]=min(c.accel,previous[i,1]+2.5)
                command,_,valid=project_joint(x[i],previous[i],p,packed,.5)
                compiled,_,ok=project_joint(x[i],previous[i],cp,packed,.5)
                assert valid and ok
            else:
                f,commands,mask=candidate_function(x[i],previous[i],step,packed)
                selected=int(model.choose(f[None],mask[None])[0]);command=commands[selected]
                compiled,compiled_index=choose_compiled(x[i],previous[i],step,packed,weights)
            observations_checked+=1;action_differences+=int(selected!=compiled_index)
            command_differences+=int(not np.array_equal(command,compiled))
            proposals.append(command)
        nxt,_,valid=batch_step(x,previous,np.array(proposals),tapes[:,5*step:5*(step+1)],packed,.5)
        if not valid.all():raise RuntimeError('Failed physical contract')
        x,previous=nxt,np.array(proposals)
        if np.all(x[:,:,0]>=c.goal):break
    result=dict(path=str(path),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                states_checked=observations_checked,action_index_differences=action_differences,
                executed_command_differences=command_differences,completed=int(np.all(x[:,:,0]>=c.goal,axis=-1).sum()),n=n)
    if command_differences:raise RuntimeError(json.dumps(result))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('paths',nargs='+');p.add_argument('--out',required=True);a=p.parse_args()
    results=[]
    for path in a.paths:
        result=audit(path);results.append(result);print(json.dumps(result),flush=True)
    Path(a.out).write_text(json.dumps(results,indent=2)+'\n')
