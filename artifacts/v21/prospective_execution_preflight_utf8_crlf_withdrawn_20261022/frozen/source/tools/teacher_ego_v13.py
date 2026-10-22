"""Demonstrations optimize the ego objective, not the traffic-wide mean."""
import argparse,json
from pathlib import Path
import numpy as np
from coopgate.ego_v13 import EgoRolloutGate
from coopgate.marl_v12 import observations,central_state,action_mask,LEVELS,batch_step
from coopgate.fast_v8 import parameters
from coopgate.merge_v4 import MergeConfig
from develop_v12 import initial


def collect(out,n=512):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);c=MergeConfig();packed=parameters(c)
    obs=[];states=[];masks=[];labels=[];episodes=[];rows=[]
    for index in range(n):
        seed=24000001+index;regime='nominal' if index<n//2 else 'dense';x=initial(seed,regime);prev=np.zeros(3)
        gate=EgoRolloutGate();rng=np.random.default_rng(seed+1301);cost=0.;variation=0.
        for step in range(50):
            command,info=gate.select(x,prev)
            obs.append(observations(x,prev,step)[1:2]);states.append(central_state(x,prev,step));masks.append(action_mask(prev)[1:2])
            labels.append([int(np.argmin(np.abs(LEVELS-info['ego_proposal'])))]);episodes.append(index)
            nxt,checked,valid=batch_step(x[None],prev[None],command[None],rng.uniform(-.5,.5,(1,5,3)),packed,.5)
            assert valid.all() and np.max(np.abs(checked[0]-command))<1e-10
            cost+=.5;variation+=abs(command[1]-prev[1]);x=nxt[0];prev=command
            if x[1,0]>=c.goal:break
        rows.append({'seed':seed,'regime':regime,'ego_time':cost,'ego_variation':variation,'completed':bool(x[1,0]>=c.goal)})
        if (index+1)%64==0:print(json.dumps({'ego_teacher_scenes':index+1,'transitions':len(labels)}),flush=True)
    np.savez_compressed(out/'teacher.npz',obs=np.array(obs),state=np.array(states),mask=np.array(masks),actions=np.array(labels),episode=np.array(episodes))
    (out/'metadata.json').write_text(json.dumps({'backend':'exact model','objective':'ego time + 0.05 ego variation until arrival',
        'episodes':n,'transitions':len(labels),'rows':rows},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--n',type=int,default=512)
    a=p.parse_args();collect(a.out,a.n)
