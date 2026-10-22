"""Training-only scalar ego-action rollout teacher; both other actors fixed."""
import argparse,json
from pathlib import Path
import numpy as np
from coopgate.single_v12 import SingleRolloutGate
from coopgate.marl_v12 import observations,action_mask,central_state,LEVELS,batch_step
from coopgate.merge_v4 import MergeConfig
from coopgate.fast_v8 import parameters
from develop_v12 import initial
from teacher_v12 import warmstart


def collect(out,n=512):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    c=MergeConfig();packed=parameters(c);obs=[];states=[];masks=[];labels=[];episodes=[];rows=[]
    for index in range(n):
        seed=21000001+index;regime='nominal' if index<n//2 else 'dense'
        x=initial(seed,regime);prev=np.zeros(3);gate=SingleRolloutGate(c)
        rng=np.random.default_rng(seed+1201);cost=0.;tv=0.
        for step in range(50):
            command,info=gate.select(x,prev)
            obs.append(observations(x,prev,step,c)[1:2]);states.append(central_state(x,prev,step,c));masks.append(action_mask(prev,c)[1:2])
            labels.append([int(np.argmin(np.abs(LEVELS-info['ego_proposal'])))]);episodes.append(index)
            nxt,checked,valid=batch_step(x[None],prev[None],command[None],rng.uniform(-.5,.5,(1,5,3)),packed,.5)
            assert valid.all() and np.max(np.abs(checked[0]-command))<1e-10
            cost+=(x[:,0]<c.goal).sum()*c.decision_dt/3.;tv+=np.abs(command-prev).sum();x=nxt[0];prev=command
            if np.all(x[:,0]>=c.goal):break
        rows.append({'seed':seed,'regime':regime,'cost':cost,'variation':tv,'completed':bool(np.all(x[:,0]>=c.goal))})
        if (index+1)%32==0:print(json.dumps({'single_teacher_scenes':index+1,'steps':len(labels)}),flush=True)
    np.savez_compressed(out/'teacher.npz',obs=np.array(obs),state=np.array(states),mask=np.array(masks),actions=np.array(labels),episode=np.array(episodes))
    (out/'metadata.json').write_text(json.dumps({'backend':'exact model, not SUMO','teacher':'single-action rollout','n':n,'steps':len(labels),'rows':rows},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--n',type=int,default=512)
    a=p.parse_args();collect(a.out,a.n)
