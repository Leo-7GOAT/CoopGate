"""Behavior-cloning initialized MAPPO; all gates later share the same checkpoint."""
import argparse
from pathlib import Path
import csv,json,time
import numpy as np
import torch
from coopgate.config import Config,ACTION_SPEEDS
from coopgate.controllers import backup_intents
from coopgate.mappo import MAPPO,train
from coopgate.sim import SumoEnv


def expert_actions(states,cfg):
    actions=[]
    for s,intent in zip(states[:3],backup_intents(states,cfg)):
        lane=int(s[1]>1.6)
        direction=int(np.clip(intent.lane-lane,-1,1))+1
        actions.append(3*direction+int(np.argmin(np.abs(np.array(ACTION_SPEEDS)-intent.speed))))
    return np.array(actions)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--steps',type=int,default=30000)
    p.add_argument('--out',required=True)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    cfg=Config(commit_lane_changes=True,continuous_network=True,gate_budget_ms=10000)
    torch.set_num_threads(1);torch.manual_seed(a.seed)
    data=[];labels=[]
    with SumoEnv(cfg,'traditional') as env:
        for k in range(90):
            obs=env.reset(1000+a.seed*100+k,('nominal','dense','braking')[k%3])
            while True:
                data.extend(obs);labels.extend(expert_actions(env.states,cfg))
                obs,_,t,u,_=env.step([4,4,4])
                if t or u:break
    model=MAPPO();optimizer=torch.optim.Adam(model.actor.parameters(),lr=1e-3)
    x=torch.tensor(np.array(data),dtype=torch.float32);y=torch.tensor(np.array(labels),dtype=torch.long)
    losses=[]
    for epoch in range(20):
        for ids in torch.randperm(len(y)).split(256):
            loss=torch.nn.functional.cross_entropy(model.actor(x[ids]),y[ids])
            optimizer.zero_grad();loss.backward();optimizer.step()
        losses.append(float(loss.detach()))
    warm=out/'bc.pt'
    torch.save({'model':model.state_dict(),'seed':a.seed,'steps':0,'config':cfg.dict()},warm)
    (out/'bc.json').write_text(json.dumps({'episodes':90,'samples':len(y),'epochs':20,'losses':losses},indent=2))
    print(json.dumps({'seed':a.seed,'stage':'BC complete','samples':len(y)}),flush=True)
    train(out,a.steps,a.seed,cfg=cfg,warmstart=warm,bc_weight=.1)


if __name__=='__main__':main()
