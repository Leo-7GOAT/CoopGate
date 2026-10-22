"""Runnable bounded-error SUMO demonstration, optionally with its native GUI."""
import argparse
import json
from pathlib import Path
import numpy as np
from coopgate.disturbance_v9 import DisturbedMergeEnv
from coopgate.experiment_v11 import METHODS,prepare
from coopgate.learning_v6 import load_models
from coopgate.merge_v4 import MergeConfig

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gui',action='store_true')
    parser.add_argument('--method',choices=METHODS,default='smooth_learned')
    parser.add_argument('--disturbance',choices=['clean','uniform','bias','pulse'],default='bias')
    parser.add_argument('--scenario',choices=['nominal','dense'],default='dense')
    parser.add_argument('--seed',type=int,default=17000001);parser.add_argument('--out',type=Path,default=Path('runs/v11_demo'))
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    models=load_models(ROOT/'artifacts/v6',[201,211,223]);c=MergeConfig();radius=.5
    warm=np.array([[-80.,12.],[-85.,13.],[-120.,11.]])
    gate,_=prepare(args.method,models,warm,np.zeros(3),c,radius,0);gate.select(warm,np.zeros(3),np.full(3,c.accel))
    print('Controller prewarmed; connecting to SUMO.',flush=True)
    with DisturbedMergeEnv(c,'constant',gui=args.gui) as env:
        env.reset_disturbed(args.seed,args.scenario,args.disturbance,radius)
        env.gate,info=prepare(args.method,models,env.x,env.prev,c,radius,args.seed)
        while True:
            _,_,done,truncated,result=env.step([4]*3)
            if done or truncated:break
        result.update(method=args.method,**info)
        (args.out/'result.json').write_text(json.dumps(result,indent=2)+'\n',newline='\n')
        (args.out/'trace.json').write_text(json.dumps(env.trace,indent=2)+'\n',newline='\n')
        print(json.dumps(result,indent=2))


if __name__=='__main__':main()
