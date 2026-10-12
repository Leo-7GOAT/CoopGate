import argparse
import json
from pathlib import Path

from .config import Config
from .experiment import evaluate, render_trace
from .mappo import train
from .shield import Supervisor
from .sim import SumoEnv


def main():
    parser=argparse.ArgumentParser(description='CoopGate: real SUMO experiments')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('smoke');p.add_argument('--gui',action='store_true');p.add_argument('--seed',type=int,default=3)
    p=sub.add_parser('train');p.add_argument('--steps',type=int,default=20000);p.add_argument('--seed',type=int,default=7);p.add_argument('--out',default='runs/train')
    p=sub.add_parser('evaluate');p.add_argument('--checkpoint',required=True);p.add_argument('--out',default='runs/eval');p.add_argument('--seeds',type=int,default=10);p.add_argument('--start-seed',type=int,default=10000);p.add_argument('--methods',nargs='+',choices=Supervisor.methods);p.add_argument('--scenarios',nargs='+',choices=('nominal','dense','braking'),default=['nominal','dense','braking']);p.add_argument('--gui',action='store_true');p.add_argument('--threshold',type=float,default=.8);p.add_argument('--horizon',type=int,default=10)
    p=sub.add_parser('render');p.add_argument('trace');p.add_argument('output')
    args=parser.parse_args()
    if args.command=='train':
        train(args.out,args.steps,args.seed)
    elif args.command=='evaluate':
        cfg=Config(gate_threshold=args.threshold,gate_return_threshold=min(.99,args.threshold+.1),gate_horizon=args.horizon)
        evaluate(args.checkpoint,args.out,args.seeds,args.start_seed,args.methods,args.scenarios,cfg,args.gui)
    elif args.command=='render':
        render_trace(args.trace,args.output)
    else:
        with SumoEnv(method='traditional',gui=args.gui) as env:
            env.reset(args.seed)
            done=False
            while not done:
                _,_,term,trunc,info=env.step([4,4,4])
                done=term or trunc
            print(json.dumps(info,indent=2))


if __name__=='__main__':
    main()
