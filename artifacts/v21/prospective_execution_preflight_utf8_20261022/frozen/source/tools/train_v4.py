import argparse
from coopgate.learning_v4 import train
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--out',required=True)
p.add_argument('--steps',type=int,default=6000);p.add_argument('--bc-episodes',type=int,default=20)
a=p.parse_args();train(a.out,a.seed,a.steps,a.bc_episodes)
