import argparse
from coopgate.learning_v3 import train
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--out',required=True);p.add_argument('--steps',type=int,default=12000)
a=p.parse_args();train(a.out,a.seed,a.steps)
