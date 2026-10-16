"""Matched API training with explicit current-command arrival-time features."""
import argparse
import train_certificate_policy_v14 as trainer


def train(out,seed,iterations,states,epochs,width,rule_only):
    # Reuse the SAME optimization, exploration, and bookkeeping as the physical
    # feature ablation; only the model and its feature/return functions change.
    trainer.train(out,seed,iterations,states,epochs,width,rule_only,variant='kinematic')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=809)
    p.add_argument('--iterations',type=int,default=6);p.add_argument('--states',type=int,default=8192)
    p.add_argument('--epochs',type=int,default=80);p.add_argument('--width',type=int,default=64);p.add_argument('--rule-only',action='store_true')
    a=p.parse_args();train(a.out,a.seed,a.iterations,a.states,a.epochs,a.width,a.rule_only)
