"""Predefined trailing-window parameter averaging, not best-checkpoint selection.

This postprocessing creates one deployable network. It adds no planning and
does not make a single-vehicle learner a multi-agent system.
"""
import argparse,hashlib,json
from pathlib import Path
import torch


def average(source,out,window=3):
    source,out=Path(source),Path(out)
    final=torch.load(source/'policy.pt',map_location='cpu',weights_only=True)
    if final['algorithm']!='rollout_policy_iteration_ego_v14':raise ValueError('Expected direct actor')
    last=final['iteration']
    if not 1<=window<=last:raise ValueError('Invalid averaging window')
    parents=[source/f'iteration_{i}.pt' for i in range(last-window+1,last+1)]
    states=[torch.load(p,map_location='cpu',weights_only=True)['model'] for p in parents]
    if any(set(s)!=set(states[0]) for s in states):raise ValueError('Inconsistent parameter names')
    averaged={k:torch.stack([s[k].double() for s in states]).mean(0).to(states[0][k].dtype) for k in states[0]}
    meta={k:v for k,v in final.items() if k!='model'}
    meta.update(checkpoint_selection=f'Uniform parameter average of the fixed last {window} iterations; no evaluation-score selection',
                averaging_window=window,averaging_parents={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in parents},
                averaging_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out.mkdir(parents=True,exist_ok=False)
    torch.save({**meta,'model':averaged},out/'policy.pt')
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    (out/'protocol.json').write_text(json.dumps(dict(purpose='development postprocessing, not formal test',
        source=str(source),window=window,parents=meta['averaging_parents']),indent=2)+'\n')
    print(json.dumps(dict(output=str(out/'policy.pt'),iterations=list(range(last-window+1,last+1)),
                         sha256=hashlib.sha256((out/'policy.pt').read_bytes()).hexdigest())),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True);p.add_argument('--window',type=int,default=3)
    a=p.parse_args();average(a.source,a.out,a.window)
