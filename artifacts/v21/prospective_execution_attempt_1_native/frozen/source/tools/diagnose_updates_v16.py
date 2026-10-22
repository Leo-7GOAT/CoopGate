"""Show every iterate's development trajectory, without selecting a checkpoint."""
import argparse,hashlib,json
from pathlib import Path
import torch
from coopgate.cooperative_iteration_v16 import load_coupled
from ablate_cooperative_v15 import rollout


def run(folder,out,n=128,seed_base=59400001):
    folder,out=Path(folder),Path(out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    reference,meta=load_coupled(folder/'initial.pt')
    paths=[folder/'initial.pt']+[folder/f'update_{u}.pt' for u in range(1,meta['target_updates']+1)]
    protocol=dict(purpose='Diagnose every intermediate policy; final-checkpoint rule remains unchanged',
        n_initials=n,seed_base=seed_base,backend='Model only, no SUMO invocations',
        models={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');results={}
    for path in paths:
        model,_=load_coupled(path);result=rollout(model,reference,n,seed_base);results[path.name]=result
        print(json.dumps(dict(model=path.name,**{k:v for k,v in result.items() if k!='records'})),flush=True)
    (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('--out',required=True)
    p.add_argument('--n',type=int,default=128);p.add_argument('--seed-base',type=int,default=59400001)
    a=p.parse_args();run(a.folder,a.out,a.n,a.seed_base)
