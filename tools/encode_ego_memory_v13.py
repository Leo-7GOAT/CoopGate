"""Re-encode fixed teacher states; no new labels or selected trajectories."""
import argparse,json
from pathlib import Path
import numpy as np
from coopgate.ego_v13 import ego_observations


def run(source,out):
    source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    with np.load(source/'teacher.npz') as file:data={k:file[k] for k in file.files}
    states=data['state'];x=np.stack((states[:,:3]*150,states[:,3:6]*20),axis=-1).astype(np.float64)
    prev=np.round(states[:,6:9]*5/.625)*.625;step=np.rint(states[:,-1]*50).astype(int)
    data['obs']=ego_observations(x,prev,step)[:,1:2]
    np.savez_compressed(out/'teacher.npz',**data)
    meta=json.loads((source/'metadata.json').read_text());meta['observation_version']='markov_joint_memory_v1'
    meta['source']=str(source);(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.source,a.out)
