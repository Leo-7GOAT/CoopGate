"""Three fixed seeds; fixed final checkpoint, no test-based seed selection."""
import argparse,hashlib,json,time
from pathlib import Path
from teacher_v12 import warmstart
from train_v12 import train


def run(out,data):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    sources=['src/coopgate/marl_v12.py','src/coopgate/training_v12.py','src/coopgate/single_v12.py',
             'tools/train_v12.py','tools/teacher_v12.py','tools/train_single_release_v12.py']
    protocol={'version':'v12-single-training','seeds':[411,421,431],'steps_per_seed':1048576,
        'bc_epochs':80,'variation_weight':.05,'selection':'fixed final checkpoint at 1048576 environment transitions',
        'training_backend':'vectorized model; checked against SUMO separately',
        'teacher_scenes':512,'teacher_sha256':hashlib.sha256((Path(data)/'teacher.npz').read_bytes()).hexdigest(),
        'source_sha256':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},
        'single_learner_vehicle':1,'other_vehicle_proposals':'fixed maximum admissible acceleration',
        'safety_filter':'shared centralized joint certificate for all methods'}
    (out/'training_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for seed in protocol['seeds']:
        warmstart(data,out/f'bc_{seed}',seed,80)
        train(out/f'ppo_{seed}',seed,1048576,variation_weight=.05,warmstart=out/f'bc_{seed}/bc.pt',single=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--data',default='runs/v12_single_teacher')
    a=p.parse_args();run(a.out,a.data)
