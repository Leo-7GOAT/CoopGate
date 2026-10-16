"""Execute one preregistered fresh two-stage ego policy training run."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
import train_policy_iteration_v14 as pi
from average_policy_v14 import average

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'artifacts/v14/single/training_protocol.json'


def run(seed,out):
    p=json.loads(PROTOCOL.read_text())
    if seed not in p['training_seeds']:raise ValueError('Seed not preregistered')
    for name,digest in p['training_source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    assert hashlib.sha256((ROOT/p['teacher']).read_bytes()).hexdigest()==p['teacher_sha256']
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(PROTOCOL,out/'training_protocol.json')
    (out/'launch.json').write_text(json.dumps(dict(seed=seed,
        protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()),indent=2)+'\n')
    a,b=p['phase_a'],p['phase_b']
    pi.train(out/'phase_a',seed,a['iterations'],a['states'],a['epochs'],
             p['teacher'],False,p['width'],None,a['latest_only'])
    pi.train(out/'phase_b',seed+b['seed_offset'],b['iterations'],b['states'],b['epochs'],
             p['teacher'],False,p['width'],out/'phase_a/policy.pt',b['latest_only'])
    average(out/'phase_b',out/'averaged',p['final_policy']['trailing_iterations'])
    (out/'complete.json').write_text(json.dumps(dict(seed=seed,
        final_policy_sha256=hashlib.sha256((out/'averaged/policy.pt').read_bytes()).hexdigest(),
        raw_final_sha256=hashlib.sha256((out/'phase_b/policy.pt').read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()),indent=2)+'\n')
    print(json.dumps(dict(seed=seed,status='training_complete',out=str(out))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run(a.seed,a.out)
