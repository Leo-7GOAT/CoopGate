"""Own child handles and record actual exits before dependent stages."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def run(out):
    out=Path(out).resolve();receipt_dir=out.parent/(out.name+'_execution')
    receipt_dir.mkdir(parents=True,exist_ok=False)
    events=[]
    for action in ('collect','review'):
        command=[sys.executable,str(ROOT/'tools/evaluate_scalar_nominal_sumo.py'),action,'--out',str(out)]
        with (receipt_dir/f'{action}.log').open('x',encoding='utf-8') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            event=dict(stage=action,pid=child.pid,started_utc=datetime.now(timezone.utc).isoformat(),command=command)
            with (receipt_dir/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(event)+'\n')
            code=child.wait();event.update(actual_native_exit_code=code,finished_utc=datetime.now(timezone.utc).isoformat())
            events.append(event)
            with (receipt_dir/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(event)+'\n')
            print(json.dumps(event),flush=True)
        if code!=0:raise SystemExit(code)
    receipt=dict(stages=events,collect_actual_native_exit_code=0,review_actual_native_exit_code=0,
        verification_sha256=__import__('hashlib').sha256((out/'verification.json').read_bytes()).hexdigest())
    with (out/'native_receipt.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(receipt,f,indent=2);f.write('\n')
    with (receipt_dir/'completion.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(receipt,f,indent=2);f.write('\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);run(p.parse_args().out)
