"""Owned native processes; no review before actual collection terminal zero."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]


def run(out):
    out=Path(out).resolve();execution=out.parent/(out.name+'_execution');execution.mkdir(parents=True,exist_ok=False);events=[]
    for action in ('collect','review'):
        command=[sys.executable,str(ROOT/'tools/evaluate_gain_margin_sumo.py'),action,'--out',str(out)]
        with (execution/f'{action}.log').open('x',encoding='utf-8') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            event=dict(stage=action,pid=child.pid,started_utc=datetime.now(timezone.utc).isoformat(),command=command)
            with (execution/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(event)+'\n')
            code=child.wait();event.update(actual_native_exit_code=code,finished_utc=datetime.now(timezone.utc).isoformat());events.append(event)
            with (execution/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(event)+'\n')
            print(json.dumps(event),flush=True)
        if code:raise SystemExit(code)
    receipt=dict(stages=events,collect_actual_native_exit_code=0,review_actual_native_exit_code=0)
    with (out/'native_receipt.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(receipt,f,indent=2);f.write('\n')
    with (execution/'completion.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(receipt,f,indent=2);f.write('\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);run(p.parse_args().out)
