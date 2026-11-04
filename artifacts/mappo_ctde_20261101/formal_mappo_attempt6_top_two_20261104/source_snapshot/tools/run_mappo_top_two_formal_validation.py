"""Native-owner64-episode validation;never authorizes formal execution."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
from mappo_top_two_formal_experiment import VALIDATION,ROOT,write,sha
from mappo_native_launch import command as native_command


def run():
    execution=VALIDATION/'execution';execution.mkdir(exist_ok=False);events=[]
    for action in ('collect','review'):
        command=native_command(ROOT/'tools/mappo_top_two_formal_experiment.py',action,'--stage','validation','--validation')
        with (execution/f'{action}.log').open('x',encoding='utf-8') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            event=dict(stage='validation',action=action,native_pid=child.pid,command=command,started_utc=datetime.now(timezone.utc).isoformat())
            with (execution/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(event)+'\n')
            code=child.wait();event.update(actual_native_exit_code=code,finished_utc=datetime.now(timezone.utc).isoformat());events.append(event)
            if code:
                with (execution/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(event)+'\n')
                write(execution/f'{action}_failed_native_exit.json',event)
                print(json.dumps(event),flush=True)
                raise SystemExit(code)
            if action=='collect':
                event.update(plan_sha256=sha(VALIDATION/'plan.json'),collection_complete_sha256=sha(VALIDATION/'validation/shard0/collection_complete.json'))
                write(VALIDATION/'validation/shard0/collect_native_exit.json',event)
            with (execution/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(event)+'\n')
            print(json.dumps(event),flush=True)
    proof=VALIDATION/'validation/shard0/verification.json'
    write(VALIDATION/'verification.json',dict(validation_only=True,formal_samples=0,error_allocated=False,
        actual_sumo_executions=64,full_independent_reviews=64,source_verification_sha256=sha(proof),stages=events))
    write(VALIDATION/'report.json',dict(validation_only=True,formal_samples=0,all_native_exits_zero=True,verification_sha256=sha(VALIDATION/'verification.json')))


if __name__=='__main__':run()
