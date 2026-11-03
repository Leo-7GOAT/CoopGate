"""Outcome-free progress from explicit owned logs;no process or effect scans."""
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/mappo_ctde_20261101'
FORMAL=BASE/'formal_mappo_attempt5_powered_20261103'


def last_count(path,key):
    if not path.is_file():return 0
    with path.open('rb') as stream:
        stream.seek(max(0,path.stat().st_size-8192));tail=stream.read()
    for line in reversed(tail.splitlines()):
        try:value=json.loads(line)
        except (ValueError,UnicodeError):continue
        if key in value:return int(value[key])
    return 0


def progress():
    stages={}
    for stage,n in (('calibration',28672),('confirmation',16384)):
        stages[stage]={}
        for action,key in (('collect','completed'),('review','reviewed')):
            counts=[last_count(FORMAL/'execution'/f'{stage}_{action}_{i}.log',key) for i in range(4)]
            stages[stage][action]=dict(logged_complete=sum(counts),expected=n,shards=counts)
    random=last_count(BASE/'development/random_proposal_ablation_20261103/worker.log','actual_sumo_executions')
    return dict(primary=stages,random_control_logged_actual_sumo=random,random_control_expected=960,
        free_disk_gib=round(shutil.disk_usage(ROOT).free/1024**3,3),
        log_counts_may_lag_unsealed_work=True,scientific_outcomes_read=False,
        process_absence_not_used_as_completion=True)


if __name__=='__main__':print(json.dumps(progress()),flush=True)
