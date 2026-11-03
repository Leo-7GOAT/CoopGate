"""Exercise real owned Win32 exit0 and exit7 fixtures;no simulation outcomes."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import watch_powered_history_pipeline_terminal as watcher

ROOT=Path(__file__).resolve().parents[1]
FIXTURE=ROOT/'.git/pipeline_terminal_watcher_fixture_20261103'
OUT=ROOT/'artifacts/mappo_ctde_20261101/pipeline_terminal_watcher_validation_20261103'


def run():
    FIXTURE.mkdir(exist_ok=False);OUT.mkdir(exist_ok=False);cases=[]
    for expected in (0,7):
        case=FIXTURE/f'exit{expected}';pipeline=case/'pipeline';pipeline.mkdir(parents=True)
        argv=[str(Path(sys._base_executable).resolve()),'-c',f'import time;time.sleep(2);raise SystemExit({expected})']
        child=subprocess.Popen(argv,cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        watcher.write(pipeline/'launch.json',dict(native_pid=child.pid,started_utc=datetime.now(timezone.utc).isoformat(),
            source_sha256=watcher.sha(ROOT/'tools/continue_powered_history_completion.py'),
            primary_plan_sha256=watcher.sha(watcher.BASE/'formal_mappo_attempt5_powered_20261103/plan.json'),
            fixture_only=True))
        watcher.write(pipeline/'report.json',dict(fixture_only=True,not_scientific_results=True))
        watcher.PIPELINE=pipeline;watcher.OUT=case/'observer'
        observed=None
        try:watcher.observe()
        except SystemExit as result:observed=result.code
        finally:actual=child.wait()
        terminal=watcher.read(watcher.OUT/'actual_pipeline_native_terminal.json')
        if observed!=actual or actual!=expected or terminal['actual_native_exit_code']!=expected:
            raise ValueError('Owned Win32 wait disagrees with actual direct native child wait')
        owner=pipeline/'owner_native_exit.json'
        if owner.exists()!=(expected==0):raise ValueError('Failure fixture must not produce success owner')
        if expected==0 and watcher.read(owner)['report_sha256']!=watcher.sha(pipeline/'report.json'):
            raise ValueError('Successful native proof is not bound to exact completed report')
        cases.append(dict(native_pid=child.pid,expected_exit=expected,actual_direct_child_exit=actual,
            actual_windows_handle_exit=terminal['actual_native_exit_code'],success_owner_written=owner.exists(),
            terminal_sha256=watcher.sha(watcher.OUT/'actual_pipeline_native_terminal.json')))
    report=dict(completed=True,cases=cases,all_actual_exits_agree=True,live_production_pipeline_modified=False,
        global_process_enumeration=False,new_sumo_runs=0,new_independent_initials=0,new_error_allocated=False,
        fixture_only=True,watcher_sha256=watcher.sha(ROOT/'tools/watch_powered_history_pipeline_terminal.py'),
        validator_sha256=watcher.sha(__file__),validator_native_pid=os.getpid())
    watcher.write(OUT/'report.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':run()
