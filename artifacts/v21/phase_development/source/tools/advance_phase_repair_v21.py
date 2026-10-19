"""Launch the frozen phase study once and finalize only after original exits."""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import sys
import time

from probe_disturbed_gate_v19 import digest,save
from start_final_v19 import require_committed
from study_gate_v19 import read_protocol
from study_phase_repair_v21 import ROOT,STUDY


def utc():return datetime.now(timezone.utc).isoformat()


def run(logs):
    logs=Path(logs).resolve()
    if logs.exists() or any(STUDY.glob('shard_*')):
        raise FileExistsError('Never duplicate workers or overwrite prior logs')
    _,protocol=read_protocol(STUDY)
    require_committed([STUDY/'protocol.json',*[ROOT/name for name in protocol['source_sha256']]])
    logs.mkdir(parents=True)
    processes,handles=[],[]
    flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
    try:
        for shard in range(protocol['spec']['shards']):
            stdout=(logs/f'worker_{shard}.stdout.log').open('w',encoding='utf-8')
            stderr=(logs/f'worker_{shard}.stderr.log').open('w',encoding='utf-8')
            handles.extend((stdout,stderr))
            processes.append(subprocess.Popen([sys.executable,'-X','utf8','tools/study_phase_repair_v21.py','worker',str(shard)],
                cwd=ROOT,stdout=stdout,stderr=stderr,creationflags=flags))
        save(logs/'launch.json',dict(utc=utc(),supervisor_pid=os.getpid(),worker_pids=[p.pid for p in processes],
                                   protocol_sha256=digest(STUDY/'protocol.json')))
        while True:
            codes=[p.poll() for p in processes]
            save(logs/'state.json',dict(utc=utc(),phase='waiting_workers',worker_pids=[p.pid for p in processes],exit_codes=codes))
            if any(code is not None and code!=0 for code in codes):
                raise RuntimeError('Worker failed; preserve every output and remaining worker')
            if all(code==0 for code in codes):break
            time.sleep(30)
        save(logs/'worker_exits.json',dict(utc=utc(),workers=[dict(pid=p.pid,exit_code=p.returncode) for p in processes]))
        save(logs/'state.json',dict(utc=utc(),phase='finalizing'))
        with (logs/'finalize.log').open('w',encoding='utf-8') as output:
            subprocess.run([sys.executable,'-X','utf8','tools/study_phase_repair_v21.py','finalize',str(logs)],
                cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,check=True,creationflags=flags)
        save(logs/'state.json',dict(utc=utc(),phase='completed',summary_sha256=digest(STUDY/'summary.json')))
    except Exception as error:
        save(logs/'failure.json',dict(utc=utc(),type=type(error).__name__,error=str(error),
            worker_pids=[p.pid for p in processes],exit_codes=[p.poll() for p in processes]))
        raise
    finally:
        for handle in handles:handle.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('logs')
    run(parser.parse_args().logs)
