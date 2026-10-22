"""Hold original process handles; analyze only after their successful exit."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time

from probe_disturbed_gate_v19 import digest, save
from start_final_v19 import require_committed
from study_gate_v19 import read_protocol
from study_matched_transfer_v21 import ROOT, STUDY


def utc(): return datetime.now(timezone.utc).isoformat()


def run(mode, logs):
    logs = Path(logs).resolve()
    if logs.exists(): raise FileExistsError('Never replace prior execution evidence')
    if mode == 'study':
        if any(STUDY.glob('shard_*')): raise FileExistsError('Never duplicate transfer workers')
        _, protocol = read_protocol(STUDY)
        require_committed([STUDY / 'protocol.json', *[ROOT / name for name in protocol['source_sha256']]])
        commands = [[sys.executable, '-X', 'utf8', 'tools/study_matched_transfer_v21.py', 'worker', str(shard)]
                    for shard in range(protocol['spec']['shards'])]
        protocol_sha = digest(STUDY / 'protocol.json')
    elif mode == 'preflight':
        commands = [[sys.executable, '-X', 'utf8', 'tools/preflight_matched_transfer_v21.py']]
        protocol_sha = None
    else: raise ValueError('Unknown execution mode')
    logs.mkdir(parents=True)
    processes, handles = [], []
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    try:
        for i, command in enumerate(commands):
            stdout = (logs / f'worker_{i}.stdout.log').open('w', encoding='utf-8')
            stderr = (logs / f'worker_{i}.stderr.log').open('w', encoding='utf-8')
            handles.extend((stdout, stderr))
            processes.append(subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr, creationflags=flags))
        save(logs / 'launch.json', dict(utc=utc(), mode=mode, supervisor_pid=os.getpid(),
            worker_pids=[p.pid for p in processes], commands=commands, protocol_sha256=protocol_sha))
        while True:
            codes = [p.poll() for p in processes]
            save(logs / 'state.json', dict(utc=utc(), phase='waiting_workers', mode=mode,
                                          worker_pids=[p.pid for p in processes], exit_codes=codes))
            if any(code is not None and code != 0 for code in codes):
                raise RuntimeError('Worker failed; preserve outputs and all remaining workers')
            if all(code == 0 for code in codes): break
            time.sleep(30)
        save(logs / 'worker_exits.json', dict(utc=utc(), workers=[dict(pid=p.pid, exit_code=p.returncode) for p in processes]))
        if mode == 'study':
            save(logs / 'state.json', dict(utc=utc(), phase='finalizing', mode=mode))
            with (logs / 'finalize.log').open('w', encoding='utf-8') as output:
                subprocess.run([sys.executable, '-X', 'utf8', 'tools/study_matched_transfer_v21.py', 'finalize', str(logs)],
                               cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, check=True, creationflags=flags)
        save(logs / 'state.json', dict(utc=utc(), phase='completed', mode=mode))
    except Exception as error:
        save(logs / 'failure.json', dict(utc=utc(), type=type(error).__name__, error=str(error),
            worker_pids=[p.pid for p in processes], exit_codes=[p.poll() for p in processes]))
        raise
    finally:
        for handle in handles: handle.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['preflight', 'study'])
    parser.add_argument('logs')
    args = parser.parse_args()
    run(args.mode, args.logs)
