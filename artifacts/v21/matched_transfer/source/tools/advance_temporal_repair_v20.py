"""Run one paired extension, observing its native workers and partner process."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from extend_temporal_repair_v20 import PARTNER, ROOT, STUDY
from probe_disturbed_gate_v19 import digest, save
from start_final_v19 import require_committed
from study_gate_v19 import read_protocol


def utc():
    return datetime.now(timezone.utc).isoformat()


class NativeWitness:
    """Hold the original kernel object so a recycled PID cannot count as success."""
    def __init__(self, pid):
        if os.name != 'nt':
            raise RuntimeError('Native Windows process observation required')
        self.pid = pid
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.kernel.WaitForSingleObject.restype = wintypes.DWORD
        self.kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        self.kernel.GetExitCodeProcess.restype = wintypes.BOOL
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL
        self.handle = self.kernel.OpenProcess(0x100000 | 0x1000, False, pid)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())

    def poll(self):
        status = self.kernel.WaitForSingleObject(self.handle, 0)
        if status == 258:
            return None
        if status != 0:
            raise ctypes.WinError(ctypes.get_last_error())
        code = wintypes.DWORD()
        if not self.kernel.GetExitCodeProcess(self.handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        return code.value

    def close(self):
        self.kernel.CloseHandle(self.handle)


def ready_to_finalize(worker_codes, partner_code):
    if any(code is not None and code != 0 for code in [*worker_codes, partner_code]):
        raise RuntimeError('Native execution failed; preserve all remaining processes and outputs')
    return bool(worker_codes) and all(code == 0 for code in worker_codes) and partner_code == 0


def run(partner_pid, partner_logs, logs):
    logs, partner_logs = Path(logs).resolve(), Path(partner_logs).resolve()
    if logs.exists() or any(STUDY.glob('shard_*')):
        raise FileExistsError('Never duplicate workers or overwrite logs')
    _, protocol = read_protocol(STUDY)
    require_committed([Path(__file__).resolve(), STUDY / 'protocol.json',
                       *[ROOT / p for p in protocol['source_sha256']]])
    if digest(PARTNER / 'protocol.json') != protocol['partner_protocol_sha256']:
        raise ValueError('Paired protocol changed')
    witness = NativeWitness(partner_pid)
    processes, handles = [], []
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    try:
        logs.mkdir(parents=True)
        for shard in range(protocol['spec']['shards']):
            stdout = (logs / f'worker_{shard}.stdout.log').open('w', encoding='utf-8')
            stderr = (logs / f'worker_{shard}.stderr.log').open('w', encoding='utf-8')
            handles.extend((stdout, stderr))
            process = subprocess.Popen([sys.executable, '-X', 'utf8', 'tools/extend_temporal_repair_v20.py', 'worker', str(shard)],
                                       cwd=ROOT, stdout=stdout, stderr=stderr, creationflags=flags)
            processes.append(process)
        save(logs / 'launch.json', dict(utc=utc(), supervisor_pid=os.getpid(), partner_native_pid=partner_pid,
                                        worker_pids=[p.pid for p in processes],
                                        protocol_sha256=digest(STUDY / 'protocol.json'), driver_sha256=digest(__file__)))
        while True:
            codes, partner_code = [p.poll() for p in processes], witness.poll()
            save(logs / 'state.json', dict(utc=utc(), phase='waiting_native_processes',
                                           worker_pids=[p.pid for p in processes], exit_codes=codes,
                                           partner_native_pid=partner_pid, partner_exit_code=partner_code))
            if partner_code is not None and not (logs / 'partner_native_exit.json').exists():
                save(logs / 'partner_native_exit.json', dict(utc=utc(), pid=partner_pid, exit_code=partner_code))
            if ready_to_finalize(codes, partner_code):
                break
            time.sleep(30)
        save(logs / 'worker_exits.json', dict(utc=utc(), workers=[dict(pid=p.pid, exit_code=p.returncode) for p in processes]))
        state = json.loads((partner_logs / 'state.json').read_text())
        exits = json.loads((partner_logs / 'worker_exits.json').read_text())
        if state['phase'] != 'completed' or any(item['exit_code'] != 0 for item in exits['workers']):
            raise ValueError('Partner completion receipts are missing')
        if state['summary_sha256'] != digest(PARTNER / 'summary.json'):
            raise ValueError('Partner completed summary changed')
        save(logs / 'state.json', dict(utc=utc(), phase='finalizing_and_independently_reviewing'))
        with (logs / 'finalize.stdout.log').open('w', encoding='utf-8') as output:
            result = subprocess.run([sys.executable, '-X', 'utf8', 'tools/extend_temporal_repair_v20.py', 'finalize', str(logs)],
                                    cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, creationflags=flags)
        save(logs / 'finalize_exit.json', dict(utc=utc(), exit_code=result.returncode))
        if result.returncode != 0:
            raise RuntimeError('Paired finalization failed; preserve its diagnostics')
        save(logs / 'state.json', dict(utc=utc(), phase='completed', summary_sha256=digest(STUDY / 'summary.json')))
    except Exception as error:
        if logs.is_dir():
            save(logs / 'failure.json', dict(utc=utc(), type=type(error).__name__, error=str(error),
                                            worker_pids=[p.pid for p in processes], exit_codes=[p.poll() for p in processes]))
        raise
    finally:
        witness.close()
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('partner_pid', type=int)
    parser.add_argument('partner_logs')
    parser.add_argument('logs')
    args = parser.parse_args()
    run(args.partner_pid, args.partner_logs, args.logs)
