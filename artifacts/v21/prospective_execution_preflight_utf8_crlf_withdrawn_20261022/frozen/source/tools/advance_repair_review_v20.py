"""Wait on one already-running native Windows process, then audit its output."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from probe_disturbed_gate_v19 import digest, save
from start_final_v19 import require_committed, verify_manifest
from review_repair_closed_loop_v20 import OUTPUT, PLAN, ROOT, STUDY


def utc():
    return datetime.now(timezone.utc).isoformat()


def observe_native(pid, logs):
    if os.name != 'nt':
        raise RuntimeError('This observer requires a native Windows process handle')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x100000 | 0x1000, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        while True:
            save(logs / 'state.json', dict(utc=utc(), phase='waiting_native_supervisor', observed_pid=pid))
            status = kernel.WaitForSingleObject(handle, 30000)
            if status == 258:
                continue
            if status != 0:
                raise ctypes.WinError(ctypes.get_last_error())
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise ctypes.WinError(ctypes.get_last_error())
            save(logs / 'native_supervisor_exit.json', dict(utc=utc(), observed_pid=pid, exit_code=code.value))
            if code.value != 0:
                raise RuntimeError('The observed native supervisor failed; preserve all outputs')
            return
    finally:
        kernel.CloseHandle(handle)


def run(pid, worker_logs, logs):
    logs, worker_logs = Path(logs).resolve(), Path(worker_logs).resolve()
    if logs.exists() or OUTPUT.exists():
        raise FileExistsError('Never overwrite or duplicate a review observer')
    require_committed([Path(__file__).resolve(), PLAN / 'plan.json'])
    logs.mkdir(parents=True)
    save(logs / 'launch.json', dict(utc=utc(), pid=os.getpid(), observed_pid=pid,
                                    driver_sha256=digest(__file__), plan_sha256=digest(PLAN / 'plan.json')))
    try:
        observe_native(pid, logs)
        state = json.loads((worker_logs / 'state.json').read_text())
        exits = json.loads((worker_logs / 'worker_exits.json').read_text())
        if state['phase'] != 'completed' or any(item['exit_code'] != 0 for item in exits['workers']):
            raise ValueError('Completed native study receipts are required')
        if digest(STUDY / 'summary.json') != state['summary_sha256']:
            raise ValueError('Completed summary identity changed')
        save(logs / 'state.json', dict(utc=utc(), phase='reviewing', observed_pid=pid))
        with (logs / 'review.stdout.log').open('w', encoding='utf-8') as output:
            result = subprocess.run([sys.executable, '-X', 'utf8', 'tools/review_repair_closed_loop_v20.py', 'run'],
                                    cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        save(logs / 'review_exit.json', dict(utc=utc(), exit_code=result.returncode))
        if result.returncode != 0:
            raise RuntimeError('Independent repair review failed; preserve its diagnostics')
        verify_manifest(OUTPUT)
        save(logs / 'state.json', dict(utc=utc(), phase='completed', audit_sha256=digest(OUTPUT / 'audit.json')))
    except Exception as error:
        save(logs / 'failure.json', dict(utc=utc(), type=type(error).__name__, error=str(error)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pid', type=int)
    parser.add_argument('worker_logs')
    parser.add_argument('logs')
    args = parser.parse_args()
    run(args.pid, args.worker_logs, args.logs)
