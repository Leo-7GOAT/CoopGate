"""One-shot retained Windows process handle -> frozen full training audit.

Collection and scientific audit sources are unchanged. Observer files live
outside the collecting tree; the exit receipt is written only after native exit.
"""
import ctypes
from ctypes import wintypes
from datetime import datetime,timezone
import json
import os
import subprocess

import torch

import verify_dmps_effective_action as audit

io=audit.io
OUT=audit.run.OUT
OBSERVER=OUT.parent/'dmps_effective_action_observer.json'


def validate_identity(start,current,creation_utc):
    if current['ProcessId']!=start['native_pid'] or current['ParentProcessId']!=start['parent_pid']:
        raise ValueError('Original native training identity changed')
    command=current['CommandLine'].replace('\\','/')
    if 'tools/train_dmps_effective_action.py' not in command:
        raise ValueError('PID no longer belongs to the intended training command')
    delay=(datetime.fromisoformat(start['utc'])-creation_utc).total_seconds()
    if not 0<=delay<=300:raise ValueError('Native process creation and start receipt differ')


def successful_receipt(output,identity,exit_code):
    if exit_code!=0:raise RuntimeError('Actual native exit was nonzero: '+str(exit_code))
    if io.digest(output/'protocol.json')!=identity['protocol_sha256']:
        raise ValueError('Collection protocol changed')
    if io.digest(output/'native_start.json')!=identity['native_start_sha256']:
        raise ValueError('Native identity file changed')
    done=io.read(output/'complete.json')
    if done['training_actual_decisions']!=audit.run.STEPS or done['new_policy_conditions']!=10 or done['old_policy_conditions']!=10 or done['baseline_conditions']!=10:
        raise ValueError('Full declared training and every evaluation condition required')
    return dict(exit_code=exit_code,native_pid=identity['native_pid'],
        native_creation_filetime=identity['native_creation_filetime'],
        complete_sha256=io.digest(output/'complete.json'),proof_source='Retained original Windows process handle',utc=io.utc())


def watch():
    if os.name!='nt':raise RuntimeError('This observer requires Windows native process handles')
    io.committed(audit.run.ROOT/'tools/watch_dmps_effective_action.py')
    if (OUT/'observed_execution_exit.json').exists():raise FileExistsError('Preserve original exit receipt')
    start=io.read(OUT/'native_start.json');pid=int(start['native_pid'])
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=(wintypes.DWORD,wintypes.BOOL,wintypes.DWORD);kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.GetProcessTimes.argtypes=(wintypes.HANDLE,*([ctypes.POINTER(wintypes.FILETIME)]*4));kernel.GetProcessTimes.restype=wintypes.BOOL
    kernel.WaitForSingleObject.argtypes=(wintypes.HANDLE,wintypes.DWORD);kernel.WaitForSingleObject.restype=wintypes.DWORD
    kernel.GetExitCodeProcess.argtypes=(wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD));kernel.GetExitCodeProcess.restype=wintypes.BOOL
    kernel.CloseHandle.argtypes=(wintypes.HANDLE,);kernel.CloseHandle.restype=wintypes.BOOL
    handle=kernel.OpenProcess(0x1000|0x100000,False,pid)
    if not handle:raise ctypes.WinError(ctypes.get_last_error())
    try:
        times=[wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle,*[ctypes.byref(t) for t in times]):raise ctypes.WinError(ctypes.get_last_error())
        ticks=(times[0].dwHighDateTime<<32)|times[0].dwLowDateTime
        created=datetime.fromtimestamp(ticks/10_000_000-11644473600,tz=timezone.utc)
        command=f'Get-CimInstance Win32_Process -Filter "ProcessId={pid}" | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress'
        current=json.loads(subprocess.check_output(['powershell','-NoProfile','-NonInteractive','-Command',command],text=True))
        validate_identity(start,current,created)
        identity=dict(native_pid=pid,native_creation_filetime=ticks,native_creation_utc=created.isoformat(),
            native_start_sha256=io.digest(OUT/'native_start.json'),protocol_sha256=io.digest(OUT/'protocol.json'),
            observer_source_sha256=io.digest(audit.run.ROOT/'tools/watch_dmps_effective_action.py'),
            observer_native_pid=os.getpid(),utc=io.utc(),scientific_audit_changed=False)
        io.exclusive_json(OBSERVER,identity);print(json.dumps(dict(observer='attached',native_pid=pid)),flush=True)
        while True:
            status=kernel.WaitForSingleObject(handle,1000)
            if status==0:break
            if status!=258:raise ctypes.WinError(ctypes.get_last_error())
        code=wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)):raise ctypes.WinError(ctypes.get_last_error())
        if code.value!=0:
            io.exclusive_json(OUT/'observed_execution_exit.json',dict(exit_code=code.value,native_pid=pid,
                native_creation_filetime=ticks,proof_source='Retained original Windows process handle',utc=io.utc()))
            raise RuntimeError('Training native process failed; original prefix retained')
        io.exclusive_json(OUT/'observed_execution_exit.json',successful_receipt(OUT,identity,code.value))
    finally:kernel.CloseHandle(handle)
    torch.set_num_threads(1);result=audit.verify()
    io.exclusive_json(OUT/'verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('paired','parameter_l2_changes')}),flush=True)


if __name__=='__main__':watch()
