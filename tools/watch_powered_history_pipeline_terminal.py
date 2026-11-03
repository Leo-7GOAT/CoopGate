"""Persist the ACTUAL exit of the already owned pipeline via its Win32 handle.

No global process enumeration,no process mutation and no scientific effect
inspection. PID,creation time,image and source binding identify the existing
owned process. An open kernel handle survives later desktop observation loss.
"""
import argparse
import ctypes
from ctypes import wintypes as w
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/mappo_ctde_20261101'
PIPELINE=BASE/'powered_history_completion_pipeline_20261103'
OUT=BASE/'powered_history_pipeline_terminal_watcher_20261103'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
def now():return datetime.now(timezone.utc).isoformat()


def api():
    if os.name!='nt':raise RuntimeError('Existing Windows process only')
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[w.DWORD,w.BOOL,w.DWORD];kernel.OpenProcess.restype=w.HANDLE
    kernel.GetProcessTimes.argtypes=[w.HANDLE,ctypes.POINTER(w.FILETIME),ctypes.POINTER(w.FILETIME),ctypes.POINTER(w.FILETIME),ctypes.POINTER(w.FILETIME)]
    kernel.GetProcessTimes.restype=w.BOOL
    kernel.QueryFullProcessImageNameW.argtypes=[w.HANDLE,w.DWORD,w.LPWSTR,ctypes.POINTER(w.DWORD)]
    kernel.QueryFullProcessImageNameW.restype=w.BOOL
    kernel.WaitForSingleObject.argtypes=[w.HANDLE,w.DWORD];kernel.WaitForSingleObject.restype=w.DWORD
    kernel.GetExitCodeProcess.argtypes=[w.HANDLE,ctypes.POINTER(w.DWORD)];kernel.GetExitCodeProcess.restype=w.BOOL
    kernel.CloseHandle.argtypes=[w.HANDLE];kernel.CloseHandle.restype=w.BOOL
    return kernel


def attach(kernel):
    launch=read(PIPELINE/'launch.json')
    if (launch['source_sha256']!=sha(ROOT/'tools/continue_powered_history_completion.py')
            or launch['primary_plan_sha256']!=sha(BASE/'formal_mappo_attempt5_powered_20261103/plan.json')):
        raise ValueError('Existing owned pipeline source/plan binding changed')
    handle=kernel.OpenProcess(0x00100000|0x1000,False,int(launch['native_pid']))
    if not handle:raise ctypes.WinError(ctypes.get_last_error())
    try:
        created,exited,cpu_kernel,cpu_user=(w.FILETIME() for _ in range(4))
        if not kernel.GetProcessTimes(handle,ctypes.byref(created),ctypes.byref(exited),ctypes.byref(cpu_kernel),ctypes.byref(cpu_user)):
            raise ctypes.WinError(ctypes.get_last_error())
        ticks=(created.dwHighDateTime<<32)|created.dwLowDateTime
        created_timestamp=ticks/10000000-11644473600
        launched_timestamp=datetime.fromisoformat(launch['started_utc']).timestamp()
        if not 0<=launched_timestamp-created_timestamp<=30:
            raise ValueError('PID creation time does not match the already owned launch')
        buffer=ctypes.create_unicode_buffer(32768);size=w.DWORD(len(buffer))
        if not kernel.QueryFullProcessImageNameW(handle,0,buffer,ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        if Path(buffer.value).resolve()!=Path(sys._base_executable).resolve():
            raise ValueError('Actual process image differs from the registered Python runtime')
        identity=dict(native_pid=launch['native_pid'],creation_filetime_ticks=ticks,
            created_utc=datetime.fromtimestamp(created_timestamp,timezone.utc).isoformat(),
            image=buffer.value,launch_sha256=sha(PIPELINE/'launch.json'),
            pipeline_source_sha256=launch['source_sha256'],identity_verified=True)
        return handle,identity
    except BaseException:
        kernel.CloseHandle(handle);raise


def observe(check=False):
    kernel=api();handle,identity=attach(kernel)
    try:
        if check:
            status=kernel.WaitForSingleObject(handle,0)
            if status not in (0,258):raise ctypes.WinError(ctypes.get_last_error())
            print(json.dumps(dict(**identity,actual_handle_wait_state='live' if status==258 else 'terminal',
                no_process_mutation=True,no_scientific_effects_read=True)),flush=True)
            return
        OUT.mkdir(exist_ok=False)
        write(OUT/'attach.json',dict(**identity,watcher_native_pid=os.getpid(),attached_utc=now(),
            watcher_source_sha256=sha(__file__),owned_kernel_handle_opened=True))
        while True:
            status=kernel.WaitForSingleObject(handle,30000)
            if status==0:break
            if status!=258:raise ctypes.WinError(ctypes.get_last_error())
            print(json.dumps(dict(state='owned_pipeline_handle_still_live',native_pid=identity['native_pid'],utc=now())),flush=True)
        code=w.DWORD()
        if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        terminal=dict(**identity,actual_native_exit_code=int(code.value),observed_by_owned_windows_process_handle=True,
            wait_result='WAIT_OBJECT_0',observed_utc=now(),watcher_source_sha256=sha(__file__))
        write(OUT/'actual_pipeline_native_terminal.json',terminal)
        if code.value==0:
            terminal['report_sha256']=sha(PIPELINE/'report.json')
            owner=PIPELINE/'owner_native_exit.json'
            if owner.exists():
                existing=read(owner)
                if existing['actual_native_exit_code']!=0 or existing['report_sha256']!=terminal['report_sha256']:
                    raise ValueError('Retain disagreeing prior terminal evidence')
            else:write(owner,terminal)
        write(OUT/'report.json',dict(completed=True,**terminal,new_sumo_runs=0,
            no_process_mutation=True,global_process_enumeration=False))
        print(json.dumps(terminal),flush=True)
        raise SystemExit(code.value)
    finally:kernel.CloseHandle(handle)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args();observe(args.check)
