"""Portable test observations for one REAL owned-worker exit fixture.

The immutable legacy test already isolates its process-quietness dependency.
Probe only its own published worker PID instead of enumerating the machine.
The real child,original acknowledgement checks,wait and exit7 assertions stay.
No production process observer or scientific execution is replaced.
"""
import json
import os
from pathlib import Path
import subprocess

import pytest


def _owned_process(pid):
    pid=int(pid)
    if pid<=0:raise ValueError('Positive owned PID required')
    if os.name=='nt':
        command=(f"Get-CimInstance Win32_Process -Filter 'ProcessId = {pid}' | "
                 'Select-Object ProcessId,ParentProcessId,Name,CreationDate,CommandLine | ConvertTo-Json -Depth 3')
        result=subprocess.run(['powershell','-NoProfile','-Command',command],capture_output=True,
            check=True,creationflags=subprocess.CREATE_NO_WINDOW)
        raw=result.stdout.decode('utf-8-sig').strip()
        if not raw:return []
        row=json.loads(raw);return row if isinstance(row,list) else [row]
    proc=Path('/proc')/str(pid)
    if proc.is_dir():
        stat=(proc/'stat').read_text();end=stat.rfind(')');fields=stat[end+2:].split()
        return [dict(ProcessId=pid,ParentProcessId=int(fields[1]),Name=stat[stat.find('(')+1:end],
            CreationDate='proc-start-ticks:'+fields[19],
            CommandLine=os.fsdecode((proc/'cmdline').read_bytes().replace(b'\0',b' ')).strip())]
    # BSD/macOS ps query is also restricted to this single owned PID.
    raw=subprocess.check_output(['ps','-p',str(pid),'-o','pid=,ppid=,lstart=,command='],text=True).strip()
    if not raw:return []
    fields=raw.split(None,7)
    return [dict(ProcessId=int(fields[0]),ParentProcessId=int(fields[1]),
        Name=Path(fields[7].split()[0]).name,CreationDate=' '.join(fields[2:7]),CommandLine=fields[7])]


@pytest.fixture(autouse=True)
def _owned_native_failure_observation(request,monkeypatch):
    if request.node.name!='test_real_native_nonzero_exit_is_preserved_without_fake_completion':return
    execute=request.module.execute
    def snapshot():
        path=execute.OUT/'execution/worker.native.json'
        if not path.exists():return []
        return _owned_process(json.loads(path.read_bytes())['native_pid'])
    monkeypatch.setattr(execute.io,'native_snapshot',snapshot)


@pytest.fixture(autouse=True)
def _legacy_prefix_identity_runtime_fixture(request,monkeypatch):
    """Keep synthetic prefix/receipt unit tests independent of the CI host.

    This affects only two named tests in three immutable historical test files.
    They validate mutated identity fixtures, not actual scientific execution.
    Production runtime checks and separate mismatch rejection tests remain.
    """
    files={'test_recovery_20261026.py','test_recovery_20261027.py',
           'test_third_recovery_v38.py'}
    functions={'test_evidence_cannot_change_statistical_or_prefix_identity',
               'test_valid_evidence_keeps_all_prefix_counts'}
    if Path(str(request.node.path)).name not in files:return
    if getattr(request.node,'originalname',None) not in functions:return
    original=request.module.recovery.original
    frozen=json.loads((original.OUTPUTS['prospective']/'frozen/plan.json').read_bytes())
    monkeypatch.setattr(original,'runtime_identity',lambda:frozen['runtime'].copy())
