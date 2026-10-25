"""Retain the real Windows collector handle and dispatch the frozen auditor."""
from datetime import datetime
from types import FunctionType

import verify_timestamped_control_probe as audit
import watch_dmps_effective_action as waiter

io=audit.io
OUT=audit.run.OUT
OBSERVER=OUT.parent/'timestamped_control_probe_observer.json'
BINDING=OUT.parent/'timestamped_control_probe_observer_binding.json'


def validate_identity(start,current,creation_utc):
    if current['ProcessId']!=start['native_pid'] or current['ParentProcessId']!=start['parent_pid']:
        raise ValueError('Timestamp probe native identity differs')
    if 'tools/execute_timestamped_control_probe.py run' not in current['CommandLine'].replace('\\','/'):
        raise ValueError('PID no longer belongs to the timestamp collector')
    delay=(datetime.fromisoformat(start['utc'])-creation_utc).total_seconds()
    if not 0<=delay<=300:raise ValueError('Timestamp collector creation differs from its start receipt')


def successful_receipt(output,identity,exit_code):
    if exit_code!=0:raise RuntimeError('Actual native collector exit was nonzero')
    assert io.digest(output/'protocol.json')==identity['protocol_sha256']
    assert io.digest(output/'native_start.json')==identity['native_start_sha256']
    done=io.read(output/'complete.json')
    assert done['actual_sumo_runs']==117 and len(done['blocks'])==9
    assert done['protocol_sha256']==identity['protocol_sha256']
    return dict(exit_code=exit_code,native_pid=identity['native_pid'],
        native_creation_filetime=identity['native_creation_filetime'],complete_sha256=io.digest(output/'complete.json'),
        proof_source='Retained original Windows process handle',utc=io.utc())


def configured_waiter():
    assert waiter.io is io
    return FunctionType(waiter.watch.__code__,{**waiter.watch.__globals__,
        'audit':audit,'OUT':OUT,'OBSERVER':OBSERVER,'validate_identity':validate_identity,
        'successful_receipt':successful_receipt},waiter.watch.__name__,waiter.watch.__defaults__,waiter.watch.__closure__)


def watch():
    source=audit.run.ROOT/'tools/watch_timestamped_control_probe.py';io.committed(source)
    protocol=io.read(OUT/'protocol.json')
    assert protocol['source_sha256']['tools/watch_timestamped_control_probe.py']==io.digest(source)
    io.exclusive_json(BINDING,dict(dispatcher_source_sha256=io.digest(source),
        retained_waiter_source_sha256=io.digest(audit.run.ROOT/'tools/watch_dmps_effective_action.py'),
        scientific_auditor_source_sha256=io.digest(audit.run.ROOT/'tools/verify_timestamped_control_probe.py'),
        protocol_sha256=io.digest(OUT/'protocol.json'),native_start_sha256=io.digest(OUT/'native_start.json'),
        private_bindings=['audit','OUT','OBSERVER','validate_identity','successful_receipt'],
        native_receipt_requires_all_117_runs=True,module_globals_mutated=False,utc=io.utc()))
    original=waiter.audit
    configured_waiter()()
    assert waiter.audit is original


if __name__=='__main__':watch()
