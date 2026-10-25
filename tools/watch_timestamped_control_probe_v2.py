"""Retained native handle for the explicit model-manifest bootstrap revision."""
from datetime import datetime
from types import FunctionType

import verify_timestamped_control_probe_v2 as audit
import watch_timestamped_control_probe as first

io=audit.io
OUT=audit.run.OUT
OBSERVER=OUT.parent/'timestamped_control_probe_v2_observer.json'
BINDING=OUT.parent/'timestamped_control_probe_v2_observer_binding.json'


def validate_identity(start,current,creation_utc):
    if current['ProcessId']!=start['native_pid'] or current['ParentProcessId']!=start['parent_pid']:
        raise ValueError('V2 collector native identity differs')
    if 'tools/execute_timestamped_control_probe_v2.py run' not in current['CommandLine'].replace('\\','/'):
        raise ValueError('PID no longer belongs to the V2 collector')
    delay=(datetime.fromisoformat(start['utc'])-creation_utc).total_seconds()
    if not 0<=delay<=300:raise ValueError('V2 collector start identity differs')


def configured_waiter():
    original=first.waiter.watch
    assert first.io is io
    return FunctionType(original.__code__,{**original.__globals__,'audit':audit,'OUT':OUT,
        'OBSERVER':OBSERVER,'validate_identity':validate_identity,'successful_receipt':first.successful_receipt},
        original.__name__,original.__defaults__,original.__closure__)


def watch():
    source=audit.run.ROOT/'tools/watch_timestamped_control_probe_v2.py';io.committed(source)
    protocol=io.read(OUT/'protocol.json')
    assert protocol['source_sha256']['tools/watch_timestamped_control_probe_v2.py']==io.digest(source)
    io.exclusive_json(BINDING,dict(dispatcher_source_sha256=io.digest(source),
        retained_waiter_source_sha256=io.digest(audit.run.ROOT/'tools/watch_dmps_effective_action.py'),
        scientific_verifier_adapter_sha256=io.digest(audit.run.ROOT/'tools/verify_timestamped_control_probe_v2.py'),
        original_scientific_verifier_sha256=io.digest(audit.run.ROOT/'tools/verify_timestamped_control_probe.py'),
        protocol_sha256=io.digest(OUT/'protocol.json'),native_start_sha256=io.digest(OUT/'native_start.json'),
        reused_full_117_receipt_function=True,module_globals_mutated=False,utc=io.utc()))
    old=first.waiter.audit
    configured_waiter()()
    assert first.waiter.audit is old


if __name__=='__main__':watch()
