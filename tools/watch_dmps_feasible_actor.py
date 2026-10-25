"""Reuse the tested Windows waiter with an explicit new experiment binding.

The retained native handle logic and receipt function stay unchanged. A private
globals mapping supplies this run's frozen auditor and validates its real command.
"""
from datetime import datetime
from types import FunctionType

import verify_dmps_feasible_actor as audit
import watch_dmps_effective_action as waiter

io=audit.io
OUT=audit.run.OUT
OBSERVER=OUT.parent/'dmps_feasible_actor_observer.json'
BINDING=OUT.parent/'dmps_feasible_actor_observer_binding.json'


def validate_identity(start,current,creation_utc):
    if current['ProcessId']!=start['native_pid'] or current['ParentProcessId']!=start['parent_pid']:
        raise ValueError('Original feasible-actor native identity changed')
    if 'tools/train_dmps_feasible_actor.py' not in current['CommandLine'].replace('\\','/'):
        raise ValueError('PID no longer belongs to feasible-actor training')
    delay=(datetime.fromisoformat(start['utc'])-creation_utc).total_seconds()
    if not 0<=delay<=300:raise ValueError('Native creation does not match the original start receipt')


def configured_waiter():
    assert waiter.io is io and audit.run.ROOT==waiter.audit.run.ROOT
    assert audit.run.STEPS==waiter.audit.run.STEPS==32768 and len(audit.run.evaluation_cases())==10
    function=FunctionType(waiter.watch.__code__,{**waiter.watch.__globals__,
        'audit':audit,'OUT':OUT,'OBSERVER':OBSERVER,'validate_identity':validate_identity},
        waiter.watch.__name__,waiter.watch.__defaults__,waiter.watch.__closure__)
    assert function.__code__ is waiter.watch.__code__
    return function


def watch():
    source=audit.run.ROOT/'tools/watch_dmps_feasible_actor.py';io.committed(source)
    plan=io.read(OUT/'protocol.json')
    audit_path=audit.run.ROOT/'tools/verify_dmps_feasible_actor.py'
    assert plan['source_sha256']['tools/verify_dmps_feasible_actor.py']==io.digest(audit_path)
    assert plan['sole_additional_recipe_change']==audit.run.RECIPE
    io.exclusive_json(BINDING,dict(dispatcher_source_sha256=io.digest(source),
        retained_waiter_source_sha256=io.digest(audit.run.ROOT/'tools/watch_dmps_effective_action.py'),
        scientific_auditor_source_sha256=io.digest(audit_path),protocol_sha256=io.digest(OUT/'protocol.json'),
        native_start_sha256=io.digest(OUT/'native_start.json'),expected_native_command='tools/train_dmps_feasible_actor.py',
        private_bindings=['audit','OUT','OBSERVER','validate_identity'],original_waiter_module_mutated=False,
        receipt_function_unchanged=True,scientific_audit_changed_after_collection_binding=False,utc=io.utc()))
    original_audit,original_output=waiter.audit,waiter.OUT
    configured_waiter()()
    assert waiter.audit is original_audit and waiter.OUT==original_output


if __name__=='__main__':watch()
