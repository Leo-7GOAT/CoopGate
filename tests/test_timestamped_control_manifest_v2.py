from datetime import datetime,timezone

import pytest

import execute_timestamped_control_probe_v2 as revised
import verify_timestamped_control_probe_v2 as verifier
import watch_timestamped_control_probe_v2 as observer


def test_all_actual_development_plans_receive_the_same_pinned_real_model_manifest(monkeypatch):
    models=revised.frozen_models()
    original_read=revised.io.read
    monkeypatch.setattr(revised.io,'read',lambda path:{'models':models} if path==revised.PLAN else original_read(path))
    for index in range(3):
        old,old_group=revised.base.source_case(index)
        assert 'models' not in old
        new,group=revised.source_case(index)
        assert group==old_group and {k:v for k,v in new.items() if k!='models'}==old
        assert new['models']==models
    for item in models.values():
        policy,_=revised.base.load_coupled(revised.ROOT/item['path'])
        assert policy is not None


def test_configured_collect_and_scientific_audit_bodies_are_unchanged():
    assert revised.configured_run().__code__ is revised.base.run.__code__
    assert revised.configured_run().__globals__['source_case'] is revised.source_case
    assert verifier.configured_verify().__code__ is verifier.original.verify.__code__
    assert verifier.configured_verify().__globals__['run'] is revised
    assert observer.configured_waiter().__code__ is observer.first.waiter.watch.__code__


def test_failed_bootstrap_is_preserved_with_zero_sumo_and_nonzero_exit():
    receipt=revised.io.read(revised.base.OUT/'observed_execution_exit.json')
    failure=revised.io.read(revised.base.OUT/'bootstrap_failure.json')
    assert receipt['exit_code']==1 and receipt['actual_sumo_runs']==0
    assert failure['actual_case_records']==0 and revised.no_case_records(revised.base.OUT/'blocks')
    assert not (revised.base.OUT/'complete.json').exists()


def test_observer_rejects_the_prior_failed_command_and_accepts_only_v2():
    created=datetime(2026,10,25,tzinfo=timezone.utc)
    start=dict(native_pid=1,parent_pid=2,utc=created.isoformat())
    body=dict(ProcessId=1,ParentProcessId=2,CommandLine='python tools/execute_timestamped_control_probe.py run')
    with pytest.raises(ValueError):observer.validate_identity(start,body,created)
    body['CommandLine']='python tools/execute_timestamped_control_probe_v2.py run'
    observer.validate_identity(start,body,created)


def test_zero_record_prefix_is_portable_but_never_hides_a_case_file(tmp_path):
    folder=tmp_path/'blocks'
    assert revised.no_case_records(folder)
    folder.mkdir();assert revised.no_case_records(folder)
    (folder/'observations.sqlite').write_bytes(b'case prefix')
    assert not revised.no_case_records(folder)
