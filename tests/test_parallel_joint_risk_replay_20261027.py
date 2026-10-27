from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import parallel_joint_risk_replay_20261027 as replay


def entry(case,record,model=None,failed=False):
    return dict(argument_sha256=replay.argument_key(case,record,model),
                review=dict(recorded_failure=failed,physical_substeps=5))


def test_cache_requires_exact_case_full_record_model_and_single_use():
    case,record={'seed':4},{'trace':[{'command':[1,2,3]}]}
    model=torch.nn.Linear(2,1).eval()
    item=entry(case,record,model,failed=True)
    lookup=replay.AuditLookup([dict(reviews=[item])])
    with pytest.raises(ValueError,match='Every original'):lookup.require_complete()
    changed=deepcopy(record);changed['trace'][0]['command'][0]=0
    with pytest.raises(ValueError,match='arguments differ'):lookup.review(case,changed,model)
    with pytest.raises(ValueError,match='arguments differ'):lookup.review({'seed':5},record,model)
    altered=deepcopy(model)
    with torch.no_grad():altered.weight.add_(1)
    with pytest.raises(ValueError,match='arguments differ'):lookup.review(case,record,altered)
    assert lookup.review(case,record,deepcopy(model))==item['review']
    lookup.require_complete()
    with pytest.raises(ValueError,match='twice'):lookup.review(case,record,model)


def test_duplicate_inputs_cannot_mask_missing_coverage():
    item=entry({}, {})
    with pytest.raises(ValueError,match='Duplicate'):
        replay.AuditLookup([dict(reviews=[item,item])])


def test_original_finalizer_code_and_original_modules_are_unchanged():
    function=replay.original.finalize
    before=replay.original.collect.audit_record
    adapted=replay.adapted_finalizer(function,replay.AuditLookup([]))
    assert adapted.__code__ is function.__code__
    assert replay.original.collect.audit_record is before
    assert function.__globals__['collect'] is replay.original.collect
    assert adapted.__globals__['collect'] is not replay.original.collect
    assert adapted.__globals__['observations'] is replay.keys.list_observations


def test_inference_cannot_run_with_unused_audits_or_failed_preinference_guard():
    lookup=replay.AuditLookup([dict(reviews=[entry({}, {})])])
    function=replay.adapted_finalizer(replay.original.finalize,lookup)
    infer=function.__globals__['certificate_from_complete_records']
    with pytest.raises(ValueError,match='Every original'):infer({},[],[])
    lookup.review({}, {},None)
    def failure():raise ValueError('registration changed')
    function=replay.adapted_finalizer(replay.original.finalize,lookup,before_inference=failure)
    with pytest.raises(ValueError,match='registration changed'):
        function.__globals__['certificate_from_complete_records']({},[],[])


def test_unregistered_worker_count_is_rejected_before_any_process():
    with pytest.raises(ValueError,match='registered worker'):replay.replay_blocks([],{},workers=2)


def test_missing_registration_blocks_any_model_or_record_access(tmp_path,monkeypatch):
    monkeypatch.setattr(replay,'PLAN',tmp_path/'missing.json')
    monkeypatch.setattr(replay.original.io,'committed',lambda p: (_ for _ in ()).throw(ValueError('missing registration')))
    with pytest.raises(ValueError,match='missing registration'):replay.finalize({})


def test_both_native_stages_precede_pool_creation_and_output(tmp_path,monkeypatch):
    monkeypatch.setattr(replay,'bound_registration_inputs',lambda:{})
    monkeypatch.setattr(replay.original,'OUT',tmp_path/'unstarted')
    monkeypatch.setattr(replay.original.io,'native_snapshot',lambda:[])
    monkeypatch.setattr(replay.original.io,'digest',lambda p:'hash')
    stages=[]
    def terminal(stage,*args):
        stages.append(stage)
        if stage=='confirmation':raise ValueError('confirmation actual exit missing')
    namespace=dict(bound_inputs=lambda:({},{}),require_quiet=lambda p:None,
                   stage_terminal=terminal)
    monkeypatch.setattr(replay,'replay_blocks',lambda *a:pytest.fail('Pool before native completion'))
    with pytest.raises(ValueError,match='actual exit missing'):replay.finalize(namespace)
    assert stages==['calibration','confirmation']
    assert not (tmp_path/'unstarted').exists()


def test_recovered_execution_retains_all_original_scientific_function_bodies():
    from types import FunctionType
    import execute_joint_risk_recovered_20261027 as route
    namespace=route.routed_functions()
    for name,function in vars(route.original).items():
        if isinstance(function,FunctionType) and function.__module__==route.original.__name__:
            assert namespace[name].__code__ is function.__code__
            assert namespace[name].__globals__ is namespace
    assert namespace['predecessor'] is route.recovered
