from pathlib import Path

import pytest

import execute_joint_risk_recovered_20261026 as route


def test_all_scientific_function_bodies_and_original_module_globals_unchanged():
    prior=route.original.predecessor
    script=route.original.SCRIPT
    namespace=route.routed_functions()
    for name in ('bind','verify_previous_attempt','worker','watch','finalize',
                 'bound_inputs','stage_terminal','verify_memoization'):
        assert namespace[name].__code__ is getattr(route.original,name).__code__
        assert namespace[name].__globals__ is namespace
    assert namespace['predecessor'] is route.recovered
    assert namespace['SCRIPT']==route.SCRIPT
    assert namespace['PLAN']==route.original.PLAN and namespace['OUT']==route.original.OUT
    assert route.original.predecessor is prior and route.original.SCRIPT==script


def test_missing_complete_confirmation_refuses_prerequisite_before_reading_effects(tmp_path,monkeypatch):
    namespace=route.routed_functions()
    monkeypatch.setattr(route.recovered,'OUTPUT',tmp_path/'incomplete')
    monkeypatch.setattr(route.original.io,'read',lambda p:pytest.fail('No partial effects may be read'))
    with pytest.raises(ValueError,match='no complete confirmation'):
        namespace['verify_previous_attempt']()


def test_existing_key_adapter_retains_recovered_predecessor_and_original_finalize_code():
    namespace=route.routed_functions()
    compatible=route.keys.compatible_function(namespace['finalize'])
    assert compatible.__code__ is route.original.finalize.__code__
    assert compatible.__globals__['predecessor'] is route.recovered
    assert compatible.__globals__['observations'] is route.keys.list_observations
    assert namespace['observations'] is route.original.observations


def test_both_stages_still_required_before_any_predecessor_effect_read(tmp_path,monkeypatch):
    namespace=route.routed_functions()
    output=tmp_path/'recovered'; end=output/'confirmation/analysis/acceptance.json'
    end.parent.mkdir(parents=True); end.write_text('{}')
    monkeypatch.setattr(route.recovered,'OUTPUT',output)
    monkeypatch.setattr(route.recovered,'configure',lambda:None)
    monkeypatch.setattr(route.recovered,'inputs',lambda:(dict(original_attempt=1),dict(attempt=1)))
    stages=[]
    def complete(plan,stage):
        stages.append(stage)
        if stage=='confirmation':raise ValueError('confirmation native exit absent')
    monkeypatch.setattr(route.recovered.mechanics,'complete_stage',complete)
    monkeypatch.setattr(route.recovered.original,'verify_manifest',lambda p:None)
    monkeypatch.setattr(route.original.io,'read',lambda p:pytest.fail('Full exits precede effects'))
    with pytest.raises(ValueError,match='native exit absent'):
        namespace['verify_previous_attempt']()
    assert stages==['calibration','confirmation']
