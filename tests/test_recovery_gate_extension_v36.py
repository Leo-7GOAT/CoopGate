import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from extend_recovery_gate_v36 import full_groups
from study_recovery_gate_v35 import MODELS, RULES, integration_groups
from study_actuator_recovery_v34 import PROFILES, SCENES


def source():
    cases = [[si*32+i,scene,i,63400001+si*1000+i,plant,'actuator_recovery']
             for si,scene in enumerate(SCENES) for i in range(32) for plant in PROFILES]
    failures = [[32+i,'dense',i,63401001+i,'lag_strong','baseline'] for i in range(1,11)]
    return cases,failures


def test_all_conditions_models_and_preflight_cases_repeat_without_selection():
    cases,failures = source()
    groups = full_groups(cases,failures)
    assert len(groups) == 640 and sum(len(g['cases']) for g in groups) == 8320
    assert len({g['condition'][0] for g in groups}) == 160
    methods = {('baseline','none')} | {(r,m) for m in MODELS for r in RULES[1:]}
    all_cases = {tuple(k) for g in groups for k in g['cases']}
    assert len(all_cases) == 8320
    assert {tuple(k) for g in integration_groups(cases,failures) for k in g['cases']} <= all_cases
    assert {tuple(g['condition']) for g in groups if g['old_baseline_failed']} == {tuple(k[:5]) for k in failures}
    for scene in SCENES:
        for plant in PROFILES:
            selected = [g for g in groups if g['condition'][1] == scene and g['condition'][4] == plant]
            assert len(selected) == 32 and {g['condition'][2] for g in selected} == set(range(32))
    for group in groups:
        assert {tuple(k[-2:]) for k in group['cases']} == methods
    assert full_groups(cases[::-1],failures[::-1]) == groups


@pytest.mark.parametrize('change',('missing_case','duplicate_case','wrong_source','missing_failure','foreign_failure'))
def test_reject_incomplete_or_changed_scope(change):
    cases,failures = copy.deepcopy(source())
    if change == 'missing_case': cases.pop()
    if change == 'duplicate_case': cases[-1] = cases[0]
    if change == 'wrong_source': cases[0][-1] = 'raw'
    if change == 'missing_failure': failures.pop()
    if change == 'foreign_failure': failures[0][3] = 0
    with pytest.raises(ValueError): full_groups(cases,failures)
