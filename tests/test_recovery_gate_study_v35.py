import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from study_recovery_gate_v35 import MODELS, RULES, integration_groups
from study_actuator_recovery_v34 import PROFILES, SCENES


def source_scope():
    cases = [[si*32+i,scene,i,63400001+si*1000+i,plant,'actuator_recovery']
             for si,scene in enumerate(SCENES) for i in range(32) for plant in PROFILES]
    failures = [[32+i,'dense',i,63401001+i,'lag_strong','baseline'] for i in range(1,11)]
    return cases,failures


def test_integration_preserves_every_coverage_cell_and_old_failure():
    cases,failures = source_scope()
    groups = integration_groups(cases,failures)
    assert len(groups) == 30 and sum(len(g['cases']) for g in groups) == 390
    assert len({g['condition'][0] for g in groups}) == 15
    assert {tuple(g['condition']) for g in groups if g['old_baseline_failed']} == {tuple(k[:5]) for k in failures}
    assert {(g['condition'][1],g['condition'][4]) for g in groups} == {(s,p) for s in SCENES for p in PROFILES}
    methods = {('baseline','none')} | {(r,m) for r in RULES[1:] for m in MODELS}
    for group in groups:
        assert {tuple(k[-2:]) for k in group['cases']} == methods
        assert all(k[:5] == group['condition'] for k in group['cases'])
    assert integration_groups(cases[::-1],failures[::-1]) == groups


@pytest.mark.parametrize('change',('missing_case','duplicate_case','wrong_source','missing_failure',
                                  'duplicate_failure','foreign_failure','wrong_failure_label'))
def test_scope_cannot_drop_or_replace_conditions(change):
    cases,failures = copy.deepcopy(source_scope())
    if change == 'missing_case': cases.pop()
    if change == 'duplicate_case': cases[-1] = cases[0]
    if change == 'wrong_source': cases[0][-1] = 'raw'
    if change == 'missing_failure': failures.pop()
    if change == 'duplicate_failure': failures[-1] = failures[0]
    if change == 'foreign_failure': failures[-1][3] = 0
    if change == 'wrong_failure_label': failures[0][-1] = 'raw'
    with pytest.raises(ValueError): integration_groups(cases,failures)
