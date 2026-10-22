from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import audit_physical_switch_v27 as audit


def test_schedule_contains_both_rules_and_alternates_order():
    group=dict(cases=[dict(case=[0,'clean','phase05_1802']),dict(case=[0,'clean','phase05_1812'])])
    result=audit.schedule(group)
    assert [r[-1] for r in result]==['mamps_shared','mps_shared','mps_shared','mamps_shared']
    assert len(result)==len(set(result))==4


def test_full_grid_cannot_drop_or_change_a_case():
    base=audit.read(audit.SCOPE)
    audit.validate_scope(base,base['groups'])
    changed=deepcopy(base['groups'])
    changed[0]['cases'].pop()
    with pytest.raises(ValueError,match='Every original'):
        audit.validate_scope(base,changed)
    with pytest.raises(ValueError,match='Every original'):
        audit.validate_scope(base,base['groups'][:-1])


def test_preflight_includes_every_declared_scene_and_plant():
    base=audit.read(audit.SCOPE)
    groups=audit.groups_for(base,'preflight')
    assert len(groups)==6 and sum(len(audit.schedule(g)) for g in groups)==144
    transfers=[g for g in groups if g['study']=='transfer']
    assert {g['scenario'] for g in transfers}==set(base['protocols']['transfer']['spec']['scenarios'])
    for group in transfers:
        assert {c['case'][4] for c in group['cases']}==set(base['protocols']['transfer']['spec']['plants'])
        assert {c['case'][-1] for c in group['cases']}=={'phase05_1802','phase05_1812','phase05_1822'}


def test_adoption_uses_applied_command_indicator_not_mode_label():
    row=dict(per_vehicle_finish=[1.,2.,3.],per_vehicle_variation=[1.,1.,1.])
    trace=[dict(actual_learning_used=False,learning_mode_mask=[True]*3),dict(actual_learning_used=True)]
    result=audit.build_row(row,trace,np.zeros((3,2)),dict(initial_index=0),[0,'clean','phase05_1802'],'mamps_shared')
    assert result['learned_action_steps']==1
    assert result['controller']=='mamps_shared_1802'
    np.testing.assert_array_equal(result['per_vehicle_cost'],[1.05,2.05,3.05])


def test_complete_exit_is_required_before_reviewing_stage():
    plan=dict(actual_sumo_runs=144,blocks=6)
    launch=dict(launcher_pid=10,native_pid=11,plan_sha256='fixed')
    done=dict(plan_sha256='fixed',native_pid=11,actual_sumo_runs=144,blocks=6)
    audit.validate_completion(plan,'fixed',dict(**launch,exit_code=0),launch,done)
    for code in (None,1):
        with pytest.raises(ValueError,match='Successful native exit'):
            audit.validate_completion(plan,'fixed',dict(**launch,exit_code=code),launch,done)
