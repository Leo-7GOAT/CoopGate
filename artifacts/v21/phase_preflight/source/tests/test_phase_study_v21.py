import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from study_phase_repair_v21 import MODELS,REFERENCE_KEYS,methods,paired_checks,phase_record_check


def fixture():
    spec=dict(n_initials=2,seed_base=63300001,modes=['clean','pulse'],methods=methods(),
              mode_selection_stream=21107,bootstrap_stream=21113,bootstrap_samples=50)
    rows=[]
    for i in range(2):
        for mode in spec['modes']:
            for name in spec['methods']:
                row={k:0 for k in REFERENCE_KEYS}
                row.update(initial_index=i,disturbance=mode,controller=name,team_cost=10.,learned_action_steps=0,
                           forecast_error_count=0,generation_error_count=0)
                rows.append(row)
    return rows,spec


def test_all_models_both_margins_and_original_reference_are_preserved():
    entries=methods()
    assert len(entries)==13
    for seed in MODELS:
        assert entries[f'phase05_{seed}']['phase_tolerance_seconds']==.05
        assert entries[f'phase10_{seed}']['phase_tolerance_seconds']==.1
        assert entries[f'reference_{seed}']['phase_tolerance_seconds'] is None
        assert entries[f'repair_{seed}']['variant']=='original'


def test_partial_grid_and_reference_changes_are_rejected():
    rows,spec=fixture()
    assert paired_checks(rows,spec)['1802']['reference_pairs']==4
    for bad in (rows[:-1],rows+[rows[0]]):
        with pytest.raises(ValueError,match='Missing or duplicated'):
            paired_checks(bad,spec)
    changed=copy.deepcopy(rows)
    next(r for r in changed if r['controller']=='reference_1802')['team_cost']=9.
    with pytest.raises(AssertionError):paired_checks(changed,spec)


def test_stricter_policy_cannot_gain_an_episode_when_reference_never_adopts():
    rows,spec=fixture()
    next(r for r in rows if r['controller']=='phase05_1812')['learned_action_steps']=1
    with pytest.raises(ValueError,match='stricter'):paired_checks(rows,spec)


def test_missing_or_failed_phase_path_cannot_count_as_valid_admission():
    method=methods()['phase05_1802']
    item=dict(phase_tolerance_seconds=.05,phase_filter_enabled=True,portfolio_joint_action_admitted=True,
              selected_phase_differences=np.zeros((33,3)).tolist())
    phase_record_check(item,method)
    for values in (np.zeros((32,3)),np.full((33,3),.051),np.full((33,3),np.nan)):
        bad={**item,'selected_phase_differences':values.tolist()}
        with pytest.raises(ValueError,match='complete valid'):phase_record_check(bad,method)
    with pytest.raises(ValueError,match='Fallback'):
        phase_record_check({**item,'portfolio_joint_action_admitted':False},method)
