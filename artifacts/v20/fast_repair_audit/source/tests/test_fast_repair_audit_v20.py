import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import audit_fast_repair_v20 as audit


def test_case_plan_covers_every_model_mode_harm_slow_tail_and_fallback():
    rows = []
    for i in range(3):
        for mode in ('clean', 'uniform', 'bias', 'pulse'):
            for name in ('baseline', 'repair_1802', 'repair_1812', 'repair_1822'):
                used = name != 'baseline' and i == 1
                rows.append(dict(initial_index=i, disturbance=mode, controller=name,
                                 completed=True, collision=False, aborted=False, contract_violations=0,
                                 per_vehicle_cost=[9.8] * 3 if used else [10.] * 3,
                                 per_vehicle_finish=[9.] * 3, learned_action_steps=int(used),
                                 repair_action_steps=int(used), controller_ms=[10 + 100 * i]))
    harm = next(r for r in rows if r['initial_index'] == 2 and r['disturbance'] == 'pulse' and r['controller'] == 'repair_1802')
    harm.update(per_vehicle_cost=[9., 9., 10.1], learned_action_steps=1, repair_action_steps=1)
    chosen = audit.select_cases(rows)
    keys = {(r['index'], r['mode'], r['method']) for r in chosen}
    assert len(chosen) == len(keys)
    assert (2, 'pulse', 'repair_1802') in keys
    for seed in (1802, 1812, 1822):
        assert {r['mode'] for r in chosen if r['method'] == f'repair_{seed}'} == {'clean', 'uniform', 'bias', 'pulse'}
        assert any(r['index'] == 0 and r['method'] == f'repair_{seed}' for r in chosen)
        assert any(r['index'] == 2 and r['method'] == f'repair_{seed}' for r in chosen)


def rejected_pair():
    original = {key: None for key in audit.IDENTICAL}
    original['portfolio_joint_action_admitted'] = False
    original['performance_model_transitions'] = 10
    original['candidate_evidence'] = [dict(composition=2, weight=.25, command=[0.] * 3,
        checked_scenarios=2, failed_scenarios=1, forecast_error=None,
        differences=[np.zeros((3, 2)).tolist(), [[.5, 0.], [0., 0.], [0., 0.]]])]
    fast = copy.deepcopy(original)
    fast['performance_model_transitions'] = 5
    fast['nominal_pruned_bounds'] = []
    fast['candidate_evidence'][0]['lower_bound_scenarios'] = [1]
    fast['candidate_evidence'][0]['differences'][1][0][0] = .25
    return original, fast


def test_verified_bound_must_be_below_full_value_and_prove_original_rejection():
    original, fast = rejected_pair()
    assert audit.verify_step(original, fast, None, None, 0, None) == (0, 1)
    for wrong in (0., .6):
        altered = copy.deepcopy(fast)
        altered['candidate_evidence'][0]['differences'][1][0][0] = wrong
        with pytest.raises(ValueError, match='rejection bound'):
            audit.verify_step(original, altered, None, None, 0, None)


def test_command_or_accepted_evidence_changes_cannot_hide_behind_speedup():
    original, fast = rejected_pair()
    for key in ('command', 'repair_weight', 'selected_bank_differences'):
        altered = copy.deepcopy(fast)
        altered[key] = 'changed'
        with pytest.raises(AssertionError):
            audit.verify_step(original, altered, None, None, 0, None)


def test_exact_forecasts_cannot_be_relabelled_with_untracked_approximations():
    original, fast = rejected_pair()
    fast['candidate_evidence'][0]['lower_bound_scenarios'] = []
    with pytest.raises(AssertionError):
        audit.verify_step(original, fast, None, None, 0, None)
