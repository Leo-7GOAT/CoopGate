import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import audit_ordered_repair_v20 as audit


def test_rejection_requires_a_true_full_forecast_violation():
    audit.verify_witness([[.25, 0.]], [[.5, 0.]], True)
    audit.verify_witness([[.5, 0.]], [[.5, 0.]], False)
    with pytest.raises(ValueError, match='exceeds'):
        audit.verify_witness([[.6, 0.]], [[.5, 0.]], True)
    with pytest.raises(ValueError, match='violating'):
        audit.verify_witness([[-.1, 0.]], [[0., 0.]], True)
    with pytest.raises(ValueError, match='finite'):
        audit.verify_witness([[np.nan, 0.]], [[.5, 0.]], True)
    with pytest.raises(AssertionError):
        audit.verify_witness([[.25, 0.]], [[.5, 0.]], False)


def accepted_pair():
    original = {key: None for key in audit.IDENTICAL}
    values = -np.arange(33 * 6, dtype=float).reshape(33, 3, 2)
    original['portfolio_joint_action_admitted'] = True
    original['selected_bank_differences'] = values.tolist()
    evidence = dict(composition=2, weight=.25, command=[0.] * 3, checked_scenarios=33,
                    failed_scenarios=0, forecast_error=None, differences=values.tolist())
    original['candidate_evidence'] = [evidence]
    item = copy.deepcopy(original)
    order = [0, *range(17, 33), *range(1, 17)]
    item['forecast_path_order'] = order
    item['nominal_pruned_bounds'] = []
    item['candidate_evidence'][0].update(evaluated_original_path_indices=order,
        lower_bound_scenarios=[], differences=values[order].tolist())
    return original, item


def test_complete_reordered_certificate_is_compared_in_original_path_order():
    original, item = accepted_pair()
    assert audit.verify_step(original, item, None, None, 0, None, None, {}) == (0, 0, 0)
    item['candidate_evidence'][0]['differences'][-1][0][0] -= .1
    with pytest.raises(AssertionError):
        audit.verify_step(original, item, None, None, 0, None, None, {})


def test_no_path_can_be_deleted_duplicated_or_relabelled():
    original, item = accepted_pair()
    item['forecast_path_order'][-1] = item['forecast_path_order'][-2]
    with pytest.raises(ValueError, match='changed its paths'):
        audit.verify_step(original, item, None, None, 0, None, None, {})
    original, item = accepted_pair()
    item['candidate_evidence'][0]['evaluated_original_path_indices'] = list(range(33))
    with pytest.raises(ValueError, match='path identities'):
        audit.verify_step(original, item, None, None, 0, None, None, {})


def test_partial_or_bounded_evidence_cannot_pass_as_a_full_admission():
    original, item = accepted_pair()
    item['candidate_evidence'][0]['lower_bound_scenarios'] = [17]
    with pytest.raises(ValueError, match='complete exact'):
        audit.verify_step(original, item, None, None, 0, None, None, {})
