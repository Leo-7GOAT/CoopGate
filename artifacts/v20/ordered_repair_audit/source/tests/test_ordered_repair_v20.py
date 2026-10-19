import numpy as np
import pytest

from coopgate.fast_repair_v20 import FastRepairPortfolioGate
from coopgate.ordered_repair_v20 import TemporalFirstRepairGate
from test_fast_repair_v20 import admitted_state, configured


def test_reordering_preserves_every_complete_tape_exactly_and_keeps_nominal_first():
    original, ordered = FastRepairPortfolioGate(None), TemporalFirstRepairGate(None)
    assert ordered.path_order[0] == 0
    assert sorted(ordered.path_order) == list(range(33))
    for index, source in enumerate(ordered.path_order):
        np.testing.assert_array_equal(ordered.tapes[index], original.tapes[source])
        assert ordered.names[index] == original.names[source]


@pytest.mark.parametrize('seed', [1802, 1812, 1822])
@pytest.mark.parametrize('bounded', [False, True])
def test_known_admissions_keep_their_original_canonical_certificate(seed, bounded):
    state, previous, step, item = admitted_state(seed)
    ordered = configured(TemporalFirstRepairGate, step, item, bounded=bounded)
    command, info = ordered.select(state, previous)
    np.testing.assert_array_equal(command, item['command'])
    for key in ('composition', 'repair_weight', 'candidate_count', 'nominal_candidate_count', 'selected_bank_differences'):
        np.testing.assert_equal(info[key], item[key])
    assert info['portfolio_joint_action_admitted'] and len(info['selected_bank_differences']) == 33
    assert info['candidate_evidence'][-1]['evaluated_original_path_indices'] == list(ordered.path_order)
    assert not info['candidate_evidence'][-1]['lower_bound_scenarios']
