from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from diagnose_temporal_harm_v20 import select_cases, validate_bank


def row(name, loss=0., completed=True):
    return dict(initial_index=1, disturbance='pulse', controller=name,
                per_vehicle_cost=[1., 1., 1. + loss], per_vehicle_finish=[1., 1., 1.],
                completed=completed, collision=False, aborted=False, contract_violations=0)


def test_selects_all_bank_harms_and_failures_with_one_shared_baseline():
    rows = [row('baseline'), row('portfolio0_1802', .5), row('sampled_1802', .1),
            row('combined_1812'), row('alternating_1822', completed=False)]
    cases = select_cases(rows)
    assert [c['controller'] for c in cases] == ['baseline', 'alternating_1822', 'portfolio0_1802', 'sampled_1802']
    assert len(cases) == 4
    with pytest.raises(ValueError, match='Duplicate'):
        select_cases(rows + [deepcopy(rows[0])])


@pytest.mark.parametrize('size', [17, 33, 49])
def test_complete_bank_required_with_no_nan_or_positive_admitted_component(size):
    bank = np.zeros((size, 3, 2))
    np.testing.assert_array_equal(validate_bank({'selected_bank_differences': bank.tolist()}, size), bank)
    with pytest.raises(ValueError):
        validate_bank({'selected_bank_differences': bank[:-1].tolist()}, size)
    for invalid in (float('nan'), .01):
        bank[-1, 2, 0] = invalid
        with pytest.raises(ValueError):
            validate_bank({'selected_bank_differences': bank.tolist()}, size)
