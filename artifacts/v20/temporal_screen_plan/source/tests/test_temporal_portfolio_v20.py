import numpy as np
import pytest

from coopgate.stress_prediction_v17 import fixed_stress_bank
from coopgate.temporal_portfolio_v20 import BANKS, TemporalPortfolioGate, temporal_stress_bank


@pytest.mark.parametrize('kind,count', [('alternating', 33), ('sampled', 33), ('combined', 49)])
def test_all_old_contract_paths_preserved_and_extra_paths_unique(kind, count):
    old_names, old_tapes = fixed_stress_bank()
    names, tapes = temporal_stress_bank(kind)
    assert names[:17] == old_names
    np.testing.assert_array_equal(tapes[:17], old_tapes)
    assert tapes.shape == (count, 250, 3)
    assert len(set(names)) == count
    assert len({tape.tobytes() for tape in tapes}) == count
    assert np.isfinite(tapes).all() and np.max(np.abs(tapes)) <= .5


def test_temporal_extensions_change_sign_and_respect_pulse_off_periods():
    names, tapes = temporal_stress_bank('combined')
    for name, tape in zip(names[17:], tapes[17:]):
        for vehicle in range(3):
            assert np.any(tape[:, vehicle] > 0) and np.any(tape[:, vehicle] < 0)
        if 'pulse' in name:
            assert np.all(tape[np.arange(250) % 20 < 10] == 0)
    np.testing.assert_array_equal(tapes[17, :10], -tapes[17, 10:20])
    np.testing.assert_array_equal(tapes[25, 10:20], -tapes[25, 30:40])


def test_bank_is_reproducible_independent_of_global_rng_and_combination_is_exact_union():
    np.random.seed(123)
    _, a = temporal_stress_bank('alternating')
    _, b = temporal_stress_bank('sampled')
    names, full = temporal_stress_bank('combined')
    np.random.seed(999)
    repeated_names, repeated = temporal_stress_bank('combined')
    assert names == repeated_names
    np.testing.assert_array_equal(full, repeated)
    np.testing.assert_array_equal(full[:33], a)
    np.testing.assert_array_equal(full[33:], b[17:])
    for start in range(33, 49, 2):
        np.testing.assert_array_equal(full[start], -full[start + 1])


def test_unsupported_bank_is_rejected_before_model_initialization():
    assert len(BANKS) == 3
    for call in (temporal_stress_bank, lambda value: TemporalPortfolioGate(None, value)):
        with pytest.raises(ValueError, match='Unknown frozen temporal bank'):
            call('chosen_from_test_noise')
