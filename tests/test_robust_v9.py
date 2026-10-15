import itertools
import numpy as np
import pytest
pytest.importorskip('numba')
from coopgate.merge_v4 import MergeConfig, admissible, transition_safe
from coopgate.fast_v8 import parameters, certificate, plan_fast
from coopgate.robust_v9 import interval_substep, interval_transition, robust_certificate, RobustGate
from coopgate.disturbance_v9 import disturbance_tape


def test_interval_encloses_arbitrary_substep_errors_and_certified_transitions():
    rng = np.random.default_rng(91601); c = parameters(MergeConfig())
    for _ in range(500):
        lo = np.column_stack((rng.uniform(-130, 60, 3), rng.uniform(0, 18, 3)))
        hi = lo+rng.uniform(0, 2, (3, 2)); u = rng.uniform(-5, 2.5, 3)
        qlo, qhi = interval_substep(lo, hi, u, c, .5)
        safe = interval_transition(lo, hi, qlo, qhi, c)
        for _ in range(5):
            actual = rng.uniform(lo, hi); nxt = actual.copy()
            nxt[:, 1] = np.clip(actual[:, 1]+(u+rng.uniform(-.5, .5, 3))*.1, 0, 20)
            nxt[:, 0] += nxt[:, 1]*.1
            assert np.all(nxt >= qlo-1e-12) and np.all(nxt <= qhi+1e-12)
            if safe:
                assert transition_safe(actual[:, 0][None], nxt[:, 0][None], MergeConfig())[0]


def test_zero_radius_matches_original_certificate_and_full_profile_selection():
    rng = np.random.default_rng(91602); config = MergeConfig(); c = parameters(config)
    for _ in range(300):
        x = np.column_stack((rng.uniform(-140, 70, 3), rng.uniform(0, 20, 3)))
        prev = rng.uniform(-5, 2.5, 3); u = admissible(rng.uniform(-5, 2.5, 3), prev, config)
        assert robust_certificate(x, prev, u, c, 0.) == certificate(x, prev, u, c)
    for shift in (0., 4., -6.):
        x = np.array([[-80., 12.], [-85., 13.], [-120., 11.]])
        x[:, 0] += shift
        old, _ = plan_fast(x, np.zeros(3), config, pruning=True)
        new = RobustGate(x, np.zeros(3), config, radius=0.)
        assert old.selected == new.selected and old.predicted_cost == new.predicted_cost


def test_backup_recursion_at_extreme_and_time_varying_disturbances():
    rng = np.random.default_rng(91603); config = MergeConfig(); c = parameters(config)
    accepted = 0
    for _ in range(100):
        x = np.column_stack((-rng.uniform(75, 115, 3), rng.uniform(8, 16, 3)))
        x[2, 0] = x[0, 0]-40
        prev = np.zeros(3); u = rng.uniform(-2.5, 2.5, 3)
        if not robust_certificate(x, prev, u, c, .5): continue
        accepted += 1
        for corner in [*itertools.product((-.5, .5), repeat=3), None]:
            y = x.copy()
            for _ in range(5):
                error = rng.uniform(-.5, .5, 3) if corner is None else np.array(corner)
                z = y.copy(); z[:, 1] = np.clip(y[:, 1]+(u+error)*.1, 0, 20); z[:, 0] += z[:, 1]*.1
                assert transition_safe(y[:, 0][None], z[:, 0][None], config)[0]
                y = z
            assert robust_certificate(y, u, np.maximum(u-2.5, -5.), c, .5)
    assert accepted >= 50


def test_bound_monotonicity_terminal_invariance_and_input_rejection():
    config = MergeConfig(); c = parameters(config); rng = np.random.default_rng(91604)
    for _ in range(150):
        x = np.column_stack((-rng.uniform(40, 130, 3), rng.uniform(8, 17, 3))); x[2, 0] = x[0, 0]-35
        u = rng.uniform(-2.5, 2.5, 3); prev = np.zeros(3)
        if robust_certificate(x, prev, u, c, .5):
            assert robust_certificate(x, prev, u, c, .25)
            assert certificate(x, prev, u, c)
    stopped = np.array([[-40., 0.], [-45., 0.], [-80., 0.]])
    assert robust_certificate(stopped, np.full(3, -5.), np.full(3, -5.), c, .5)
    for radius in (-.1, 5., np.nan):
        with pytest.raises(ValueError): RobustGate(stopped, np.zeros(3), radius=radius)
    with pytest.raises(ValueError): RobustGate(np.full((3, 2), np.nan), np.zeros(3))


def test_exogenous_tape_is_reproducible_bounded_and_method_independent():
    for mode in ('clean', 'uniform', 'bias', 'pulse'):
        tape = disturbance_tape(77, mode, .5, 250)
        np.testing.assert_array_equal(tape, disturbance_tape(77, mode, .5, 250))
        assert tape.shape == (250, 3) and np.max(np.abs(tape)) <= .5
