import numpy as np

from coopgate.fast_v8 import parameters
from coopgate.merge_v4 import MergeConfig
from coopgate.robust_v9 import robust_certificate
from coopgate.shifted_sumo_v19 import apply_actuator, plant_parameters, shifted_initial


def test_declared_shifted_sampler_stays_in_the_nominal_initial_recoverable_set():
    packed = parameters(MergeConfig())
    for seed in range(100):
        for scenario in ('slow_yield', 'asymmetric_speed', 'tight_headway'):
            state = shifted_initial(seed, scenario)
            assert robust_certificate(state, np.zeros(3), np.full(3, -2.5), packed, .5)


def test_mild_physical_shifts_are_action_dependent_and_stay_within_declared_error_envelope():
    rng = np.random.default_rng(19083)
    for kind in ('gain_drag', 'lag_mild'):
        for seed in range(10):
            plant = plant_parameters(seed, kind)
            actuator, command = np.zeros(3), np.zeros(3)
            for step in range(250):
                if step % 5 == 0:
                    command = np.clip(command + rng.uniform(-2.5, 2.5, 3), -5., 2.5)
                applied, actuator = apply_actuator(command, rng.uniform(0, 20, 3), actuator,
                                                   rng.uniform(-.1, .1, 3), plant)
                assert np.max(np.abs(applied - command)) <= .5 + 1e-9
    plant = plant_parameters(1, 'lag_strong')
    applied, _ = apply_actuator(np.full(3, 2.5), np.full(3, 12.), np.zeros(3), np.zeros(3), plant)
    assert np.max(np.abs(applied - 2.5)) > .5
