"""Independent recurrence, containment and continuing-backup checks."""
import numpy as np
import pytest

from coopgate.actuator_recovery_v34 import (
    ActuatorRecoveryBaseline, certificate, class_bounds, interval_substep,
    memory_step, validate_inputs,
)
from coopgate.fast_v8 import parameters
from coopgate.merge_v4 import MergeConfig, transition_safe
from coopgate.shifted_sumo_v19 import apply_actuator, plant_parameters


PROFILES = ('nominal', 'gain_drag', 'lag_mild', 'lag_strong')


@pytest.mark.parametrize('profile', PROFILES)
def test_interval_contains_independent_plant_substeps(profile):
    rng = np.random.default_rng(34001)
    c = MergeConfig()
    b = class_bounds(profile)
    packed = parameters(c)
    for seed in range(300):
        lo = np.column_stack((rng.uniform(-150., 100., 3), rng.uniform(0., 18., 3)))
        hi = lo + np.column_stack((rng.uniform(0., 4., 3), rng.uniform(0., 2., 3)))
        mlo = rng.uniform(-5., 1., 3)
        mhi = mlo + rng.uniform(0., 1.5, 3)
        command = rng.uniform(-5., 2.5, 3)
        qlo, qhi, amlo, amhi = interval_substep(lo, hi, mlo, mhi, command, b, packed)
        # Include state/memory extrema as well as an interior realization.
        for fraction in (0., 1., rng.random()):
            x = lo + fraction*(hi-lo)
            m = mlo + (1.-fraction)*(mhi-mlo)
            p = plant_parameters(seed, profile)
            innovation = rng.choice(np.array([-.1, .1]), 3)
            applied, memory = apply_actuator(command, x[:, 1], m, innovation, p)
            q = x.copy()
            q[:, 1] = np.clip(x[:, 1]+applied*c.dt, 0., c.vmax)
            q[:, 0] += q[:, 1]*c.dt
            assert np.all(q >= qlo-1e-10) and np.all(q <= qhi+1e-10)
            assert np.all(memory >= amlo-1e-12) and np.all(memory <= amhi+1e-12)


@pytest.mark.parametrize('profile', PROFILES)
def test_command_only_observer_contains_full_random_history(profile):
    rng = np.random.default_rng(34002)
    b = class_bounds(profile)
    plant = plant_parameters(34002, profile)
    memory, lo, hi = np.zeros(3), np.zeros(3), np.zeros(3)
    for _ in range(250):
        command = rng.uniform(-5., 2.5, 3)
        lo, hi = memory_step(lo, hi, command, b)
        _, memory = apply_actuator(command, rng.uniform(0., 20., 3), memory,
                                   rng.uniform(-.1, .1, 3), plant)
        assert np.all(memory >= lo-1e-12) and np.all(memory <= hi+1e-12)


@pytest.mark.parametrize('profile', PROFILES)
def test_certified_backup_stays_recoverable_and_stopped(profile):
    c = MergeConfig()
    b = class_bounds(profile)
    plant = plant_parameters(34003, profile)
    state = np.array([[-140., 12.], [-220., 10.], [-190., 12.]])
    previous, memory = np.zeros(3), np.zeros(3)
    mlo, mhi = np.zeros(3), np.zeros(3)
    initial = certificate(state, previous, np.full(3, 2.5), mlo, mhi, b)
    assert initial['certified']
    for decision in range(25):
        command = np.full(3, 2.5) if decision == 0 else np.maximum(previous-2.5, -5.)
        result = certificate(state, previous, command, mlo, mhi, b)
        assert result['certified'], (profile, decision, result)
        for _ in range(5):
            before = state[:, 0].copy()
            applied, memory = apply_actuator(command, state[:, 1], memory,
                                             np.array([.1, -.1, .1]), plant)
            state[:, 1] = np.clip(state[:, 1]+applied*.1, 0., 20.)
            state[:, 0] += state[:, 1]*.1
            mlo, mhi = memory_step(mlo, mhi, command, b)
            assert bool(transition_safe(before[None], state[:, 0][None], c)[0])
            assert np.all(memory >= mlo-1e-12) and np.all(memory <= mhi+1e-12)
        previous = command
    np.testing.assert_array_equal(state[:, 1], 0.)
    assert np.all(mhi <= -.1/b[:, 0])


def test_stopped_speed_alone_is_not_an_invariant_set():
    # A positive stored actuator can restart a stopped vehicle despite a new
    # braking command. A zero tail horizon must not certify speed alone.
    from coopgate.actuator_recovery_v34 import stopping_certificate
    c = MergeConfig(tail_steps=0)
    state = np.array([[-140., 0.], [-220., 0.], [-190., 0.]])
    result = stopping_certificate(state, np.full(3, 2.5), np.zeros(3),
        np.full(3, 2.5), np.full(3, 2.5), class_bounds('lag_strong'), parameters(c))
    assert not result[0] and result[1] == 3
    assert np.all(result[4][:, 1] > 0.)


@pytest.mark.parametrize('profile', PROFILES)
def test_recovery_baseline_observer_uses_executed_command(profile):
    gate = ActuatorRecoveryBaseline(profile)
    x = np.array([[-140., 12.], [-220., 10.], [-190., 12.]])
    _, info = gate.select(x, np.zeros(3))
    np.testing.assert_array_equal(info['actuator_memory_lower'], np.zeros(3))
    # Supply a different valid executed history; the prior proposal is not
    # the actuator observer's input.
    executed = np.full(3, -2.5)
    _, info = gate.select(x, executed)
    lo, hi = np.zeros(3), np.zeros(3)
    for _ in range(5):
        lo, hi = memory_step(lo, hi, executed, class_bounds(profile))
    np.testing.assert_array_equal(info['actuator_memory_lower'], lo)
    np.testing.assert_array_equal(info['actuator_memory_upper'], hi)


def test_nominal_planner_exception_still_requires_a_valid_recovery(monkeypatch):
    import coopgate.actuator_recovery_v34 as module
    def failed(*args):
        raise RuntimeError('Deliberate nominal-proposal failure')
    monkeypatch.setattr(module, 'deduplicated_joint_rollout', failed)
    gate = ActuatorRecoveryBaseline('lag_strong')
    state = np.array([[-140., 12.], [-220., 10.], [-190., 12.]])
    command, info = gate.select(state, np.zeros(3))
    np.testing.assert_array_equal(command, np.full(3, -2.5))
    assert info['recovery_applied'] and info['recovery_certificate']['certified']
    assert info['nominal_proposal_error'] == 'Deliberate nominal-proposal failure'
    unsafe = np.array([[-140., 12.], [-220., 10.], [-141., 12.]])
    with pytest.raises(RuntimeError, match='No actuator-aware'):
        ActuatorRecoveryBaseline('lag_strong').select(unsafe, np.zeros(3))


@pytest.mark.parametrize('field', ('state', 'previous', 'memory', 'bounds'))
def test_invalid_interval_inputs_fail_closed(field):
    x = np.array([[-140., 12.], [-220., 10.], [-190., 12.]])
    p, lo, hi, b = np.zeros(3), np.zeros(3), np.zeros(3), class_bounds('nominal')
    if field == 'state': x[0, 1] = np.nan
    if field == 'previous': p[0] = 3.
    if field == 'memory': lo[0] = .1
    if field == 'bounds': b[0, 5] = 1.
    with pytest.raises(ValueError): validate_inputs(x, p, lo, hi, b)


def test_input_limits_and_unavailable_recovery_are_explicit():
    state = np.array([[-140., 12.], [-220., 10.], [-190., 12.]])
    result = certificate(state, np.zeros(3), np.full(3, -5.),
                          np.zeros(3), np.zeros(3), class_bounds('lag_strong'))
    assert not result['certified'] and result['reason_code'] == 1
    state[2, 0] = state[0, 0]-1.
    result = certificate(state, np.zeros(3), np.full(3, -2.5),
                          np.zeros(3), np.zeros(3), class_bounds('lag_strong'))
    assert not result['certified'] and result['reason_code'] == 2


@pytest.mark.parametrize('profile', PROFILES)
def test_declared_bounds_include_generated_parameters(profile):
    b = class_bounds(profile)
    for seed in range(100):
        plant = plant_parameters(seed, profile)
        gain, drag, tau = [np.asarray(plant[k]) for k in ('gain', 'drag', 'tau')]
        r = np.zeros(3)
        r[tau > 0.] = np.exp(-.1/tau[tau > 0.])
        for values, low, high in ((gain, 0, 1), (drag, 2, 3), (r, 4, 5)):
            assert np.all(values >= b[:, low]) and np.all(values <= b[:, high])
