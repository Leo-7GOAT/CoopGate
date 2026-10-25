from dataclasses import replace

import numpy as np
import pytest

from coopgate.aged_observation_envelope import PublicEnvelope
from coopgate.compound_backup_tube import RetainedBrakePlan, certify_braking_tube, route_box_safe
from coopgate.merge_v4 import MergeConfig, transition_safe
from coopgate.ood_mismatch_v39 import compound_substep, delayed_command


def initial_box():
    lo = np.array([[-200., 7.75], [-320., 7.75], [-440., 7.75]])
    hi = lo+np.array([.5, .5])
    return lo, hi


def test_accepted_tube_contains_independent_hidden_delay_rollouts_and_invariant_tail():
    lo, hi = initial_box()
    cert = certify_braking_tube(lo, hi, np.zeros(3), np.zeros(3), [], [-2.5]*3)
    assert cert['certified'] and cert['reason'] == 'stopped_invariant_set'
    assert not cert['numerically_verified_interval_arithmetic']
    assert not cert['performance_risk_certified'] and not cert['task_completion_certified']
    assert cert['substeps'] == 5*len(cert['commands'])
    b, rng = PublicEnvelope(), np.random.default_rng(461001)
    replayed = 0
    for trial in range(24):
        plant = dict(tau=rng.uniform(b.tau_lower, b.tau_upper, 3),
                     gain=rng.uniform(b.gain_lower, b.gain_upper, 3),
                     drag=rng.uniform(b.drag_lower, b.drag_upper, 3))
        if trial < 8:
            plant['tau'][:] = b.tau_upper if trial & 1 else b.tau_lower
            plant['gain'][:] = b.gain_upper if trial & 2 else b.gain_lower
            plant['drag'][:] = b.drag_upper if trial & 4 else b.drag_lower
        delay = np.roll(np.array([1, 2, 3]), trial % 3)
        x, memory, history = lo+rng.uniform(size=(3, 2))*(hi-lo), np.zeros(3), []
        index = 0
        for command in cert['commands']:
            history.append(np.asarray(command))
            delayed = delayed_command(history, delay)
            for _ in range(5):
                previous = x[:, 0].copy()
                w = rng.uniform(-.1, .1, 3) if trial >= 8 else np.full(3, .1 if trial & 1 else -.1)
                a, memory = compound_substep(delayed, x[:, 1], memory, w, plant)
                x[:, 1] = np.clip(x[:, 1]+.1*a, 0, 20)
                x[:, 0] += .1*x[:, 1]
                step = cert['tube'][index]
                assert np.all(x >= step['lower']) and np.all(x <= step['upper'])
                assert np.all(memory >= step['memory_lower']) and np.all(memory <= step['memory_upper'])
                assert transition_safe(previous[None], x[:, 0][None], MergeConfig()).all()
                index += 1; replayed += 1
        stopped = x.copy()
        for _ in range(40):
            history.append(np.asarray(cert['terminal_hold']))
            delayed = delayed_command(history, delay)
            for _ in range(5):
                a, memory = compound_substep(delayed, x[:, 1], memory, np.full(3, .1), plant)
                x[:, 1] = np.clip(x[:, 1]+.1*a, 0, 20)
                x[:, 0] += .1*x[:, 1]
                np.testing.assert_array_equal(x, stopped)
                replayed += 1
    assert replayed >= 5000


def test_zero_velocity_and_negative_memory_do_not_hide_pending_positive_commands():
    b = replace(PublicEnvelope(), tau_lower=100., tau_upper=100.)
    lo, _ = initial_box(); lo[:, 1] = 0.
    queued = [[2.5]*3, [-5.]*3]
    result = certify_braking_tube(lo, lo, [-5]*3, [-5]*3, queued, [-5]*3,
                                  bounds=b, max_decisions=1)
    assert np.all(np.asarray(result['tube'][-1]['upper'])[:, 1] == 0.)
    assert not result['certified'] and result['reason'] == 'horizon_exhausted'
    drained = certify_braking_tube(lo, lo, [-5]*3, [-5]*3, [[-5.]*3]*3,
                                   [-5]*3, bounds=b, max_decisions=1)
    assert drained['certified']


def test_interval_geometry_rejects_possible_conflict_even_when_midpoint_is_safe():
    lo = np.array([[-35., 0.], [-55., 0.], [-150., 0.]])
    hi = lo.copy(); hi[0, 0] = -24.; hi[1, 0] = -24.
    mid = (lo+hi)/2
    assert route_box_safe(mid, mid, mid, mid)
    assert not route_box_safe(lo, hi, lo, hi)
    cert = certify_braking_tube(lo, hi, [-5]*3, [-5]*3, [[-5]*3]*3, [-5]*3)
    assert not cert['certified'] and cert['reason'] == 'route_box_not_certified'


def test_box_geometry_acceptance_implies_original_contract_for_sampled_affine_paths():
    rng = np.random.default_rng(461002)
    accepted = checked = 0
    for trial in range(500):
        p = rng.uniform(-75., 75., 3)
        p[2] = p[0]-rng.uniform(7., 60.)
        widths = rng.uniform(0, 2., 3)
        lo = np.column_stack((p, np.zeros(3)))
        hi = np.column_stack((p+widths, np.zeros(3)))
        qlo = lo.copy(); qhi = hi.copy()
        qlo[:, 0] += rng.uniform(0, 2., 3)
        qhi[:, 0] += rng.uniform(2., 4., 3)
        if not route_box_safe(lo, hi, qlo, qhi):
            continue
        accepted += 1
        starts = lo[:, 0]+rng.uniform(size=(64, 3))*(hi[:, 0]-lo[:, 0])
        ends = qlo[:, 0]+rng.uniform(size=(64, 3))*(qhi[:, 0]-qlo[:, 0])
        monotone = np.all(ends >= starts, axis=1)
        assert transition_safe(starts[monotone], ends[monotone], MergeConfig()).all()
        checked += int(monotone.sum())
    assert accepted > 50 and checked > 3000


def test_candidate_must_obey_jerk_and_failure_is_not_silently_repaired():
    lo, hi = initial_box()
    cert = certify_braking_tube(lo, hi, [0]*3, [0]*3, [], [-5]*3)
    assert not cert['certified'] and cert['reason'] == 'candidate_command_or_slew'
    assert cert['commands'] == [] and cert['tube'] == []


def test_inputs_are_not_mutated_and_all_past_pending_commands_are_retained():
    lo, hi = initial_box(); ml = np.zeros(3); mh = np.zeros(3)
    hist = np.array([[2.5]*3, [0.]*3]); first = np.array([-2.5]*3)
    originals = [x.copy() for x in (lo, hi, ml, mh, hist, first)]
    cert = certify_braking_tube(lo, hi, ml, mh, hist, first)
    assert cert['certified']
    for actual, old in zip((lo, hi, ml, mh, hist, first), originals):
        np.testing.assert_array_equal(actual, old)


@pytest.mark.parametrize('field,value', [('lower', [[float('nan'), 0.]]*3),
    ('upper', [[-100., 21.]]*3), ('memory_upper', [3.]*3),
    ('executed_commands', [[float('inf')]*3]), ('max_decisions', 0),
    ('geometry_guard', -1.), ('jerk', 0.)])
def test_invalid_public_domains_are_rejected(field, value):
    lo, hi = initial_box()
    args = dict(lower=lo, upper=hi, memory_lower=[0]*3, memory_upper=[0]*3,
                executed_commands=[], first_command=[-2.5]*3)
    args[field] = value
    with pytest.raises(ValueError):
        certify_braking_tube(**args)


def test_rejected_replanning_retains_entire_tape_and_terminal_hold():
    lo, hi = initial_box()
    manager = RetainedBrakePlan()
    accepted = manager.offer(lo, hi, [0]*3, [0]*3, [-2.5]*3)
    assert accepted['accepted']
    tape = accepted['check']['commands']
    for index, command in enumerate(tape):
        # Trying to accelerate beyond command bounds must not erase fallback.
        rejected = manager.offer(lo, hi, [0]*3, [0]*3, [3.]*3)
        assert not rejected['accepted'] and rejected['retained_plan_available']
        assert rejected['generation'] == 1
        np.testing.assert_array_equal(manager.next_command(), command)
        manager.acknowledge(command)
    for _ in range(5):
        command = manager.next_command()
        np.testing.assert_array_equal(command, tape[-1])
        manager.acknowledge(command)
    assert manager.snapshot()['remaining_commands'] == []


def test_plan_replacement_is_transactional_and_acknowledgment_required():
    lo, hi = initial_box()
    manager = RetainedBrakePlan()
    with pytest.raises(RuntimeError, match='initial certified'):
        manager.next_command()
    assert not manager.offer(lo, hi, [0]*3, [0]*3, [-5]*3)['accepted']
    assert manager.offer(lo, hi, [0]*3, [0]*3, [-2.5]*3)['accepted']
    assert manager.offer(lo, hi, [0]*3, [0]*3, [0.]*3)['generation'] == 2
    issued = manager.next_command(); issued[:] = 99.
    np.testing.assert_array_equal(manager.next_command(), 0.)
    with pytest.raises(RuntimeError, match='before replanning'):
        manager.offer(lo, hi, [0]*3, [0]*3, [-2.5]*3)
    manager.acknowledge([0.]*3)
    with pytest.raises(RuntimeError, match='No issued decision'):
        manager.acknowledge([0.]*3)
    snapshot = manager.snapshot(); snapshot['remaining_commands'][0][:] = [99.]*3
    assert manager.next_command().max() <= 2.5


def test_wrong_physical_command_invalidates_claim_instead_of_advancing_backup():
    lo, hi = initial_box()
    manager = RetainedBrakePlan()
    assert manager.offer(lo, hi, [0]*3, [0]*3, [-2.5]*3)['accepted']
    manager.next_command()
    with pytest.raises(RuntimeError, match='differs'):
        manager.acknowledge([0.]*3)
    assert manager.snapshot()['invalidated']
    with pytest.raises(RuntimeError, match='violated'):
        manager.next_command()
    with pytest.raises(RuntimeError, match='violated'):
        manager.offer(lo, hi, [0]*3, [0]*3, [-2.5]*3)


def test_domain_audit_calculation_cannot_access_hidden_state_or_future_tape():
    from coopgate.aged_observation_envelope import AgeAwareStateObserver
    from audit_compound_backup_domain import check_case

    class PublicOnly(dict):
        def __getitem__(self, key):
            assert key in {'decision','profile','previous_command','raw_state',
                           'source_decision','envelope'}
            return super().__getitem__(key)

    lo, _ = initial_box()
    estimate = AgeAwareStateObserver('clean_reference').update(lo, 0, [0]*3)
    frame = PublicOnly(decision=0,profile='clean_reference',previous_command=[0]*3,
                      raw_state=lo,source_decision=0,envelope=estimate)
    assert check_case([frame])[0]['certified']
