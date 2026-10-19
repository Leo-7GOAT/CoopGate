from types import SimpleNamespace

import numpy as np
import pytest

import coopgate.repair_portfolio_v20 as module
from coopgate.fast_v8 import parameters
from coopgate.joint_planner_v15 import joint_rollout
from coopgate.merge_v4 import MergeConfig
from coopgate.portfolio_gate_v19 import joint_compositions
from coopgate.robust_v9 import robust_certificate


def geometry():
    state = np.array([[-80., 12.], [-85., 13.], [-120., 11.]])
    previous, packed = np.zeros(3), parameters(MergeConfig())
    baseline, _, _, _ = joint_rollout(state, previous, 0, packed)
    raw = np.array([-2.5, 1.875, 0.])
    return state, previous, packed, baseline, raw


def test_extension_keeps_every_original_joint_command_and_excludes_baseline_duplicates():
    state, previous, packed, baseline, raw = geometry()
    originals = joint_compositions(state, previous, raw, baseline, packed)
    candidates, count, errors = module.expanded_compositions(state, previous, raw, baseline, packed)
    assert count == len(originals) and not errors
    assert count < len(candidates) <= 28
    for (code, command, _), (new_code, weight, new_command) in zip(originals, candidates[:count]):
        assert code == new_code and weight == 1.
        np.testing.assert_array_equal(command, new_command)
    assert len({command.tobytes() for _, _, command in candidates}) == len(candidates)
    for _, _, command in candidates:
        assert not np.array_equal(command, baseline)
        assert robust_certificate(state, previous, command, packed, .5)
    reference, reference_count, _ = module.expanded_compositions(state, previous, raw, baseline, packed, False)
    assert len(reference) == reference_count == count


def test_extra_projection_failure_cannot_remove_original_candidates(monkeypatch):
    state, previous, packed, baseline, raw = geometry()

    def fail(*_):
        raise RuntimeError('additional projection failed')

    monkeypatch.setattr(module, 'attenuated_candidates', fail)
    candidates, count, errors = module.expanded_compositions(state, previous, raw, baseline, packed)
    assert len(candidates) == count > 0 and len(errors) == count
    assert all(weight == 1. for _, weight, _ in candidates)


def controlled_gate(monkeypatch, paths=1):
    state, previous, packed, baseline, raw = geometry()
    candidates, count, _ = module.expanded_compositions(state, previous, raw, baseline, packed)
    original = candidates[0]
    repair = next(candidate for candidate in candidates[count:] if candidate[0] == original[0])
    selected = [original, repair]
    gate = module.RepairPortfolioGate(None, 'sampled')
    gate.names = tuple('path_' + str(i) for i in range(paths))
    gate.tapes = np.zeros((paths, 250, 3))
    gate.primary = SimpleNamespace(step=0, select=lambda *_: (baseline, {'proposal': raw.tolist()}))
    monkeypatch.setattr(module, 'expanded_compositions', lambda *_: (selected, 1, []))
    monkeypatch.setattr(module, 'baseline_suffix', lambda *_: (np.full((3, 2), 10.), baseline, 0, 0))
    return gate, state, previous, baseline, original, repair


def test_extra_nominal_forecast_failure_does_not_abort_or_erase_viable_original(monkeypatch):
    gate, state, previous, _, original, repair = controlled_gate(monkeypatch)

    def nominal(x, p, k, command, packed):
        if np.array_equal(command, repair[2]):
            raise RuntimeError('extra branch unavailable')
        value = np.full((3, 2), 10.)
        value[:, 0] -= .1
        return value, 0, 0

    monkeypatch.setattr(module, 'action_then_baseline', nominal)
    command, info = gate.select(state, previous)
    np.testing.assert_array_equal(command, original[2])
    assert info['portfolio_joint_action_admitted'] and info['repair_weight'] == 1.
    assert len(info['forecast_errors']) == 1 and not info['work_counts_complete']


def test_better_nominal_repair_still_rejected_for_another_vehicle_time_loss(monkeypatch):
    gate, state, previous, _, original, repair = controlled_gate(monkeypatch, paths=2)

    def difference(command, disturbed=False):
        value = np.zeros((3, 2))
        repaired = np.array_equal(command, repair[2])
        value[:, 0] = -.2 if repaired else -.1
        if disturbed and repaired:
            value[2, 1] = .5
        return value

    monkeypatch.setattr(module, 'action_then_baseline', lambda x, p, k, a, c: (10. + difference(a), 0, 0))
    monkeypatch.setattr(module, 'disturbed_transition', lambda *_: None)

    def suffix(x, p, k, tape, packed):
        if k == 0:
            return np.full((3, 2), 10.), 0, 0
        stage = np.column_stack((.5 + .05 * np.abs(p - previous), np.full(3, .5)))
        return 10. + difference(p, True) - stage, 0, 0

    monkeypatch.setattr(module, 'disturbed_suffix', suffix)
    command, info = gate.select(state, previous)
    np.testing.assert_array_equal(command, original[2])
    assert not info['repair_action_admitted']
    assert info['candidate_evidence'][0]['failed_scenarios'] == 1
    assert len(info['selected_bank_differences']) == 2


def test_shared_counterfactual_failure_falls_back_without_fake_learning_adoption(monkeypatch):
    gate, state, previous, baseline, _, _ = controlled_gate(monkeypatch, paths=2)
    value = np.full((3, 2), 10.)
    value[:, 0] -= .1
    monkeypatch.setattr(module, 'action_then_baseline', lambda *_: (value.copy(), 0, 0))

    def fail(*_):
        raise RuntimeError('baseline counterfactual unavailable')

    monkeypatch.setattr(module, 'disturbed_suffix', fail)
    command, info = gate.select(state, previous)
    np.testing.assert_array_equal(command, baseline)
    assert not info['portfolio_joint_action_admitted'] and not info['repair_action_admitted']
    assert info['repair_weight'] is None and not info['selected_bank_differences']
    assert len(info['forecast_errors']) == 2


def test_complete_bank_chooses_greater_nominal_gain_without_relaxing_threshold(monkeypatch):
    gate, state, previous, _, _, repair = controlled_gate(monkeypatch)

    def nominal(x, p, k, command, packed):
        value = np.full((3, 2), 10.)
        value[:, 0] -= .2 if np.array_equal(command, repair[2]) else .1
        return value, 0, 0

    monkeypatch.setattr(module, 'action_then_baseline', nominal)
    command, info = gate.select(state, previous)
    np.testing.assert_array_equal(command, repair[2])
    assert info['repair_action_admitted'] and info['max_failed_scenarios'] == 0
    assert np.max(info['selected_bank_differences']) <= 1e-10


def test_invalid_repair_configuration_is_rejected():
    with pytest.raises(ValueError, match='boolean'):
        module.RepairPortfolioGate(None, with_repair=1)


@pytest.mark.parametrize('bad_value', [np.nan, np.inf])
def test_nonfinite_extra_forecast_cannot_erase_a_viable_original(monkeypatch, bad_value):
    gate, state, previous, _, original, repair = controlled_gate(monkeypatch)

    def nominal(x, p, k, command, packed):
        value = np.full((3, 2), 10.)
        value[:, 0] -= .1
        if np.array_equal(command, repair[2]):
            value[0, 0] = bad_value
        return value, 0, 0

    monkeypatch.setattr(module, 'action_then_baseline', nominal)
    command, info = gate.select(state, previous)
    np.testing.assert_array_equal(command, original[2])
    assert info['portfolio_joint_action_admitted'] and len(info['forecast_errors']) == 1
