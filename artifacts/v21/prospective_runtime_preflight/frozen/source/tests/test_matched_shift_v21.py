import copy

import numpy as np
import pytest

from coopgate.disturbance_v9 import disturbance_tape
from coopgate.matched_shift_v21 import audit_matched_trace, matched_plant_parameters, tape_fingerprint
from coopgate.shifted_sumo_v19 import PLANTS, apply_actuator, plant_parameters


def physical_example(plant_name):
    seed = 62101001
    plant = matched_plant_parameters(seed, plant_name)
    tape = disturbance_tape(seed, 'uniform', .1, 250)
    state = np.array([[-80., 12.], [-85., 13.], [-120., 11.]])
    initial = state.copy()
    actuator, previous, variation = np.zeros(3), np.zeros(3), np.zeros(3)
    trace, maximum, exceeded = [], 0., 0
    for step, command in enumerate((np.full(3, 2.5), np.zeros(3))):
        innovations, memories, errors = [], [], []
        variation += np.abs(command - previous)
        for j in range(5):
            innovation = tape[step * 5 + j]
            applied, actuator = apply_actuator(command, state[:, 1], actuator, innovation, plant)
            error = applied - command
            maximum = max(maximum, float(np.abs(error).max()))
            exceeded += int(np.abs(error).max() > .5 + 1e-9)
            state[:, 1] = np.clip(state[:, 1] + applied * .1, 0, 20)
            state[:, 0] += state[:, 1] * .1
            innovations.append(innovation.tolist())
            memories.append(actuator.tolist())
            errors.append(error.tolist())
        trace.append(dict(t=(step + 1) * .5, x=state.tolist(), command=command.tolist(),
                          innovation=innovations, actuator_states=memories, disturbance=errors,
                          realized_error_in_contract=bool(np.max(np.abs(errors)) <= .5 + 1e-9)))
        previous = command
    row = dict(seed=seed, initial=initial.tolist(), plant_name=plant_name, plant=plant,
               disturbance='uniform', radius=.5, physical_error_radius=.5, external_innovation_radius=.1,
               innovation_tape_sha256=tape_fingerprint(tape), per_vehicle_finish=[25.] * 3,
               per_vehicle_variation=variation.tolist(), elapsed=1., max_actual_error=maximum,
               max_command_slew=5., error_contract_exceeded_substeps=exceeded,
               nominal_physical_assumption_observed=exceeded == 0, contract_violations=0,
               completed=False, collision=False, aborted=True, abort_reason='synthetic retained abort')
    return trace, row


def test_equal_external_noise_does_not_change_legacy_non_nominal_plant_parameters():
    for seed in (0, 62101001, 63400001):
        for plant in PLANTS:
            actual = matched_plant_parameters(seed, plant)
            assert actual['innovation_radius'] == .1
            old = plant_parameters(seed, plant)
            for key in ('gain', 'drag', 'tau'): assert actual[key] == old[key]
            if plant != 'nominal': assert actual == old
        assert matched_plant_parameters(seed, 'nominal', .5) == plant_parameters(seed, 'nominal')
    for radius in (True, -.1, .6, np.nan, np.inf):
        with pytest.raises(ValueError): matched_plant_parameters(1, 'nominal', radius)


@pytest.mark.parametrize('plant', PLANTS)
def test_independent_recurrence_reconstructs_clipped_vehicle_dynamics_and_keeps_abort(plant):
    trace, row = physical_example(plant)
    audit = audit_matched_trace(trace, row)
    assert audit['aborted'] and not audit['completed'] and audit['substeps'] == 10
    if plant == 'lag_strong': assert audit['error_contract_exceeded_substeps'] > 0
    else: assert audit['error_contract_exceeded_substeps'] == 0


def test_audit_detects_noise_substitution_actuator_memory_tampering_and_hidden_abort():
    trace, row = physical_example('lag_strong')
    for field in ('innovation', 'actuator_states', 'disturbance'):
        changed = copy.deepcopy(trace)
        changed[1][field][0][0] += .01
        with pytest.raises(AssertionError): audit_matched_trace(changed, row)
    with pytest.raises(ValueError, match='tape'):
        audit_matched_trace(trace, {**row, 'innovation_tape_sha256': 'not-the-tape'})
    with pytest.raises(ValueError, match='Completion'):
        audit_matched_trace(trace, {**row, 'completed': True})
    with pytest.raises(ValueError, match='mismatch'):
        audit_matched_trace(trace, {**row, 'error_contract_exceeded_substeps': 0})


def test_matched_noise_does_not_artificially_match_action_dependent_errors():
    plant = matched_plant_parameters(62101001, 'lag_strong')
    innovation = disturbance_tape(62101001, 'uniform', .1, 250)[0]
    first, _ = apply_actuator(np.zeros(3), np.full(3, 12.), np.zeros(3), innovation, plant)
    second, _ = apply_actuator(np.full(3, 2.5), np.full(3, 12.), np.zeros(3), innovation, plant)
    assert not np.allclose(first, second - 2.5)
