"""A visited-state forecast must keep the original clock and actuator memory."""
import numpy as np
import pytest

from coopgate.actuator_prediction_v30 import predict_suffix
from coopgate.disturbance_v9 import disturbance_tape
from coopgate.matched_shift_v21 import matched_plant_parameters
from coopgate.shifted_sumo_v19 import PLANTS


@pytest.mark.parametrize('plant_name', PLANTS)
def test_splitting_a_feedback_rollout_retains_its_exact_suffix_and_original_cost_accounting(plant_name):
    state = np.array([[-80.,12.],[-85.,13.],[-120.,11.]])
    tape = disturbance_tape(30023, 'uniform', .1, 250)
    plant = matched_plant_parameters(30023, plant_name)
    whole = predict_suffix(state,np.zeros(3),np.zeros(3),0,tape,plant,record_trace=True)
    assert whole['steps'] > 3
    boundary = whole['trace'][2]
    suffix = predict_suffix(np.asarray(boundary['x']),np.asarray(boundary['command']),
        np.asarray(boundary['actuator_states'][-1]),3,tape,plant,record_trace=True)
    assert suffix['trace'] == whole['trace'][3:]
    assert suffix['complete'] == whole['complete']
    assert suffix['error'] == whole['error']
    previous, spent = np.zeros(3), np.zeros((3,2))
    for item in whole['trace'][:3]:
        active = state[:,0] < 60.
        command = np.asarray(item['command'])
        spent[:,0] += active*(.5+.05*np.abs(command-previous))
        spent[:,1] += active*.5
        state, previous = np.asarray(item['x']),command
    np.testing.assert_allclose(np.asarray(suffix['values'])+spent,whole['values'],atol=1e-10,rtol=0)
