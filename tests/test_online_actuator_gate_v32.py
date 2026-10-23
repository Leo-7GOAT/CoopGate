from copy import deepcopy

import numpy as np
import pytest

from coopgate.actuator_prediction_roundoff_v31 import predict_suffix
from coopgate.disturbance_v9 import disturbance_tape
from coopgate.matched_shift_v21 import matched_plant_parameters
from coopgate.marl_v12 import LEVELS
from coopgate.online_actuator_gate_v32 import ReferenceActuatorGate, forecast, reference_parameters
from coopgate.shifted_sumo_v19 import PLANTS

STATE = np.array([[-80.,12.],[-85.,13.],[-120.,11.]])


@pytest.mark.parametrize('plant',PLANTS)
@pytest.mark.parametrize('mode',['clean','uniform','bias','pulse'])
@pytest.mark.parametrize('first',[None,np.zeros(3)])
def test_compiled_full_prediction_matches_independent_verified_model(plant,mode,first):
    params = matched_plant_parameters(32031,plant)
    tape = disturbance_tape(32031,mode,.1,250)
    expected = predict_suffix(STATE,np.zeros(3),np.zeros(3),0,tape,params,first,record_trace=True)
    actual = forecast(STATE,np.zeros(3),np.zeros(3),0,tape,params,first,record_trace=True)
    for key in ('values','final_state','final_actuator','max_actual_error'):
        np.testing.assert_allclose(actual[key],expected[key],atol=1e-9,rtol=0)
    for key in ('complete','steps','route_violations','error_contract_exceeded_substeps'):
        assert actual[key] == expected[key]
    np.testing.assert_allclose([np.nan if x is None else x for x in actual['crossing']],
        [np.nan if x is None else x for x in expected['crossing']],atol=1e-9,rtol=0,equal_nan=True)
    for a,b in zip(actual['trace'],expected['trace'],strict=True):
        assert a['command'] == b['command']
        for key in ('x','actuator_states','disturbance'):
            np.testing.assert_allclose(a[key],b[key],atol=1e-9,rtol=0)


class ConstantProposal:
    def __init__(self,indices=None):
        self.indices = np.array([np.argmin(np.abs(LEVELS))]*3) if indices is None else np.asarray(indices)
    def proposal(self,observations,mask):
        return self.indices.copy()


def test_nominal_reference_and_nominal_ablation_share_exact_actions_candidates_and_paths():
    left = ReferenceActuatorGate(ConstantProposal(),'nominal','reference')
    right = ReferenceActuatorGate(ConstantProposal(),'nominal','nominal')
    a,ia = left.select(STATE,np.zeros(3))
    b,ib = right.select(STATE,np.zeros(3))
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(left.tapes,right.tapes)
    for key in ('candidate_evidence','selected_bank_differences','forecast_calls','reconstructed_actuator',
                'finite_bank_performance_margin','nominal_joint_action_admitted'):
        assert ia[key] == ib[key]
    assert not ia['future_realization_used'] and not ia['true_randomized_plant_used']


def test_memory_uses_the_executed_command_even_if_it_differs_from_last_proposal():
    gate = ReferenceActuatorGate(ConstantProposal(),'lag_strong','reference')
    selected,_ = gate.select(STATE,np.zeros(3))
    executed = np.array([1.25,1.25,1.25])
    assert not np.array_equal(selected,executed)
    _,info = gate.select(STATE,executed)
    expected = np.zeros(3)
    rho = np.exp(-.1/.4)
    for _ in range(5): expected = rho*expected+(1-rho)*executed
    np.testing.assert_array_equal(info['reconstructed_actuator'],expected)


def test_model_parameters_and_disturbance_bank_do_not_depend_on_episode_seed():
    left = ReferenceActuatorGate(ConstantProposal(),'lag_mild')
    right = ReferenceActuatorGate(ConstantProposal(),'lag_mild')
    np.testing.assert_array_equal(left.tapes,right.tapes)
    assert left.plant == right.plant == dict(gain=[1.]*3,drag=[0.]*3,tau=[.04]*3)
    assert left.plant != {k:v for k,v in matched_plant_parameters(32031,'lag_mild').items() if k in left.plant}
    assert np.max(np.abs(left.tapes)) <= .1
    params = reference_parameters('lag_mild')
    params['tau'][0] = 10.
    assert reference_parameters('lag_mild')['tau'][0] == .04


def test_incomplete_branches_never_supply_a_performance_certificate():
    params = reference_parameters('lag_strong')
    actual = forecast(STATE,np.zeros(3),np.zeros(3),50,np.zeros((250,3)),params)
    assert not actual['complete'] and actual['steps'] == 0
    unsafe = np.array([[0.,10.],[0.,10.],[-1.,10.]])
    actual = forecast(unsafe,np.zeros(3),np.zeros(3),0,np.zeros((250,3)),params)
    assert not actual['complete'] and actual['error_code'] == 1


@pytest.mark.parametrize('change',['state','clock','memory','tape','command'])
def test_invalid_online_forecast_inputs_are_rejected(change):
    args = dict(state=STATE,previous=np.zeros(3),memory=np.zeros(3),step=0,
        tape=np.zeros((250,3)),plant=reference_parameters('nominal'))
    if change == 'state': args['state'] = np.full((3,2),np.nan)
    elif change == 'clock': args['step'] = True
    elif change == 'memory': args['memory'] = np.full(3,2.5+1e-10)
    elif change == 'tape': args['tape'] = np.zeros((5,3))
    else: args['first_command'] = np.full(3,-5.)
    with pytest.raises(ValueError): forecast(**args)
