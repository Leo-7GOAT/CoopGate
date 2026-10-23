import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from study_actuator_recovery_v34 import PROFILES, SCENES, complete_cases, review_trace
from coopgate.actuator_recovery_v34 import ActuatorRecoveryBaseline
from coopgate.disturbance_v9 import disturbance_tape
from coopgate.matched_shift_v21 import matched_plant_parameters, tape_fingerprint
from coopgate.merge_v4 import MergeConfig, transition_safe
from coopgate.shifted_sumo_v19 import apply_actuator


def scope():
    return [(32*s+i,scene,i,63400001+32*s+i,plant,'baseline')
            for s,scene in enumerate(SCENES) for i in range(32) for plant in PROFILES]


def test_scope_keeps_every_original_condition():
    cases = complete_cases(scope())
    assert len(cases) == 640
    assert len({tuple(k[:4]) for k in cases}) == 160
    assert all(k[-1] == 'actuator_recovery' for k in cases)


@pytest.mark.parametrize('change', ('missing','duplicate','other_controller','other_plant'))
def test_scope_rejects_changed_or_selected_conditions(change):
    cases = scope()
    if change == 'missing': cases.pop()
    if change == 'duplicate': cases[-1] = cases[0]
    if change == 'other_controller': cases[0] = (*cases[0][:-1],'raw')
    if change == 'other_plant': cases[0] = (*cases[0][:4],'unknown','baseline')
    with pytest.raises(ValueError): complete_cases(cases)


def simulated_fixture(profile):
    """Independent deterministic model fixture; explicitly not a SUMO run."""
    c = MergeConfig()
    seed, mode = 34019, 'uniform'
    plant = matched_plant_parameters(seed,profile,.1)
    tape = disturbance_tape(seed,mode,.1,250)
    x = np.array([[-140.,12.],[-220.,10.],[-190.,12.]])
    initial = x.copy()
    prev, memory, variation = np.zeros(3),np.zeros(3),np.zeros(3)
    active, finish = np.ones(3,bool),np.full(3,25.)
    gate = ActuatorRecoveryBaseline(profile)
    trace = []
    maximum = max_slew = 0.
    exceeded = violations = 0
    for k in range(50):
        command,info = gate.select(x,prev)
        variation += active*np.abs(command-prev)
        max_slew = max(max_slew,float(np.max(np.abs(command-prev))/.5))
        errors,memories,innovations = [],[],[]
        for j in range(5):
            w = tape[5*k+j]
            applied,memory = apply_actuator(command,x[:,1],memory,w,plant)
            error = applied-command
            errors.append(error.tolist()); memories.append(memory.tolist()); innovations.append(w.tolist())
            magnitude = float(np.max(np.abs(error)))
            maximum = max(maximum,magnitude)
            exceeded += int(magnitude > .5+1e-9)
            before = x[:,0].copy()
            x[:,1] = np.clip(x[:,1]+.1*applied,0.,20.)
            x[:,0] += .1*x[:,1]
            violations += not bool(transition_safe(before[None],x[:,0][None],c)[0])
        elapsed = (k+1)*.5
        newly = active & (x[:,0] >= 60.)
        finish[newly],active[newly] = elapsed,False
        trace.append(dict(t=elapsed,x=x.tolist(),command=command.tolist(),innovation=innovations,
            disturbance=errors,actuator_states=memories,
            realized_error_in_contract=bool(np.max(np.abs(errors)) <= .5+1e-9),**info))
        prev = command
        if not active.any(): break
    cost = finish+.05*variation
    row = dict(seed=seed,plant_name=profile,plant=plant,initial=initial.tolist(),disturbance=mode,
        physical_error_radius=.5,external_innovation_radius=.1,innovation_tape_sha256=tape_fingerprint(tape),
        certificate_contract='bounded_actuator_memory_stopping',per_vehicle_finish=finish.tolist(),
        per_vehicle_variation=variation.tolist(),per_vehicle_cost=cost.tolist(),team_cost=float(cost.mean()),
        max_actual_error=maximum,max_command_slew=max_slew,elapsed=elapsed,aborted=False,abort_reason=None,
        completed=bool(not active.any()),collision=False,contract_violations=int(violations),
        error_contract_exceeded_substeps=exceeded,nominal_physical_assumption_observed=exceeded == 0)
    return row,trace


@pytest.fixture(scope='module')
def fixtures():
    return {p:simulated_fixture(p) for p in PROFILES}


@pytest.mark.parametrize('profile', PROFILES)
def test_independent_reconstruction_matches_full_model_fixture(fixtures,profile):
    row,trace = fixtures[profile]
    review = review_trace(row,trace)
    assert review['commands'] == len(trace)
    assert review['physical_violations'] == 0
    assert review['all_memory_intervals_contain_realized_state']


@pytest.mark.parametrize('change', ('memory','coefficient','noise','command','bounds','cost','prefix'))
def test_reviewer_rejects_altered_evidence(fixtures,change):
    row,trace = copy.deepcopy(fixtures['lag_strong'])
    if change == 'memory': trace[0]['actuator_states'][0][0] += .01
    if change == 'coefficient': row['plant']['tau'][0] = .2
    if change == 'noise': trace[0]['innovation'][0][0] += .01
    if change == 'command': trace[0]['command'][0] -= .1
    if change == 'bounds': trace[0]['actuator_bounds'][0][5] = 0.
    if change == 'cost': row['team_cost'] += .1
    if change == 'prefix': trace.pop()
    with pytest.raises((ValueError,AssertionError)): review_trace(row,trace)
