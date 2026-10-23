import copy

import numpy as np
import pytest

from coopgate.disturbance_v9 import disturbance_tape
from coopgate.marl_v12 import LEVELS
from coopgate.matched_shift_v21 import matched_plant_parameters, tape_fingerprint
from coopgate.merge_v4 import MergeConfig, transition_safe
from coopgate.recovery_gate_audit_v35 import cell_summary, relative_outcome, review_joint_trace
from coopgate.recovery_joint_gate_v35 import RecoveryJointGate
from coopgate.shifted_sumo_v19 import apply_actuator


class CoastingActor:
    def proposal(self, observations, mask):
        scores = np.broadcast_to(-np.abs(LEVELS),mask.shape).copy()
        scores[~mask] = -np.inf
        return scores.argmax(-1)


def model_fixture(profile, rule):
    """Deterministic model fixture, explicitly not a SUMO efficacy observation."""
    c, seed, mode = MergeConfig(), 35027, 'uniform'
    model = None if rule == 'baseline' else CoastingActor()
    gate = RecoveryJointGate(model,profile,rule)
    plant = matched_plant_parameters(seed,profile,.1)
    tape = disturbance_tape(seed,mode,.1,250)
    x = np.array([[-140.,12.],[-220.,10.],[-190.,12.]])
    initial = x.copy()
    previous,memory,variation = np.zeros(3),np.zeros(3),np.zeros(3)
    finish,active = np.full(3,25.),np.ones(3,bool)
    trace = []
    maximum = max_slew = 0.
    exceeded = violations = 0
    for step in range(50):
        command,info = gate.select(x,previous)
        variation += active*np.abs(command-previous)
        max_slew = max(max_slew,float(np.max(np.abs(command-previous))/.5))
        errors,memories,innovations = [],[],[]
        for sub in range(5):
            noise = tape[step*5+sub]
            acceleration,memory = apply_actuator(command,x[:,1],memory,noise,plant)
            error = acceleration-command
            errors.append(error.tolist());memories.append(memory.tolist());innovations.append(noise.tolist())
            magnitude = float(np.max(np.abs(error)))
            maximum = max(maximum,magnitude)
            exceeded += int(magnitude > .5+1e-9)
            before = x[:,0].copy()
            x[:,1] = np.clip(x[:,1]+.1*acceleration,0.,20.)
            x[:,0] += .1*x[:,1]
            violations += not bool(transition_safe(before[None],x[:,0][None],c)[0])
        elapsed = (step+1)*.5
        newly = active & (x[:,0] >= 60.)
        finish[newly],active[newly] = elapsed,False
        trace.append(dict(t=elapsed,x=x.tolist(),command=command.tolist(),innovation=innovations,
            disturbance=errors,actuator_states=memories,
            realized_error_in_contract=bool(np.max(np.abs(errors)) <= .5+1e-9),**info))
        previous = command
        if not active.any(): break
    cost = finish+.05*variation
    row = dict(seed=seed,plant_name=profile,plant=plant,initial=initial.tolist(),disturbance=mode,
        rule=rule,physical_error_radius=.5,external_innovation_radius=.1,
        innovation_tape_sha256=tape_fingerprint(tape),certificate_contract='bounded_actuator_memory_stopping',
        per_vehicle_finish=finish.tolist(),per_vehicle_variation=variation.tolist(),per_vehicle_cost=cost.tolist(),
        team_cost=float(cost.mean()),max_actual_error=maximum,max_command_slew=max_slew,elapsed=elapsed,
        aborted=False,abort_reason='',completed=bool(not active.any()),collision=False,contract_violations=int(violations),
        error_contract_exceeded_substeps=exceeded,nominal_physical_assumption_observed=exceeded == 0,
        control_steps=len(trace),learned_action_steps=sum(t['actual_learning_used'] for t in trace))
    return row,trace,model


CASES = [(p,'baseline') for p in ('nominal','gain_drag','lag_mild','lag_strong')]
CASES += [('lag_strong',r) for r in ('mps_shared','mamps_shared','nominal_performance','bank_performance')]


@pytest.fixture(scope='module')
def fixtures():
    return {case:model_fixture(*case) for case in CASES}


@pytest.mark.parametrize('case',CASES)
def test_all_physical_steps_and_complete_gate_replay(fixtures,case):
    row,trace,model = fixtures[case]
    first = review_joint_trace(row,trace,model,replay=False)
    full = review_joint_trace(row,trace,model,replay=True)
    assert first == full
    assert full['commands'] == len(trace) and full['physical_violations'] == 0
    assert full['all_realized_memory_inside_online_intervals']


@pytest.mark.parametrize('change',('memory','coefficient','noise','command','bounds','cost','prefix',
    'adoption','prediction_count','baseline_evidence','future_information'))
def test_reviewer_rejects_altered_evidence(fixtures,change):
    row,trace,model = copy.deepcopy(fixtures[('lag_strong','baseline')])
    if change == 'memory': trace[0]['actuator_states'][0][0] += .01
    if change == 'coefficient': row['plant']['tau'][0] = .2
    if change == 'noise': trace[0]['innovation'][0][0] += .01
    if change == 'command': trace[0]['command'][0] -= .1
    if change == 'bounds': trace[0]['actuator_bounds'][0][5] = 0.
    if change == 'cost': row['team_cost'] += .1
    if change == 'prefix': trace.pop()
    if change == 'adoption': trace[0]['actual_learning_used'] = True
    if change == 'prediction_count': trace[0]['forecast_calls'] += 1
    if change == 'baseline_evidence': trace[0]['baseline_evidence']['nominal_proposal'][0] += .1
    if change == 'future_information': trace[0]['future_realization_used'] = True
    with pytest.raises((ValueError,AssertionError)):
        review_joint_trace(row,trace,model,replay=True)


def pair(*,adopted=False,completed=True,baseline_completed=True,change=0.):
    baseline = dict(completed=baseline_completed,aborted=not baseline_completed,collision=False,
        contract_violations=0,per_vehicle_cost=[10.,10.,10.],per_vehicle_finish=[9.,9.,9.],team_cost=10.)
    row = dict(baseline,completed=completed,aborted=not completed,learned_action_steps=int(adopted),
        per_vehicle_cost=[10.+change,10.,10.],team_cost=10.+change/3.)
    return dict(row=row,baseline=baseline,calls=[dict(wall_ms=1.,error=None)],**relative_outcome(row,baseline))


def test_zero_adoption_is_undefined_conditional_risk():
    summary = cell_summary([pair(change=-1.)])
    assert summary['empirical_conditional_harm'] is None
    assert summary['adopted'] == summary['harmful_adopted'] == 0


def test_any_car_harm_and_failures_count_without_hiding_missing_baseline():
    cases = [pair(adopted=True,change=1.),pair(adopted=True,completed=False),
             pair(adopted=True,baseline_completed=False),pair(adopted=True,change=-1.)]
    summary = cell_summary(cases)
    assert summary['adopted'] == 4 and summary['adopted_with_valid_baseline'] == 3
    assert summary['harmful_adopted'] == 2 and summary['empirical_conditional_harm'] == 2/3
    assert summary['adopted_with_incomplete_baseline'] == 1
    assert cases[2]['relative_harm'] is None
    assert summary['all_scope_cost_gain_percent'] is None
    assert summary['both_complete'] == 2


def test_any_failure_withholds_all_scope_gain_but_retains_capped_cost():
    summary = cell_summary([pair(change=-1.),pair(completed=False,change=-5.)])
    assert summary['all_scope_cost_gain_percent'] is None
    assert summary['complete_pair_descriptive_gain_percent'] > 0.
    assert summary['original_capped_mean_cost'] == 9.
