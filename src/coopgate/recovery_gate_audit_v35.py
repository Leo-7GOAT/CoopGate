"""Independent physical reconstruction and complete shared-recovery gate replay.

Realized coefficients/innovations belong only to this post-execution checker.
The replayed gate receives the same observed state and actual previous command
as deployment, never the audit's true actuator memory or future innovations.
"""
import numpy as np

from .actuator_recovery_v34 import ActuatorRecoveryBaseline, class_bounds, stopping_certificate
from .disturbance_v9 import disturbance_tape
from .matched_shift_v21 import matched_plant_parameters, tape_fingerprint
from .merge_v4 import MergeConfig, transition_safe
from .recovery_joint_gate_v35 import RULES, RecoveryJointGate


def review_joint_trace(row, trace, model, *, replay=False):
    rule, profile = row['rule'], row['plant_name']
    if rule not in RULES or type(replay) is not bool:
        raise ValueError('Explicit shared-recovery rule and replay mode required')
    c = MergeConfig()
    if len(trace) > 50 or row['control_steps'] != len(trace):
        raise ValueError('Recorded decisions differ from the full trace')
    plant = matched_plant_parameters(row['seed'], profile, .1)
    if (row['plant'] != plant or row['physical_error_radius'] != .5
            or row['external_innovation_radius'] != .1
            or row['certificate_contract'] != 'bounded_actuator_memory_stopping'):
        raise ValueError('Changed realized plant or physical contract')
    tape = disturbance_tape(row['seed'], row['disturbance'], .1, 250)
    if row['innovation_tape_sha256'] != tape_fingerprint(tape):
        raise ValueError('Changed external innovation sequence')
    gain, drag, tau = [np.asarray(plant[k]) for k in ('gain','drag','tau')]
    retention = np.zeros(3)
    retention[tau > 0.] = np.exp(-.1/tau[tau > 0.])
    bounds = class_bounds(profile)
    for value, left, right in ((gain,0,1),(drag,2,3),(retention,4,5)):
        if np.any(value < bounds[:,left]) or np.any(value > bounds[:,right]):
            raise ValueError('Actual plant is outside the declared class')
    base = ActuatorRecoveryBaseline(profile)
    gate = RecoveryJointGate(model, profile, rule) if replay else None
    x, previous, memory = np.asarray(row['initial'], dtype=float).copy(), np.zeros(3), np.zeros(3)
    finish, variation = np.full(3,25.), np.zeros(3)
    active = x[:,0] < c.goal
    finish[~active] = 0.
    cursor = exceeded = violations = adopted = recovered = 0
    maximum = max_slew = 0.
    for step, item in enumerate(trace):
        baseline, evidence = base.select(x, previous)
        np.testing.assert_array_equal(item['baseline_command'], baseline)
        if item['baseline_evidence'] != {k:v for k,v in evidence.items() if k != 'gate_ms'}:
            raise ValueError('Recorded baseline is not the actual recovery controller')
        if (item['rule'] != rule or item['actuator_class'] != profile
                or item['actuator_bounds'] != bounds.tolist()
                or item['actuator_memory_lower'] != evidence['actuator_memory_lower']
                or item['actuator_memory_upper'] != evidence['actuator_memory_upper']
                or item['future_realization_used'] or item['true_randomized_plant_used']
                or item['margin_is_probability'] or item['conditional_performance_risk_certified']
                or not item['certified'] or not item['in_set']):
            raise ValueError('Shared online information or certificate contract changed')
        command = np.asarray(item['command'], dtype=float)
        if command.shape != (3,) or not np.isfinite(command).all():
            raise ValueError('Three finite executed commands required')
        lo, hi = np.asarray(evidence['actuator_memory_lower']), np.asarray(evidence['actuator_memory_upper'])
        if not stopping_certificate(x, previous, command, lo, hi, bounds, base.packed)[0]:
            raise ValueError('Executed command lacks the common stopping certificate')
        if np.any(memory < lo-1e-10) or np.any(memory > hi+1e-10):
            raise ValueError('True actuator memory escaped the online interval')
        if not np.array_equal(command[~active], baseline[~active]):
            raise ValueError('Cleared cars must retain the baseline command')
        used = bool(np.any(np.abs(command[active]-baseline[active]) > 1e-9))
        adopted += int(used)
        if (item['actual_learning_used'] != used or item['actual_learning_used_ever'] != (adopted > 0)
                or item['intervention'] != float(not used) or rule == 'baseline' and used):
            raise ValueError('Adoption differs from the actual active-vehicle command')
        expected_recovery = bool(not used and evidence['recovery_applied'])
        if (item['recovery_applied'] != expected_recovery
                or item['baseline_recovery_proposed'] != evidence['recovery_applied']):
            raise ValueError('Recovery proposal and actual execution were conflated')
        recovered += int(expected_recovery)
        if rule in ('nominal_performance','bank_performance') and used:
            paths = 1 if rule == 'nominal_performance' else 33
            delta = np.asarray(item['selected_bank_differences'])
            phases = np.asarray(item['selected_phase_differences'])
            if (delta.shape != (paths,3,2) or phases.shape != (paths,3)
                    or not np.isfinite(delta).all() or not np.isfinite(phases).all()
                    or np.any(delta > 1e-10) or np.any(phases > .05+1e-10)
                    or delta[0,:,0].mean() >= -1e-9):
                raise ValueError('Every admitted action needs complete finite-path evidence')
        if replay:
            actual, info = gate.select(x, previous)
            np.testing.assert_array_equal(actual, command)
            for key, value in info.items():
                if key != 'gate_ms' and item.get(key) != value:
                    raise ValueError('Complete online gate replay differs: '+key)
        np.testing.assert_allclose(item['t'], (step+1)*.5, rtol=0, atol=1e-10)
        variation += active*np.abs(command-previous)
        max_slew = max(max_slew, float(np.max(np.abs(command-previous))/.5))
        for key in ('innovation','actuator_states','disturbance'):
            values = np.asarray(item[key])
            if values.shape != (5,3) or not np.isfinite(values).all():
                raise ValueError('Missing finite physical substep evidence')
        step_error = 0.
        for sub in range(5):
            innovation = tape[cursor]
            cursor += 1
            np.testing.assert_array_equal(innovation, item['innovation'][sub])
            # Independent recurrence: do not call the environment's plant update.
            memory = retention*memory+(1.-retention)*command
            acceleration = gain*memory-drag*x[:,1]**2+innovation
            error = acceleration-command
            np.testing.assert_allclose(memory, item['actuator_states'][sub], rtol=0, atol=1e-9)
            np.testing.assert_allclose(error, item['disturbance'][sub], rtol=0, atol=1e-9)
            error_size = float(np.max(np.abs(error)))
            maximum, step_error = max(maximum,error_size), max(step_error,error_size)
            exceeded += int(error_size > .5+1e-9)
            before = x[:,0].copy()
            x[:,1] = np.clip(x[:,1]+.1*acceleration, 0., 20.)
            x[:,0] += .1*x[:,1]
            violations += not bool(transition_safe(before[None], x[:,0][None], c)[0])
        np.testing.assert_allclose(x, item['x'], rtol=0, atol=1e-8)
        if item['realized_error_in_contract'] != (step_error <= .5+1e-9):
            raise ValueError('Legacy additive-error diagnostic differs')
        x = np.asarray(item['x']).copy()
        newly = active & (x[:,0] >= c.goal)
        finish[newly], active[newly] = item['t'], False
        previous = command
    if row['aborted'] and replay:
        try:
            gate.select(x, previous)
        except (RuntimeError, ValueError) as exc:
            if str(exc) != row['abort_reason']:
                raise ValueError('Terminal gate failure identity differs') from exc
        else:
            raise ValueError('Recorded terminal gate failure did not reproduce')
    for actual, key in ((finish,'per_vehicle_finish'),(variation,'per_vehicle_variation'),
                        (maximum,'max_actual_error'),(max_slew,'max_command_slew'),(len(trace)*.5,'elapsed')):
        np.testing.assert_allclose(actual, row[key], rtol=0, atol=1e-8)
    costs = finish+.05*variation
    np.testing.assert_allclose(costs, row['per_vehicle_cost'], rtol=0, atol=1e-9)
    np.testing.assert_allclose(costs.mean(), row['team_cost'], rtol=0, atol=1e-9)
    completed = bool(not active.any() and not row['collision'] and not row['aborted'])
    if (completed != row['completed'] or exceeded != row['error_contract_exceeded_substeps']
            or violations != row['contract_violations'] or bool(row['abort_reason']) != row['aborted']
            or row['nominal_physical_assumption_observed'] != (exceeded == 0)
            or row['learned_action_steps'] != adopted):
        raise ValueError('Full outcome or learning-adoption accounting differs')
    if not completed and not row['aborted'] and len(trace) != 50:
        raise ValueError('An incomplete prefix is not a full timeout execution')
    if violations or row['collision']:
        raise ValueError('Certified execution violated the declared physical contract')
    return dict(commands=len(trace), substeps=cursor, adopted_steps=adopted,
        recovery_steps=recovered, completed=completed, aborted=bool(row['aborted']),
        legacy_error_exceeded_substeps=exceeded, physical_violations=violations,
        all_realized_memory_inside_online_intervals=True)


def relative_outcome(row, baseline):
    bad = lambda r:bool(not r['completed'] or r['aborted'] or r['collision'] or r['contract_violations'])
    baseline_failed, controller_failed = bad(baseline), bad(row)
    delta_cost = np.asarray(row['per_vehicle_cost'])-baseline['per_vehicle_cost']
    delta_time = np.asarray(row['per_vehicle_finish'])-baseline['per_vehicle_finish']
    harmful = None if baseline_failed else bool(controller_failed or np.any(delta_cost > 1e-9)
                                               or np.any(delta_time > 1e-9))
    return dict(adopted=row['learned_action_steps'] > 0, baseline_failed=baseline_failed,
        controller_failed=controller_failed, relative_harm=harmful,
        per_vehicle_cost_difference=delta_cost.tolist(), per_vehicle_time_difference=delta_time.tolist())


def cell_summary(pairs):
    if not pairs:
        raise ValueError('Nonempty paired comparison required')
    complete = [p for p in pairs if not p['baseline_failed'] and not p['controller_failed']]
    comparable = [p for p in pairs if p['adopted'] and not p['baseline_failed']]
    gain = None
    if complete:
        before = sum(p['baseline']['team_cost'] for p in complete)
        gain = 100*(before-sum(p['row']['team_cost'] for p in complete))/before
    times = [t['wall_ms'] for p in pairs for t in p['calls']]
    harms = sum(bool(p['relative_harm']) for p in comparable)
    return dict(runs=len(pairs),completed=sum(not p['controller_failed'] for p in pairs),
        baseline_completed=sum(not p['baseline_failed'] for p in pairs),
        adopted=sum(p['adopted'] for p in pairs),adopted_with_valid_baseline=len(comparable),
        harmful_adopted=harms,empirical_conditional_harm=harms/len(comparable) if comparable else None,
        adopted_with_incomplete_baseline=sum(p['adopted'] and p['baseline_failed'] for p in pairs),
        both_complete=len(complete),all_scope_cost_gain_percent=gain if len(complete)==len(pairs) else None,
        complete_pair_descriptive_gain_percent=gain,
        original_capped_mean_cost=float(np.mean([p['row']['team_cost'] for p in pairs])),
        original_capped_baseline_mean_cost=float(np.mean([p['baseline']['team_cost'] for p in pairs])),
        callback_count=len(times),terminal_callbacks=sum(t['error'] is not None for p in pairs for t in p['calls']),
        callback_sum_ms=sum(times),callback_p95_ms=float(np.quantile(times,.95)) if times else None,
        callback_max_ms=max(times) if times else None,callback_over_500ms=sum(t > 500. for t in times))
