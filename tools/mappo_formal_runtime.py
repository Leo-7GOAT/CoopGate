"""Owned actual SUMO execution and independently reconstructed adverse outcomes.

Generated from the original fixed-N audit;collisions/violations are outcomes,
not reasons to drop samples. No model/plant seed reaches the online gate.
"""
import time
import numpy as np
from coopgate.disturbance_v9 import disturbance_tape
from coopgate.matched_shift_v21 import MatchedShiftedMergeEnv,matched_plant_parameters,tape_fingerprint
from coopgate.memoized_recovery_v37 import MemoizedRecoveryGate,make_cache,CAPACITY
from coopgate.scalar_nominal_recovery import forecast_kernel
from coopgate.pruned_gain_margin_gate import PrunedGainMarginGate
from coopgate.pruned_gain_margin_audit import validate_prefix_evidence
from coopgate.episode_thinning_control import EpisodeThinnedNominal
from coopgate.merge_v4 import MergeConfig,transition_safe
from mappo_formal_common import TimedGate,check_calls,json_ready,array_sha256,canonical_sha256
from mappo_joint_data import episode_outcome

TARGET_METHODS=('4703','4704','4705')
CALIBRATION_METHODS=('baseline',)+TARGET_METHODS+tuple('nominal_'+s for s in TARGET_METHODS)
CONFIRMATION_METHODS=CALIBRATION_METHODS+tuple(kind+'_'+s for kind in ('thinned','mps','mamps') for s in TARGET_METHODS)
METHODS=CONFIRMATION_METHODS


class ScalarNominalOnlyGate(MemoizedRecoveryGate):
    def __init__(self,model,profile):super().__init__(model,profile,'nominal_performance')
    def predict(self,state,previous,mlo,mhi,command,path):
        if getattr(self,'_cache_step',None)!=self.step:
            self._plan_cache,self._cache_stats=make_cache();self._cache_step=self.step
        return forecast_kernel(state,previous,self.memory,mlo,mhi,self.step,self.tapes[path],
            self.gain,self.drag,self.retention,self.bounds,self.packed,
            np.zeros(3) if command is None else command,command is not None,
            self._plan_cache,self._cache_stats,CAPACITY)


def nominal_factory(model,profile,rule):
    if rule=='baseline':return MemoizedRecoveryGate(None,profile,'baseline')
    if rule=='nominal_performance':return ScalarNominalOnlyGate(model,profile)
    raise ValueError('Only nominal or identical baseline branches are registered')


def truthful_rule(method,case):
    if method not in METHODS:raise ValueError('Unknown fixed method')
    if method=='baseline':return 'baseline'
    if method in TARGET_METHODS:return 'parameter_scenario_confidence'
    kind,seed=method.split('_')
    if kind=='nominal':return 'nominal_performance'
    if kind=='thinned':return 'nominal_performance' if case['controller_coins'][seed] else 'baseline'
    return 'mps_shared' if kind=='mps' else 'mamps_shared'


def make_controller(method,model,case):
    profile=case['plant'];rule=truthful_rule(method,case)
    if method in TARGET_METHODS:return PrunedGainMarginGate(model,profile,.0025)
    if method.startswith('thinned_'):
        seed=method.split('_')[1]
        return EpisodeThinnedNominal(model,profile,case['controller_coins'][seed],factory=nominal_factory)
    if method.startswith('nominal_'):return ScalarNominalOnlyGate(model,profile)
    return MemoizedRecoveryGate(model if method!='baseline' else None,profile,rule)

def execute_episode(case, method, model, env, *, controller_factory=make_controller):
    rule = truthful_rule(method, case)
    wrapped = TimedGate(controller_factory(method, model, case))
    started = time.perf_counter()
    env.reset_disturbed(case['seed'], case['scenario'], case['disturbance'], 0.5)
    initial = env.x.copy()
    env.gate = wrapped
    while True:
        try:
            _, _, done, truncated, _ = env.step([4] * 3)
        except Exception as exc:
            if wrapped.calls and wrapped.calls[-1]['error'] is not None:
                env.abort_reason=str(exc)
                break
            raise
        if done or truncated:
            break
    row, trace = (env.metrics(), env.trace)
    previous, state, variation = (np.zeros(3), initial.copy(), np.zeros(3))
    for item in trace:
        command = np.array(item['command'])
        variation += (state[:, 0] < env.c.goal) * np.abs(command - previous)
        previous, state = (command, np.array(item['x']))
    finish = np.nan_to_num(env.finish, nan=env.c.horizon)
    cost = finish + 0.05 * variation
    row.update(initial=initial.tolist(), plant_name=case['plant'], rule=rule, model='none' if method == 'baseline' else method, certificate_contract='bounded_actuator_memory_stopping', per_vehicle_cost=cost.tolist(), team_cost=float(cost.mean()), per_vehicle_finish=finish.tolist(), per_vehicle_variation=variation.tolist(), control_steps=len(trace), learned_action_steps=sum((bool(t['actual_learning_used']) for t in trace)))
    return json_ready(dict(cluster_id=case['cluster_id'], method=method, row=row, trace=trace, calls=wrapped.calls, episode_wall_seconds=time.perf_counter() - started))

def audit_record(case, record, model, *, reference_factory=make_controller):
    """Retain genuine collisions/violations; reject inconsistent evidence.

Unlike the old development auditor, an accurately recorded physical failure is
an outcome, not a reason to silently lose the completed study's sample.
SUMO collision flags are retained; route geometry is independently reconstructed.
"""
    row, trace = (record['row'], record['trace'])
    method = record['method']
    rule = truthful_rule(method, case)
    if method not in METHODS or record['cluster_id'] != case['cluster_id'] or row['seed'] != case['seed'] or (row['scenario'] != case['scenario']) or (row['disturbance'] != case['disturbance']) or (row['plant_name'] != case['plant']) or (row['rule'] != rule) or (row['model'] != ('none' if method == 'baseline' else method)) or (row['control_steps'] != len(trace)) or (len(trace) > 50):
        raise ValueError('Executed case identity or trace length changed')
    check_calls(row, trace, record['calls'])
    plant = matched_plant_parameters(case['seed'], case['plant'], 0.1)
    tape = disturbance_tape(case['seed'], case['disturbance'], 0.1, 250)
    x = np.asarray(row['initial'], dtype=float)
    if x.shape != (3, 2) or not np.isfinite(x).all() or row['plant'] != plant or (array_sha256(x, decimals=8) != case['initial_sha256']) or (canonical_sha256(plant) != case['dynamics_sha256']) or (array_sha256(tape) != case['disturbance_sha256']) or (row['innovation_tape_sha256'] != tape_fingerprint(tape)) or (row['physical_error_radius'] != 0.5) or (row['external_innovation_radius'] != 0.1):
        raise ValueError('Physical pairing inputs changed')
    reference = reference_factory(method, model, case)
    memory, previous, variation = (np.zeros(3), np.zeros(3), np.zeros(3))
    finish, active = (np.full(3, 25.0), x[:, 0] < 60)
    finish[~active] = 0.0
    tau = np.asarray(plant['tau'])
    retention = np.zeros(3)
    retention[tau > 0] = np.exp(-0.1 / tau[tau > 0])
    gain, drag = (np.asarray(plant['gain']), np.asarray(plant['drag']))
    cursor = violations = exceeded = adopted = 0
    max_error = max_slew = 0.0
    for step, entry in enumerate(trace):
        if method in TARGET_METHODS:
            validate_prefix_evidence(entry)
        expected, info = reference.select(x, previous)
        command = np.asarray(entry['command'])
        if command.shape != (3,) or not np.isfinite(command).all() or np.any(command < -5.0 - 1e-10) or np.any(command > 2.5 + 1e-10) or np.any(np.abs(command - previous) > 2.5 + 1e-10):
            raise ValueError('Invalid executed command or slew')
        np.testing.assert_array_equal(command, expected)
        for key, value in info.items():
            if key != 'gate_ms' and entry.get(key) != value:
                raise ValueError('Uncached reference gate replay differs: ' + key)
        used = bool(np.any(np.abs(command[active] - np.asarray(info['baseline_command'])[active]) > 1e-09))
        if entry['actual_learning_used'] != used:
            raise ValueError('Adoption differs from executed active commands')
        adopted += used
        variation += active * np.abs(command - previous)
        max_slew = max(max_slew, float(np.max(np.abs(command - previous))) / 0.5)
        for key in ('innovation', 'actuator_states', 'disturbance'):
            values = np.asarray(entry[key])
            if values.shape != (5, 3) or not np.isfinite(values).all():
                raise ValueError('Incomplete finite physical substeps')
        step_error = 0.0
        for sub in range(5):
            np.testing.assert_array_equal(entry['innovation'][sub], tape[cursor])
            memory = retention * memory + (1 - retention) * command
            applied = gain * memory - drag * x[:, 1] ** 2 + tape[cursor]
            np.testing.assert_allclose(entry['actuator_states'][sub], memory, atol=1e-09, rtol=0)
            np.testing.assert_allclose(entry['disturbance'][sub], applied - command, atol=1e-09, rtol=0)
            error = float(np.max(np.abs(applied - command)))
            max_error, step_error = (max(max_error, error), max(step_error, error))
            exceeded += int(error > 0.5 + 1e-09)
            before = x.copy()
            x[:, 1] = np.clip(x[:, 1] + 0.1 * applied, 0, 20)
            x[:, 0] += 0.1 * x[:, 1]
            violations += not bool(transition_safe(before[:, 0][None], x[:, 0][None], MergeConfig())[0])
            cursor += 1
        np.testing.assert_allclose(entry['x'], x, atol=1e-08, rtol=0)
        np.testing.assert_allclose(entry['t'], (step + 1) * 0.5, atol=1e-10, rtol=0)
        if entry['realized_error_in_contract'] != (step_error <= 0.5 + 1e-09):
            raise ValueError('Realized error indicator changed')
        x = np.array(entry['x'])
        newly = active & (x[:, 0] >= 60)
        finish[newly], active[newly] = (entry['t'], False)
        previous = command
    if row['aborted']:
        try:
            reference.select(x, previous)
        except Exception as exc:
            if str(exc) != row['abort_reason']:
                raise ValueError('Terminal error differs from reference') from exc
        else:
            raise ValueError('Terminal error did not replay')
    elif active.any() and (not row['collision']) and (len(trace) != 50):
        raise ValueError('Incomplete prefix is not a full episode')
    for value, name in ((finish, 'per_vehicle_finish'), (variation, 'per_vehicle_variation'), (finish + 0.05 * variation, 'per_vehicle_cost'), (max_error, 'max_actual_error'), (max_slew, 'max_command_slew'), (len(trace) * 0.5, 'elapsed')):
        np.testing.assert_allclose(row[name], value, atol=1e-09, rtol=0)
    if row['contract_violations'] != violations or row['learned_action_steps'] != adopted or row['error_contract_exceeded_substeps'] != exceeded or (row['nominal_physical_assumption_observed'] != (exceeded == 0)) or (row['completed'] != bool(not active.any() and (not row['aborted']) and (not row['collision']))):
        raise ValueError('Full episode accounting changed')
    outcome = episode_outcome(row)
    return dict(physical_substeps=cursor, reference_replayed_commands=len(trace), recorded_failure=outcome['failed'], genuine_failures_retained=True, runtime_independently_certified=False)
