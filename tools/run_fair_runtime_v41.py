"""SUMO execution and replay of an eight-method case (no formal launch yet)."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from coopgate.disturbance_v9 import disturbance_tape
from coopgate.episode_store_v21 import EpisodeStore
from coopgate.fair_comparison_v39 import (COST_CONTRACT, METHODS, CaseIdentity, DecisionEvidence,
    EpisodeRecord, FairComparisonProtocol, WorkLedger, fingerprint, summarize_matched_records)
from coopgate.fair_runtime_v41 import EvidenceFactory, FairSelectorGate
from coopgate.matched_shift_v21 import MatchedShiftedMergeEnv, matched_plant_parameters, tape_fingerprint
from coopgate.merge_v4 import MergeConfig, transition_safe
import freeze_fair_comparison_v39 as plan_builder
import execute_ood_validation_v39 as persistence


def json_ready(value):
    return json.loads(json.dumps(value, allow_nan=False,
        default=lambda item: item.item() if isinstance(item, np.generic) else item.tolist()))


def decode_evidence(value):
    value = dict(value)
    value['case'] = CaseIdentity(**value['case'])
    value['ledger'] = WorkLedger(**value['ledger'])
    for key in ('proposal', 'baseline', 'active', 'advantage_samples_s', 'proposed_bank_command'):
        value[key] = tuple(value[key])
    value['safety_flags_by_mask'] = tuple(tuple(row) for row in value['safety_flags_by_mask'])
    return DecisionEvidence(**value)


def finish_record(case, method, spec, env, gate):
    row = env.metrics()
    row.update(initial=env.initial.tolist(), plant_name=case.dynamics)
    previous = np.zeros(3)
    active = np.asarray(env.initial)[:, 0] < env.c.goal
    variation = np.zeros(3)
    for entry in env.trace:
        command = np.array(entry['command'])
        variation += active * np.abs(command - previous)
        active &= np.array(entry['x'])[:, 0] < env.c.goal
        previous = command
    arrivals = np.nan_to_num(env.finish, nan=env.c.horizon)
    costs = arrivals + .05 * variation
    timed_out = any(call['error'] is None and call['evidence']['ledger']['timed_out'] for call in gate.calls)
    episode = EpisodeRecord(case=case, method=method, protocol_sha256=fingerprint(spec),
        cost_contract=COST_CONTRACT, timeout_ms=spec.budget.decision_timeout_ms,
        adopted=any(entry['actual_learning_used'] for entry in env.trace),
        completed=bool(np.all(np.isfinite(env.finish)) and not row['collision'] and not row['aborted']),
        timed_out=bool(timed_out), collision=bool(row['collision']), aborted=bool(row['aborted']),
        contract_violations=int(row['contract_violations']), per_vehicle_cost=tuple(costs),
        per_vehicle_arrival=tuple(arrivals), team_cost=float(costs.mean()))
    return json_ready(dict(episode=asdict(episode), simulator=row, calls=gate.calls, trace=env.trace,
                           per_vehicle_variation=variation.tolist()))


def run_episode(case, method, spec, factory, env):
    env.reset_disturbed(case.initial_id, case.scenario, case.disturbance, .5)
    if (plan_builder.array_sha256(env.initial, decimals=8) != case.initial_sha256
            or plan_builder.canonical_sha256(env.plant) != case.dynamics_sha256
            or plan_builder.array_sha256(env.tape) != case.disturbance_sha256):
        raise ValueError('Actual SUMO initial/dynamics/tape differs from case identity')
    gate = FairSelectorGate(method, factory)
    env.gate = gate
    while True:
        _, _, done, truncated, _ = env.step([0, 0, 0])
        if done or truncated:
            break
    return finish_record(case, method, spec, env, gate)


def decode_episode(value):
    value = dict(value)
    value['case'] = CaseIdentity(**value['case'])
    value['per_vehicle_cost'] = tuple(value['per_vehicle_cost'])
    value['per_vehicle_arrival'] = tuple(value['per_vehicle_arrival'])
    return EpisodeRecord(**value)


def audit_episode(record, factory, verified_packets=None):
    """Independent physical recurrence plus deterministic packet/selector replay."""
    verified_packets = {} if verified_packets is None else verified_packets
    episode = decode_episode(record['episode'])
    case, row, trace, calls = episode.case, record['simulator'], record['trace'], record['calls']
    if case != factory.case or episode.protocol_sha256 != fingerprint(factory.spec):
        raise ValueError('Recorded case/protocol differs')
    if len(trace) > 50 or len(calls) != len(trace) + int(episode.aborted):
        raise ValueError('Incomplete successful/aborted decision calls')
    plant = matched_plant_parameters(case.initial_id, case.dynamics, .1)
    tape = disturbance_tape(case.initial_id, case.disturbance, .1, 250)
    x = np.asarray(row['initial'], dtype=float)
    if (x.shape != (3, 2) or not np.isfinite(x).all()
            or row['seed'] != case.initial_id or row['plant_name'] != case.dynamics
            or row['disturbance'] != case.disturbance or row['external_innovation_radius'] != .1
            or row['physical_error_radius'] != .5 or row['radius'] != .5
            or row['innovation_tape_sha256'] != tape_fingerprint(tape)
            or row['plant'] != plant or plan_builder.array_sha256(x, decimals=8) != case.initial_sha256
            or plan_builder.canonical_sha256(plant) != case.dynamics_sha256
            or plan_builder.array_sha256(tape) != case.disturbance_sha256):
        raise ValueError('Changed paired physical inputs')
    previous, memory = np.zeros(3), np.zeros(3)
    history, active = [], x[:, 0] < 60
    arrival, variation = np.full(3, 25.), np.zeros(3)
    tau = np.asarray(plant['tau'])
    retention = np.zeros(3)
    retention[tau > 0] = np.exp(-.1 / tau[tau > 0])
    gain, drag = np.asarray(plant['gain']), np.asarray(plant['drag'])
    violations = cursor = adopted = timeouts = exceeded = 0
    maximum_error = max_slew = 0.
    for step, (entry, call) in enumerate(zip(trace, calls)):
        if call['error'] is not None:
            raise ValueError('A successful transition cannot have a failed call')
        evidence = decode_evidence(call['evidence'])
        proof = call['proof']
        np.testing.assert_allclose(proof['observed'], x, atol=1e-8, rtol=0)
        if proof['history'] != history or evidence.step != step:
            raise ValueError('Evidence used a different executed command history')
        key = fingerprint(dict(case=asdict(case), observed=proof['observed'], executed_commands=history))
        if key != call['packet_key'] or key != entry['evidence_key']:
            raise ValueError('Incorrect shared evidence identity')
        proof_hash = fingerprint(dict(evidence=call['evidence'], proof=proof))
        if key in verified_packets:
            if verified_packets[key] != proof_hash:
                raise ValueError('One shared state received different evidence')
        else:
            factory.audit(evidence, proof)
            verified_packets[key] = proof_hash
        decision = FairComparisonProtocol(factory.spec).select(episode.method, evidence)
        if fingerprint(asdict(decision)) != fingerprint(call['decision']):
            raise ValueError('Selector decision replay differs')
        command = np.array(entry['command'])
        if (command.shape != (3,) or not np.isfinite(command).all()
                or np.any(command < -5.-1e-10) or np.any(command > 2.5+1e-10)
                or np.any(np.abs(command-previous) > 2.5+1e-10)):
            raise ValueError('Invalid executed command or slew')
        np.testing.assert_array_equal(command, decision.command)
        np.testing.assert_array_equal(entry['baseline_command'], evidence.baseline)
        if (entry['evidence_sha256'] != decision.evidence_sha256
                or entry['actual_learning_used'] != decision.adopted
                or entry['common_evidence_timeout'] != evidence.ledger.timed_out):
            raise ValueError('Recorded gate outcome differs')
        variation += active * np.abs(command - previous)
        max_slew = max(max_slew, float(np.max(np.abs(command-previous)))/.5)
        adopted += decision.adopted
        timeouts += evidence.ledger.timed_out
        for name in ('innovation', 'actuator_states', 'disturbance'):
            values = np.asarray(entry[name])
            if values.shape != (5, 3) or not np.isfinite(values).all():
                raise ValueError('Incomplete finite physical substeps: '+name)
        step_error = 0.
        for sub in range(5):
            np.testing.assert_array_equal(entry['innovation'][sub], tape[cursor])
            memory = retention * memory + (1 - retention) * command
            acceleration = gain * memory - drag * x[:, 1]**2 + tape[cursor]
            error_size = float(np.max(np.abs(acceleration-command)))
            step_error = max(step_error, error_size)
            maximum_error = max(maximum_error, error_size)
            exceeded += int(error_size > .5+1e-9)
            np.testing.assert_allclose(entry['actuator_states'][sub], memory, atol=1e-9, rtol=0)
            np.testing.assert_allclose(entry['disturbance'][sub], acceleration-command, atol=1e-9, rtol=0)
            before = x.copy()
            x[:, 1] = np.clip(x[:, 1] + .1*acceleration, 0, 20)
            x[:, 0] += .1*x[:, 1]
            violations += not bool(transition_safe(before[:, 0][None], x[:, 0][None], MergeConfig())[0])
            cursor += 1
        if entry['realized_error_in_contract'] != (step_error <= .5+1e-9):
            raise ValueError('Realized error contract flag differs')
        np.testing.assert_allclose(entry['x'], x, atol=1e-8, rtol=0)
        np.testing.assert_allclose(entry['t'], (step+1)*.5, atol=1e-10, rtol=0)
        x = np.array(entry['x'])  # SUMO readback is the next observed input
        newly = active & (x[:, 0] >= 60)
        arrival[newly] = entry['t']
        active[newly] = False
        previous = command
        history.append(command.tolist())
    if episode.aborted:
        error = calls[-1]['error']
        if error is None or error['history'] != history:
            raise ValueError('Missing terminal error history')
        np.testing.assert_allclose(error['observed'], x, atol=1e-8, rtol=0)
        expected, replay = factory.build(error['observed'], history, replay_clock=error['checkpoints_ms'])
        if expected is not None or fingerprint(replay) != fingerprint(error):
            raise ValueError('Terminal error cannot be replayed')
        key = fingerprint(dict(case=asdict(case), observed=error['observed'], executed_commands=history))
        proof_hash = fingerprint(dict(evidence=None, proof=error))
        if key in verified_packets and verified_packets[key] != proof_hash:
            raise ValueError('One failed shared state received different evidence')
        verified_packets[key] = proof_hash
    elif len(trace) < 50 and active.any() and not episode.collision:
        raise ValueError('Truncated prefix is not a completed episode')
    np.testing.assert_allclose(episode.per_vehicle_arrival, arrival, atol=1e-9, rtol=0)
    np.testing.assert_allclose(episode.per_vehicle_cost, arrival+.05*variation, atol=1e-9, rtol=0)
    np.testing.assert_allclose(episode.team_cost, np.mean(arrival+.05*variation), atol=1e-9, rtol=0)
    for actual, name in ((maximum_error, 'max_actual_error'), (max_slew, 'max_command_slew'),
                         (len(trace)*.5, 'elapsed')):
        np.testing.assert_allclose(actual, row[name], atol=1e-9, rtol=0)
    np.testing.assert_allclose(record['per_vehicle_variation'], variation, atol=1e-9, rtol=0)
    if (episode.adopted != bool(adopted) or episode.timed_out != bool(timeouts)
            or episode.contract_violations != violations
            or row['contract_violations'] != violations
            or row['error_contract_exceeded_substeps'] != exceeded
            or row['nominal_physical_assumption_observed'] != (exceeded == 0)
            or bool(row['abort_reason']) != episode.aborted
            or episode.completed != bool(not active.any() and not episode.collision and not episode.aborted)
            or episode.collision != bool(row['collision']) or episode.aborted != bool(row['aborted'])):
        raise ValueError('Episode outcome accounting differs')
    return dict(physical_substeps=cursor, replayed_decisions=len(trace),
                all_actual_failures_retained=True, wall_time_independently_certified=False)


def block_schedule(group):
    return [[item['case_id'], item['method']] for item in group['assignments']]


def collect_block(plan, group, directory, models, calibrations, plan_sha, binding_sha,
                  *, run_fn=run_episode, audit_fn=audit_episode, env_cls=MatchedShiftedMergeEnv):
    """Exact paired collection: transaction commits BEFORE any replay audit.

    A failed audit leaves the immutable unsealed prefix on disk. No resume,
    seed rejection, sample substitution, or partial-effect summarization.
    """
    directory = Path(directory)
    cases = {value['case_id']: CaseIdentity(**value) for value in plan['cases']}
    if (len(cases) != len(plan['cases']) or len(set(group['case_ids'])) != len(group['case_ids'])
            or not set(group['case_ids']) <= set(cases)
            or group['actual_sumo_runs'] != len(group['assignments'])
            or set(group['case_ids']) != {item['case_id'] for item in group['assignments']}
            or any(sorted(item['method'] for item in group['assignments'] if item['case_id'] == key)
                   != sorted(METHODS) for key in group['case_ids'])):
        raise ValueError('A block must retain exactly eight methods per case')
    directory.mkdir(exist_ok=False)
    spec = plan_builder.protocol_from_dict(plan['protocol'])
    metadata = dict(plan_sha256=plan_sha, binding_sha256=binding_sha, block_index=group['block_index'])
    store = EpisodeStore(directory/'observations.sqlite', metadata)
    reviews, factories, verified = [], {}, {}
    try:
        for item in group['assignments']:
            case, method = cases[item['case_id']], item['method']
            if case.case_id not in factories:
                factories[case.case_id] = EvidenceFactory(case, spec, models[case.policy_id],
                    calibrations[plan['calibration']['historical_model_key_map'][case.policy_id]])
                verified[case.case_id] = {}
            factory = factories[case.case_id]
            with env_cls(plant=case.dynamics) as env:
                record = run_fn(case, method, spec, factory, env)
            trace = record['trace']
            body = {key: value for key, value in record.items() if key != 'trace'}
            store.append([case.case_id, method], body, trace)
            reviews.append(dict(case_id=case.case_id, method=method,
                **audit_fn(record, factory, verified[case.case_id])))
        store.seal(block_schedule(group))
    finally:
        store.close()
    receipt = dict(**metadata, actual_sumo_runs=len(reviews), reviews=reviews,
        unique_packets_computed=sum(f.actual_builds for f in factories.values()),
        cached_packet_uses=sum(f.cache_hits for f in factories.values()),
        independently_replayed_packets=sum(len(v) for v in verified.values()),
        database_sha256=persistence.digest(directory/'observations.sqlite'),
        seal_sha256=persistence.digest(directory/'observations.sqlite.seal.json'))
    persistence.exclusive_json(directory/'complete.json', receipt)
    return receipt


def run_matched_case(case, spec, model, calibration, *, method_order=METHODS):
    if len(method_order) != len(METHODS) or set(method_order) != set(METHODS):
        raise ValueError('Every case requires the complete eight-method family')
    factory = EvidenceFactory(case, spec, model, calibration)
    records, reviews, verified = [], [], {}
    with MatchedShiftedMergeEnv(plant=case.dynamics) as env:
        for method in method_order:
            record = run_episode(case, method, spec, factory, env)
            # The future durable worker must persist this record BEFORE audit.
            records.append(record)
            reviews.append(audit_episode(record, factory, verified))
    summary = summarize_matched_records([decode_episode(r['episode']) for r in records], spec)
    return dict(records=records, reviews=reviews, summary=summary,
        actual_sumo_runs=len(records), unique_packets_computed=factory.actual_builds,
        cached_packet_uses=factory.cache_hits, independently_replayed_packets=len(verified))
