from copy import deepcopy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import freeze_fair_comparison_v39 as planner
import run_fair_runtime_v41 as runner
from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.fair_comparison_v39 import CaseIdentity, METHODS, FairComparisonProtocol, fingerprint
from coopgate.fair_runtime_v41 import EvidenceFactory
from coopgate.matched_shift_v21 import matched_plant_parameters
from coopgate.disturbance_v9 import disturbance_tape
from coopgate.recovery_joint_gate_v35 import RecoveryJointGate
from coopgate.episode_store_v21 import inventory, observations


@pytest.fixture(scope='module')
def setup():
    torch.set_num_threads(1)
    plan = planner.build_plan()
    spec = planner.protocol_from_dict(plan['protocol'])
    model_id = '1802'
    model, _ = load_coupled(ROOT / plan['models'][model_id]['path'])
    calibration = json.loads((ROOT/planner.CALIBRATION).read_text())['701']
    seed, scene, plant, mode = 139_881_001, 'nominal', 'nominal', 'clean'
    x = planner.planned_initial(seed, scene)
    case = CaseIdentity(case_id='development-fair-runtime', independent_cluster='development-only',
        policy_id=model_id, policy_sha256=plan['models'][model_id]['sha256'], scenario=scene,
        initial_id=seed, dynamics=plant, disturbance=mode,
        initial_sha256=planner.array_sha256(x, decimals=8),
        dynamics_sha256=planner.canonical_sha256(matched_plant_parameters(seed, plant, .1)),
        disturbance_sha256=planner.array_sha256(disturbance_tape(seed, mode, .1, 250)))
    return case, spec, model, calibration, x


def test_real_policy_evidence_and_bank_command_replay(setup):
    case, spec, model, calibration, x = setup
    # Unlimited DEVELOPMENT timing checks mathematical identity, not deadline performance.
    spec = replace(spec, budget=replace(spec.budget, decision_timeout_ms=1e9))
    factory = EvidenceFactory(case, spec, model, calibration)
    evidence, proof, hit, key = factory.get(x, [])
    expected, info = RecoveryJointGate(model, case.dynamics).select(x, np.zeros(3))
    assert not hit and factory.actual_builds == 1
    np.testing.assert_array_equal(evidence.proposed_bank_command, expected)
    assert proof['details']['bank']['info']['forecast_calls'] == info['forecast_calls']
    assert proof['details']['raw_kernel_calls'] == 66
    assert evidence.ledger.forecast_paths == 33
    assert evidence.ledger.learned_proposal_calls == evidence.ledger.baseline_proposal_calls == 1
    assert factory.audit(evidence, proof)['replayed']
    again, same, hit, key2 = factory.get(x, [])
    assert hit and key == key2 and fingerprint(again) == fingerprint(evidence)
    assert factory.actual_builds == 1 and factory.cache_hits == 1
    broken = deepcopy(proof)
    broken['details']['raw_paths'][0]['advantage_s'] += .1
    with pytest.raises(ValueError, match='replay differs'):
        factory.audit(evidence, broken)


def test_shared_timeout_uses_real_baseline_and_replays_for_all_methods(setup):
    case, spec, model, calibration, x = setup
    factory = EvidenceFactory(case, spec, model, calibration)
    evidence, record = factory.build(x, [], replay_clock=[501.])
    assert evidence.ledger.timed_out
    assert evidence.ledger.baseline_proposal_calls == 1
    assert evidence.ledger.learned_proposal_calls == 0
    assert not evidence.proposed_bank_admitted
    assert factory.audit(evidence, record)['timeout']
    decisions = FairComparisonProtocol(spec).select_all(evidence)
    assert len(decisions) == 8
    assert all(not decision.adopted and decision.command == evidence.baseline for decision in decisions)
    assert len({decision.evidence_sha256 for decision in decisions}) == 1


def test_nonzero_command_history_matches_stateful_bank_gate(setup):
    case, spec, model, calibration, x = setup
    spec = replace(spec, budget=replace(spec.budget, decision_timeout_ms=1e9))
    gate = RecoveryJointGate(model, case.dynamics)
    command, _ = gate.select(x, np.zeros(3))
    after = x.copy()
    for _ in range(5):
        after[:, 1] = np.clip(after[:, 1] + .1*command, 0, 20)
        after[:, 0] += .1*after[:, 1]
    expected, original = gate.select(after, command)
    factory = EvidenceFactory(case, spec, model, calibration)
    evidence, proof = factory.build(after, [command.tolist()])
    np.testing.assert_array_equal(evidence.proposed_bank_command, expected)
    assert proof['details']['bank']['info']['reconstructed_actuator'] == original['reconstructed_actuator']
    assert factory.audit(evidence, proof)['replayed']


@pytest.mark.parametrize('profile', ['nominal', 'lag_strong'])
def test_multiple_executed_commands_restore_actuator_and_interval_memory(setup, profile):
    case, spec, model, calibration, x = setup
    case = replace(case, dynamics=profile,
        dynamics_sha256=planner.canonical_sha256(matched_plant_parameters(case.initial_id, profile, .1)))
    spec = replace(spec, budget=replace(spec.budget, decision_timeout_ms=1e9))
    gate = RecoveryJointGate(model, profile)
    factory = EvidenceFactory(case, spec, model, calibration)
    plant = matched_plant_parameters(case.initial_id, profile, .1)
    tau = np.array(plant['tau'])
    retention = np.zeros(3)
    retention[tau > 0] = np.exp(-.1/tau[tau > 0])
    history, actuator, state = [], np.zeros(3), x.copy()
    for _ in range(4):
        previous = np.array(history[-1]) if history else np.zeros(3)
        expected, original = gate.select(state, previous)
        evidence, proof = factory.build(state, history)
        np.testing.assert_array_equal(evidence.proposed_bank_command, expected)
        for field in ('reconstructed_actuator', 'actuator_memory_lower', 'actuator_memory_upper'):
            assert proof['details']['bank']['info'][field] == original[field]
        history.append(expected.tolist())
        for _ in range(5):
            actuator = retention*actuator+(1-retention)*expected
            acceleration = np.array(plant['gain'])*actuator-np.array(plant['drag'])*state[:, 1]**2
            state[:, 1] = np.clip(state[:, 1]+.1*acceleration, 0, 20)
            state[:, 0] += .1*state[:, 1]


def one_case_plan(setup):
    case, spec, model, calibration, _ = setup
    group = dict(block_index=0, actual_sumo_runs=8, case_ids=[case.case_id], assignments=[
        dict(case_id=case.case_id, method=method, order=i) for i, method in enumerate(METHODS)])
    plan = dict(cases=[asdict(case)], protocol=asdict(spec),
                calibration=dict(historical_model_key_map={'1802': '701'}))
    return plan, group, {'1802': model}, {'701': calibration}


def test_failed_audit_preserves_committed_unsealed_prefix(setup, tmp_path):
    plan, group, models, calibrations = one_case_plan(setup)
    target = tmp_path/'block'
    seen = []

    class DummyEnv:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass

    def run(case, method, *args):
        return dict(method=method, trace=[])

    def audit(record, *args):
        assert len(inventory(target/'observations.sqlite')['cases']) == len(seen)+1
        seen.append(record['method'])
        if len(seen) == 2:
            raise ValueError('Injected audit failure')
        return dict(replayed=True)

    with pytest.raises(ValueError, match='Injected audit failure'):
        runner.collect_block(plan, group, target, models, calibrations, 'plan', 'binding',
                             run_fn=run, audit_fn=audit, env_cls=DummyEnv)
    assert len(inventory(target/'observations.sqlite')['cases']) == 2
    assert not (target/'observations.sqlite.seal.json').exists()
    assert not (target/'complete.json').exists()
    with pytest.raises(FileExistsError):
        runner.collect_block(plan, group, target, models, calibrations, 'plan', 'binding',
                             run_fn=run, audit_fn=audit, env_cls=DummyEnv)


@pytest.mark.skipif(os.environ.get('COOPGATE_RUN_SUMO_SMOKE') != '1', reason='Opt-in actual SUMO')
def test_all_eight_selectors_execute_and_replay_in_real_sumo(setup, tmp_path):
    case, spec, model, calibration, _ = setup
    plan, group, models, calibrations = one_case_plan(setup)
    target = tmp_path/'real_block'
    result = runner.collect_block(plan, group, target, models, calibrations, 'development', 'development')
    assert result['actual_sumo_runs'] == 8
    assert result['cached_packet_uses'] > 0
    assert result['independently_replayed_packets'] == result['unique_packets_computed']
    metadata = dict(plan_sha256='development', binding_sha256='development', block_index=0)
    rows = list(observations(target/'observations.sqlite', metadata, runner.block_schedule(group)))
    assert {body['episode']['method'] for _, body, _ in rows} == set(METHODS)
    from coopgate.fair_comparison_v39 import summarize_matched_records
    summarize_matched_records([runner.decode_episode(body['episode']) for _, body, _ in rows], spec)
    body, trace = next((body, trace) for _, body, trace in rows if trace)
    record = dict(**body, trace=trace)
    factory = EvidenceFactory(case, spec, model, calibration)
    for field, value in [('max_actual_error', 99.), ('elapsed', -1.),
                         ('error_contract_exceeded_substeps', -1), ('seed', -1)]:
        changed = deepcopy(record)
        changed['simulator'][field] = value
        with pytest.raises((ValueError, AssertionError)):
            runner.audit_episode(changed, factory)
    changed = deepcopy(record)
    changed['trace'][0]['command'][0] = 9.
    with pytest.raises(ValueError, match='command or slew'):
        runner.audit_episode(changed, factory)


@pytest.mark.parametrize('change', ['count', 'duplicate', 'unknown', 'missing_method'])
def test_bad_schedule_rejected_before_creating_a_block(setup, tmp_path, change):
    plan, group, models, calibrations = one_case_plan(setup)
    if change == 'count':
        group['actual_sumo_runs'] = 7
    elif change == 'duplicate':
        group['case_ids'] *= 2
    elif change == 'unknown':
        group['case_ids'] = ['absent']
    else:
        group['assignments'][-1]['method'] = group['assignments'][0]['method']
    target = tmp_path/'invalid'
    with pytest.raises(ValueError, match='exactly eight'):
        runner.collect_block(plan, group, target, models, calibrations, 'p', 'b')
    assert not target.exists()
