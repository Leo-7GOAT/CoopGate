from collections import Counter
from copy import deepcopy
from pathlib import Path
import os
import sys
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import design_joint_risk_v42 as design
import run_joint_risk_v42 as runner
from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.episode_store_v21 import inventory, observations
from coopgate.joint_risk_data_v42 import (METHODS, paired_cluster, stage_arrays,
                                         certificate_from_complete_records)


@pytest.fixture(scope='module')
def plan():
    return design.build_plan()


def test_exact_sampling_design_and_disjoint_stages(plan):
    design.validate_plan(plan)
    calibration, confirmation = (plan['stages'][s] for s in design.STAGES)
    assert len(calibration) == 3072 and len(confirmation) == 1024
    assert Counter(c['plant'] for c in calibration) == {p: 768 for p in design.PLANTS}
    assert Counter(c['plant'] for c in confirmation) != {p: 256 for p in design.PLANTS}
    assert not {c['seed'] for c in calibration} & {c['seed'] for c in confirmation}
    assert not {c['seed'] for c in calibration+confirmation} & set(range(739000001, 739000641))
    assert plan['contract']['expected_sumo_runs'] == 16384
    assert plan['contract']['previous_attempt_terminal_required']
    assert plan['contract']['confirmation_runs_even_if_calibration_certificate_fails']
    assert all(set(c['methods']) == set(METHODS) and len(c['methods']) == 4 for c in calibration+confirmation)


@pytest.mark.parametrize('kind', ['case', 'missing', 'threshold', 'model', 'source'])
def test_changed_design_rejected(plan, kind):
    altered = deepcopy(plan)
    if kind == 'case': altered['stages']['confirmation'][0]['seed'] += 1
    elif kind == 'missing': altered['stages']['calibration'].pop()
    elif kind == 'threshold': altered['contract']['risk_limit'] = .1
    elif kind == 'model': altered['models']['1802']['sha256'] = '0'*64
    else: altered['source_sha256']['tools/run_joint_risk_v42.py'] = '0'*64
    with pytest.raises(ValueError):
        design.validate_plan(altered)


def row(cost=11., *, adopted=True, failed=False):
    return dict(per_vehicle_cost=[cost]*3, per_vehicle_finish=[9.]*3, team_cost=cost,
                learned_action_steps=int(adopted), completed=not failed, aborted=failed,
                collision=False, contract_violations=0)


def cluster():
    return {'baseline': row(adopted=False), **{m: row(10.8) for m in METHODS[1:]}}


def test_baseline_failure_is_preserved_and_conservatively_harmful():
    rows = cluster()
    rows['baseline'].update(completed=False, aborted=True)
    result = paired_cluster(rows)
    assert result['baseline_failed']
    assert all(result['harmful']) and all(result['baseline_unassessable'])
    assert len(result['gain_percent']) == 3


def test_one_vehicle_harm_is_not_hidden_by_team_gain():
    rows = cluster()
    rows['1802']['per_vehicle_cost'] = [11.1, 10., 10.]
    rows['1802']['team_cost'] = np.mean([11.1, 10., 10.])
    result = paired_cluster(rows)
    assert result['harmful'][0] and result['gain_percent'][0] > 0


def test_successful_zero_adoption_requires_paired_cost_identity():
    rows = cluster()
    rows['1802'] = deepcopy(rows['baseline'])
    assert not paired_cluster(rows)['adopted'][0]
    rows['1802']['per_vehicle_cost'][0] += .1
    rows['1802']['team_cost'] = np.mean(rows['1802']['per_vehicle_cost'])
    with pytest.raises(ValueError, match='Zero adoption'):
        paired_cluster(rows)


def test_failure_before_first_adoption_is_retained_not_filtered():
    rows = cluster()
    rows['1802'] = row(25., adopted=False, failed=True)
    result = paired_cluster(rows)
    assert not result['adopted'][0]
    assert result['failed'][0] and result['harmful'][0]
    assert result['gain_percent'][0] < 0


def synthetic_records(cases):
    records = []
    for case in cases:
        rows = cluster()
        for model in METHODS[1:]:
            rows[model] = row(10.8+.01*np.sin(case['index']))
        for method, outcome in rows.items():
            outcome.update(seed=case['seed'], scenario=case['scenario'], disturbance=case['disturbance'],
                plant_name=case['plant'], model='none' if method == 'baseline' else method,
                rule='baseline' if method == 'baseline' else 'bank_performance')
            records.append(dict(cluster_id=case['cluster_id'], method=method, row=outcome))
    return records


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'foreign', 'seed'])
def test_incomplete_or_mispaired_records_cannot_enter_inference(plan, mutation):
    cases = plan['stages']['confirmation'][:2]
    records = synthetic_records(cases)
    if mutation == 'missing': records.pop()
    elif mutation == 'duplicate': records.append(records[0])
    elif mutation == 'foreign': records[0]['cluster_id'] = 'foreign'
    else: records[0]['row']['seed'] = -1
    with pytest.raises(ValueError):
        stage_arrays(cases, records)


def test_complete_synthetic_adapter_uses_joint_certificate_and_rejects_failure(plan):
    calibration = synthetic_records(plan['stages']['calibration'])
    confirmation = synthetic_records(plan['stages']['confirmation'])
    success = certificate_from_complete_records(plan, calibration, confirmation)
    assert success['joint_deployment_certified']  # synthetic unit test, never a SUMO finding
    calibration[0]['row'].update(completed=False, aborted=True)
    failure = certificate_from_complete_records(plan, calibration, confirmation)
    assert not failure['joint_deployment_certified']
    assert failure['observed_execution_failures'] == 1
    assert failure['baseline_failed_pairs_imputed_harm'] == 3
    assert failure['baseline_failure_samples_removed'] == 0


def test_transaction_commits_before_replay_and_preserves_failed_prefix(plan, tmp_path):
    case = plan['stages']['calibration'][0]
    target = tmp_path/'block'
    seen = []
    class Env:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
    def run(case, method, *args):
        return dict(cluster_id=case['cluster_id'], method=method, trace=[], row={})
    def audit(case, record, model):
        seen.append(record['method'])
        assert len(inventory(target/'observations.sqlite')['cases']) == len(seen)
        raise ValueError('Injected inconsistent trace')
    kwargs = dict(plan_sha='development', binding_sha='development', run_fn=run, audit_fn=audit, env_cls=Env)
    with pytest.raises(ValueError, match='Injected'):
        runner.collect_cluster(case, target, {m: None for m in METHODS[1:]}, **kwargs)
    assert not (target/'complete.json').exists()
    assert not (target/'observations.sqlite.seal.json').exists()
    with pytest.raises(FileExistsError):
        runner.collect_cluster(case, target, {m: None for m in METHODS[1:]}, **kwargs)


def development_case(seed=139_882_001):
    from coopgate.disturbance_v9 import disturbance_tape
    from coopgate.matched_shift_v21 import matched_plant_parameters
    scene, plant, mode = 'nominal', 'lag_strong', 'clean'
    return dict(stage='development', index=0, cluster_id='development-risk-only', seed=seed,
        scenario=scene, plant=plant, disturbance=mode, methods=list(METHODS),
        initial_sha256=design.shared.array_sha256(design.shared.planned_initial(seed, scene), decimals=8),
        dynamics_sha256=design.shared.canonical_sha256(matched_plant_parameters(seed, plant, .1)),
        disturbance_sha256=design.shared.array_sha256(disturbance_tape(seed, mode, .1, 250)))


@pytest.mark.skipif(os.environ.get('COOPGATE_RUN_SUMO_SMOKE') != '1', reason='Opt-in real SUMO')
def test_real_four_method_cluster_persists_and_replays(plan, tmp_path):
    torch.set_num_threads(1)
    models = {name: load_coupled(ROOT/item['path'])[0] for name, item in plan['models'].items()}
    case = development_case()
    target = tmp_path/'sumo'
    result = runner.collect_cluster(case, target, models, plan_sha='development', binding_sha='development')
    assert result['actual_sumo_runs'] == 4
    metadata = dict(stage='development', cluster_id=case['cluster_id'],
                    plan_sha256='development', binding_sha256='development')
    stored = list(observations(target/'observations.sqlite', metadata, runner.schedule(case)))
    assert len(stored) == 4
    stage_arrays([case], [body for _, body, _ in stored])
    _, body, trace = next(item for item in stored if item[2])
    tampered = dict(**deepcopy(body), trace=deepcopy(trace))
    tampered['trace'][0]['actuator_states'][0][0] += .01
    with pytest.raises(AssertionError):
        runner.audit_record(case, tampered, None if body['method'] == 'baseline' else models[body['method']])


@pytest.mark.skipif(os.environ.get('COOPGATE_RUN_SUMO_SMOKE') != '1', reason='Opt-in real SUMO')
def test_real_pre_adoption_failure_is_durable_and_enters_failure_counts(tmp_path):
    class BrokenPolicy:
        def proposal(self, *args):
            raise RuntimeError('Injected pre-adoption policy failure')
    case = development_case(139_882_002)
    target = tmp_path/'failed_policy'
    result = runner.collect_cluster(case, target, {m: BrokenPolicy() for m in METHODS[1:]},
                                    plan_sha='fault-injection', binding_sha='fault-injection')
    assert result['actual_sumo_runs'] == 4
    metadata = dict(stage='development', cluster_id=case['cluster_id'],
                    plan_sha256='fault-injection', binding_sha256='fault-injection')
    records = [body for _, body, _ in observations(target/'observations.sqlite', metadata, runner.schedule(case))]
    arrays = stage_arrays([case], records)
    assert not arrays['baseline_failed'].any()
    assert not arrays['adopted'].any()
    assert arrays['failed'].all() and arrays['harmful'].all()
    for record in records:
        if record['method'] != 'baseline':
            assert record['row']['per_vehicle_finish'] == [25.]*3
            assert record['row']['control_steps'] == 0
            assert len(record['calls']) == 1
