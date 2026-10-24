import os
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import execute_ood_validation_v39 as execute
import audit_ood_validation_v39 as auditor
from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.episode_store_v21 import observations
from coopgate.ood_mismatch_v39 import OODMismatchMergeEnv
from coopgate.bounded_observation_v45 import project_velocity, ProjectedObservationGate
import coopgate.bounded_observation_v45 as projection


def test_projection_is_identity_in_domain_preserves_positions_and_does_not_mutate():
    raw = np.array([[-100., 0.], [-120., 20.], [-140., 7.5]])
    np.testing.assert_array_equal(project_velocity(raw), raw)
    outside = raw.copy()
    outside[:, 1] = [-2., 20.199, 7.5]
    before = outside.copy()
    result = project_velocity(outside)
    np.testing.assert_array_equal(outside, before)
    np.testing.assert_array_equal(result[:, 0], outside[:, 0])
    np.testing.assert_array_equal(result[:, 1], [0., 20., 7.5])


def test_projection_cannot_increase_velocity_error_when_truth_is_in_known_interval():
    rng = np.random.default_rng(450045)
    truth = rng.uniform(0., 20., (10000, 3))
    noisy = truth+rng.normal(0., 40., truth.shape)
    projected = np.clip(noisy, 0., 20.)
    assert np.all(np.abs(projected-truth) <= np.abs(noisy-truth)+1e-12)
    # All scalar boundary regimes are included, independently of random draws.
    for y in (-100., -1., 0., 10., 20., 21., 100.):
        for v in (0., 4., 20.):
            x = np.array([[3.,y]]*3)
            assert abs(project_velocity(x)[0,1]-v) <= abs(y-v)


@pytest.mark.parametrize('raw', [np.zeros((2,2)), [[0.,float('nan')]]*3,
                                 [[float('inf'),1.]]*3])
def test_invalid_observations_are_not_sanitized(raw):
    with pytest.raises(ValueError, match='finite three-vehicle'):
        project_velocity(raw)


def test_wrapper_routes_only_projected_observation_and_retains_metadata(monkeypatch):
    class FakeGate:
        def __init__(self,*args):
            self.c = SimpleNamespace(vmax=20.)
            self.calls = []
        def select(self, state, previous, unused):
            self.calls.append((state.copy(),previous.copy(),unused))
            return np.zeros(3), {'actual_learning_used':False}
    monkeypatch.setattr(projection,'RecoveryJointGate',FakeGate)
    gate = ProjectedObservationGate(None,'lag_strong','baseline')
    raw, previous = np.array([[-100.,20.199],[-130.,-1.],[-140.,8.]]), np.ones(3)
    command, info = gate.select(raw,previous)
    np.testing.assert_array_equal(gate.inner.calls[0][0],project_velocity(raw))
    np.testing.assert_array_equal(gate.inner.calls[0][1],previous)
    assert info['velocity_projection']['applied']
    assert info['velocity_projection']['restores_physical_certificate'] is False
    assert raw[0,1] == 20.199


def private_call(function, **bindings):
    namespace = dict(function.__globals__)
    namespace.update(bindings)
    return FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)


def non_timing(value):
    if isinstance(value,dict):
        return {k:non_timing(v) for k,v in value.items()
                if not k.endswith('_ms') and k not in ('velocity_projection','episode_wall_seconds')}
    if isinstance(value,list):
        return [non_timing(v) for v in value]
    return value


@pytest.fixture(scope='module')
def models():
    torch.set_num_threads(1)
    plan,_ = execute.binding_inputs()
    archive = ROOT/'artifacts/v44/ood_learned_development'
    manifest = execute.read(archive/'manifest.json')
    for name,expected in manifest['files_sha256'].items():
        assert execute.digest(archive/name) == expected
    return {name:load_coupled(ROOT/item['path'])[0] for name,item in plan['models'].items()}


@pytest.mark.skipif(os.environ.get('COOPGATE_RUN_SUMO_SMOKE') != '1',reason='Opt-in actual SUMO probe')
@pytest.mark.parametrize('index',range(3))
def test_paired_full_family_projection_probe(index,models,tmp_path):
    reference = ROOT/'artifacts/v44/ood_learned_development'/f'{index:05d}'
    plan = execute.read(reference/'development_plan.json')
    group = execute.read(reference/'development_group.json')
    def route(method,models):
        return execute.route_gate(method,models,gate_cls=ProjectedObservationGate)
    run = private_call(execute.run_case,route_gate=route)
    audit = private_call(auditor.audit_episode,RecoveryJointGate=ProjectedObservationGate)
    assert run.__code__ is execute.run_case.__code__
    assert audit.__code__ is auditor.audit_episode.__code__
    assert auditor.audit_episode.__globals__['RecoveryJointGate'] is not ProjectedObservationGate
    target = tmp_path/f'{index:05d}'
    with OODMismatchMergeEnv() as env:
        reviews = execute.collect_block(plan,group,target,models,env,'development-projection-v45',
            'same-frozen-v39-runtime',run_fn=run,audit_fn=audit)
    assert len(reviews) == 13
    projected = list(observations(target/'observations.sqlite',
        execute.metadata(group,'development-projection-v45','same-frozen-v39-runtime'),execute.schedule(group)))
    original = list(observations(reference/'observations.sqlite',
        execute.metadata(group,'development-only-v44','frozen-v39-runtime-development'),execute.schedule(group)))
    assert len(projected) == len(original) == 13
    for (key,body,trace),(old_key,old_body,old_trace) in zip(projected,original):
        assert key == old_key
        assert not body['row']['physical_certificate_applicable']
        assert np.isfinite(body['row']['failure_penalized_loss'])
        if index in (0,2):
            # Raw/true velocities in these two fixed negative controls stay in
            # range. Verify full non-timing identity rather than merely costs.
            assert non_timing(body) == non_timing(old_body)
            assert non_timing(trace) == non_timing(old_trace)
        for step in trace:
            raw = np.asarray(step['observed_state'])
            expected = project_velocity(raw)
            meta = step['velocity_projection']
            np.testing.assert_array_equal(meta['projected_velocity'],expected[:,1])
            truth = np.asarray(step['true_state_before'])[:,1]
            assert np.all(np.abs(expected[:,1]-truth) <= np.abs(raw[:,1]-truth)+1e-8)
    # No assertion selects, discards or requires favorable task outcomes in the
    # changed-input case. Its exact successes and failures must be reported.
