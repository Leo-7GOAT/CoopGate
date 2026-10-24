"""Opt-in real SUMO engineering checks; never part of efficacy data.

Run with COOPGATE_RUN_SUMO_SMOKE=1. These use a separate development seed,
handwritten commands, and four decisions, without any trained actor or plan.
"""
import os
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import design_ood_validation_v39 as design
import execute_ood_validation_v39 as execute
import audit_ood_validation_v39 as auditor
from coopgate.ood_mismatch_v39 import OODMismatchMergeEnv


pytestmark = pytest.mark.skipif(os.environ.get('COOPGATE_RUN_SUMO_SMOKE') != '1',
                              reason='Explicit real SUMO engineering smoke only')


class ScriptedGate:
    def __init__(self):
        self.observations = []

    def select(self, observed, previous):
        self.observations.append(np.asarray(observed).copy())
        return np.array([1., 0., -1.]), dict(in_set=False, certified=False, intervention=0.)


@pytest.mark.parametrize('profile', design.PERCEPTIONS)
def test_real_sumo_compound_dynamics_and_observation_firewall(profile):
    seed = 139_880_001  # disjoint from frozen study seeds; not a policy trial
    initial = np.array([[-100., 8.], [-110., 8.], [-140., 8.]])
    spec = design.plant_realization(seed, 'delay_gain_lag_heterogeneous')
    perception = design.perception_realization(seed, profile)
    env = OODMismatchMergeEnv()
    gate = ScriptedGate()
    try:
        env.reset_v39(seed=seed, regime='engineering_smoke', initial=initial.tolist(),
            plant=spec, perception=perception,
            perception_tape=design._perception_tape(seed, profile),
            innovation_stream_seed=seed + 100_000_000)
        env.gate = gate
        np.testing.assert_allclose(env.x, initial, atol=1e-8, rtol=0)
        memory = np.zeros(3)
        history = []
        true_history = [initial.copy()]
        for decision in range(4):
            _, _, terminal, truncated, row = env.step(None)
            assert not row['aborted'] and not terminal and not truncated
            entry = env.trace[-1]
            np.testing.assert_array_equal(gate.observations[-1], entry['observed_state'])
            history.append(np.array([1., 0., -1.]))
            delayed = np.array([history[decision - spec['command_delay_decisions'][i]][i]
                if decision >= spec['command_delay_decisions'][i] else 0. for i in range(3)])
            np.testing.assert_array_equal(entry['delayed_command'], delayed)
            expected = true_history[-1].copy()
            retention = np.exp(-env.c.dt / np.array(spec['lag_tau_seconds']))
            for innovation in entry['innovation']:
                memory = retention * memory + (1 - retention) * delayed
                accel = np.array(spec['gain']) * memory - np.array(spec['quadratic_drag']) * expected[:, 1]**2 + innovation
                expected[:, 1] = np.clip(expected[:, 1] + env.c.dt * accel, 0, env.c.vmax)
                expected[:, 0] += env.c.dt * expected[:, 1]
            np.testing.assert_allclose(env.x, expected, atol=1e-8, rtol=0)
            np.testing.assert_allclose(env.actuator_memory, memory, atol=1e-12, rtol=0)
            if profile == 'clean_reference':
                np.testing.assert_array_equal(gate.observations[-1], entry['true_state_before'])
            elif profile == 'bounded_bias_jitter':
                assert not np.array_equal(gate.observations[-1], entry['true_state_before'])
            elif decision > 0:
                for actor, source in enumerate(entry['perception']['source_decision']):
                    np.testing.assert_array_equal(gate.observations[-1][actor], true_history[source][actor])
                    assert source < decision
            assert row['physical_certificate_applicable'] is False
            true_history.append(env.x.copy())
        assert len(env.trace) == len(gate.observations) == 4
    finally:
        env.close()


@pytest.mark.parametrize('profile', design.PERCEPTIONS)
def test_real_sumo_baseline_episode_passes_independent_replay(profile):
    seed = 139_880_002
    plant_name = design.PLANTS[0]
    regime = design.REGIMES[0]
    group = dict(initial_id=f'development:{seed}', seed=seed, regime=regime,
        plant=plant_name, perception=profile, plant_realization_key='development-plant',
        perception_realization_key='development-perception', innovation_stream_seed=seed+100_000_000)
    plan = dict(initials=[dict(initial_id=group['initial_id'], seed=seed, regime=regime,
                             state=design.initial_state(seed, regime))],
        plant_contract=dict(realizations={'development-plant': design.plant_realization(seed, plant_name)}),
        perception_contract=dict(realizations={'development-perception': design.perception_realization(seed, profile)}))
    method = dict(rule='baseline', model='none')
    with OODMismatchMergeEnv() as env:
        record, trace = execute.run_case(plan, group, method, {}, env)
    review = auditor.audit_episode(plan, group, method, record, trace, None)
    assert review['gate_decisions'] > 0
    assert review['physical_substeps'] == 5 * len(trace)
    assert review['compound_dynamics_replayed'] and review['perception_replayed']
    assert review['fallback_identity_exact']
    assert record['row']['physical_certificate_applicable'] is False
    # Pass/fail here is replay correctness, never the baseline's performance.
    corrupted = copy.deepcopy(trace)
    corrupted[0]['perception']['observed_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='Perception evidence differs'):
        auditor.audit_episode(plan, group, method, record, corrupted, None)
    corrupted = copy.deepcopy(trace)
    corrupted[0]['observed_state'][0][0] += .001
    corrupted[0]['perception']['observed_sha256'] = auditor.array_digest(corrupted[0]['observed_state'])
    with pytest.raises(AssertionError):
        auditor.audit_episode(plan, group, method, record, corrupted, None)
