"""Real-policy OOD engineering coverage, separate from the frozen efficacy grid.

Three deterministic development initials cross one new regime/plant/perception
combination each with the entire thirteen-method family. No winner selection,
parameter tuning or formal-risk inference is performed on these observations.
"""
from copy import deepcopy
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import execute_ood_validation_v39 as execute
import audit_ood_validation_v39 as auditor
import design_ood_validation_v39 as design
from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.episode_store_v21 import inventory, observations
from coopgate.ood_mismatch_v39 import OODMismatchMergeEnv

pytestmark = pytest.mark.skipif(os.environ.get('COOPGATE_RUN_SUMO_SMOKE') != '1',
                              reason='Opt-in real-policy SUMO engineering checks')


@pytest.fixture(scope='module')
def frozen_models():
    torch.set_num_threads(1)
    plan, _ = execute.binding_inputs()
    return {name: load_coupled(ROOT/value['path'])[0] for name,value in plan['models'].items()}


@pytest.mark.parametrize('index', range(3))
def test_full_learned_family_survives_persistence_and_independent_ood_replay(index, frozen_models, tmp_path):
    seed = 139_886_001+index
    regime, plant, perception = design.REGIMES[index], design.PLANTS[index], design.PERCEPTIONS[index]
    group = dict(block_index=index, initial_id=f'development-ood-learned:{seed}',
        seed=seed, regime=regime, plant=plant, perception=perception,
        plant_realization_key='development-plant', perception_realization_key='development-perception',
        innovation_stream_seed=seed+100_000_000, method_order=design.ordered_methods(index))
    plan = dict(initials=[dict(initial_id=group['initial_id'], seed=seed, regime=regime,
                             state=design.initial_state(seed, regime))],
        plant_contract=dict(realizations={'development-plant':design.plant_realization(seed, plant)}),
        perception_contract=dict(realizations={
            'development-perception':design.perception_realization(seed, perception)}))
    target = tmp_path/f'{index:05d}'
    audited = []
    def review_after_commit(plan, group, method, record, trace, model):
        # Retain even a genuine policy abort before trying to replay its cause.
        assert len(inventory(target/'observations.sqlite')['cases']) == len(audited)+1
        review = auditor.audit_episode(plan, group, method, record, trace, model)
        audited.append((method['rule'],method['model']))
        return review
    with OODMismatchMergeEnv() as env:
        reviews = execute.collect_block(plan, group, target, frozen_models, env,
            'development-only-v44', 'frozen-v39-runtime-development', audit_fn=review_after_commit)
    assert len(reviews) == len(audited) == 13
    assert set(audited) == {(m['rule'],m['model']) for m in design.method_keys()}
    stored = list(observations(target/'observations.sqlite',
        execute.metadata(group, 'development-only-v44', 'frozen-v39-runtime-development'),
        execute.schedule(group)))
    assert len(stored) == 13
    for key, body, trace in stored:
        row = body['row']
        assert row['physical_certificate_applicable'] is False
        assert np.isfinite(row['failure_penalized_loss'])
        assert row['failure_penalized_loss'] == pytest.approx(row['team_cost']+25.*row['failure'])
        assert len(body['calls']) == len(trace)+int(row['aborted'])
        if row['aborted']:
            assert row['terminal_attempt']['error'] is not None
        for step in trace:
            if not step['actual_learning_used']:
                np.testing.assert_array_equal(step['command'], step['baseline_command'])

    # A learned-rule trace must also reject a corrupted observed input rather
    # than silently replaying it using the simulator's true state.
    chosen = next((record for record in stored if record[0][-2] != 'baseline' and record[2]), None)
    if chosen is not None:
        key, body, trace = chosen
        corrupted = deepcopy(trace)
        corrupted[0]['observed_state'][0][0] += .001
        corrupted[0]['perception']['observed_sha256'] = auditor.array_digest(corrupted[0]['observed_state'])
        method = dict(rule=key[-2],model=key[-1])
        with pytest.raises(AssertionError):
            auditor.audit_episode(plan, group, method, body, corrupted, frozen_models[key[-1]])
    else:
        # Complete immediate failures are valid engineering outcomes. Verify a
        # corrupted terminal observation is rejected in that branch as well.
        key, body, trace = next(r for r in stored if r[0][-2] != 'baseline')
        corrupted = deepcopy(body)
        terminal = corrupted['row']['terminal_attempt']
        terminal['observed_state'][0][0] += .001
        terminal['perception']['observed_sha256'] = auditor.array_digest(terminal['observed_state'])
        with pytest.raises(AssertionError):
            auditor.audit_episode(plan, group, dict(rule=key[-2],model=key[-1]),
                                 corrupted, trace, frozen_models[key[-1]])
