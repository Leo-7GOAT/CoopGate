import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import extend_temporal_repair_v20 as extension
from advance_temporal_repair_v20 import ready_to_finalize


def protocols():
    spec = dict(seed_base=63100001, n_initials=2, sampling='iid_mixture', modes=['clean', 'uniform'],
                mode_selection_stream=20207, bootstrap_stream=20213, bootstrap_samples=100)
    partner = dict(spec={**spec, 'methods': {'baseline': {'kind': 'baseline'}, **{
        f'sampled_{s}': dict(kind='temporal', seed=s, bank='sampled', bank_size=33) for s in (1802, 1812, 1822)}}},
        models={'fixed': 'identical'}, source_sha256={'simulator.py': 'fixed'})
    new = dict(spec={**spec, 'methods': extension.methods()}, models=partner['models'],
               source_sha256={'simulator.py': 'fixed', 'extension.py': 'new'})
    return partner, new


def rows(spec):
    return [dict(initial_index=i, disturbance=mode, controller=name)
            for i in range(spec['n_initials']) for mode in spec['modes'] for name in spec['methods']]


def test_shared_initials_and_mode_selection_are_preserved():
    partner, new = protocols()
    joined, spec = extension.join_rows(rows(partner['spec']), rows(new['spec']), partner, new)
    assert spec['n_initials'] == 2
    assert len(joined) == 2 * 2 * 7
    for key in ('seed_base', 'mode_selection_stream', 'sampling', 'modes'):
        wrong = copy.deepcopy(new)
        wrong['spec'][key] = 'different'
        with pytest.raises(ValueError, match='Paired population'):
            extension.validate_pairing(partner, wrong)


def test_shared_source_and_model_changes_are_rejected():
    partner, new = protocols()
    for key in ('models', 'source_sha256'):
        wrong = copy.deepcopy(new)
        wrong[key] = {}
        with pytest.raises(ValueError):
            extension.validate_pairing(partner, wrong)


def test_no_missing_model_duplicate_or_partial_collection_is_merged():
    partner, new = protocols()
    good = rows(new['spec'])
    for wrong in (good[:-1], good + [good[0]]):
        with pytest.raises(ValueError, match='Missing or duplicate'):
            extension.join_rows(rows(partner['spec']), wrong, partner, new)
    new['spec']['methods'].pop('repair_1802')
    with pytest.raises(ValueError, match='all three'):
        extension.validate_pairing(partner, new)


def test_native_handles_must_all_exit_successfully_before_finalization():
    assert not ready_to_finalize([0, None], 0)
    assert not ready_to_finalize([0, 0], None)
    assert not ready_to_finalize([], 0)
    assert ready_to_finalize([0, 0], 0)
    for codes, partner in (([0, 1], 0), ([0, 0], 1), ([None, None], 7)):
        with pytest.raises(RuntimeError, match='Native execution failed'):
            ready_to_finalize(codes, partner)


def test_individual_regression_is_retained_even_with_positive_team_gain():
    partner, new = protocols()
    joined, spec = extension.join_rows(rows(partner['spec']), rows(new['spec']), partner, new)
    for row in joined:
        used = row['controller'] != 'baseline' and (row['initial_index'] == 0 or row['controller'].startswith('repair'))
        row.update(learned_action_steps=int(used), completed=True, collision=False, aborted=False,
                   contract_violations=0, per_vehicle_cost=[9.8] * 3 if used else [10.] * 3,
                   per_vehicle_finish=[9.] * 3, team_cost=9.8 if used else 10.)
    target = next(r for r in joined if r['controller'] == 'repair_1802' and r['initial_index'] == 1 and r['disturbance'] == 'uniform')
    target['per_vehicle_cost'], target['team_cost'] = [9., 9., 10.1], 28.1 / 3
    report = extension.paired_results(joined, spec)['1802']
    assert report['added_harmful_mode_runs'] == 1
    assert report['all_mode_incremental_gain']['mean_seconds'] > 0
    target['learned_action_steps'] = 0
    original = next(r for r in joined if r['controller'] == 'sampled_1802' and r['initial_index'] == 1 and r['disturbance'] == 'uniform')
    original['learned_action_steps'] = 1
    with pytest.raises(ValueError, match='lost an episode adoption'):
        extension.paired_results(joined, spec)
