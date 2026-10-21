import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
specification = importlib.util.spec_from_file_location('study_temporal_v20', TOOLS / 'study_temporal_v20.py')
study = importlib.util.module_from_spec(specification)
specification.loader.exec_module(study)


def example():
    spec = dict(n_initials=4, seed_base=63100001, sampling='iid_mixture',
                modes=['clean', 'uniform', 'bias', 'pulse'], mode_selection_stream=20207,
                bootstrap_stream=20213, bootstrap_samples=100,
                methods={'baseline': {'kind': 'baseline'}, 'combined_1802': {'kind': 'temporal', 'bank_size': 49}})
    rows = []
    for i in range(4):
        for mode in spec['modes']:
            for name in spec['methods']:
                used = name != 'baseline' and i < 2
                costs = np.full(3, 10. - (.2 if used else 0.))
                if used and i == 1 and mode == 'uniform':
                    costs = np.array([9., 9., 10.1])
                rows.append(dict(initial_index=i, seed=spec['seed_base'] + i, disturbance=mode, controller=name,
                                 scenario=study.regime_for(spec, i), initial=[[-80., 12.], [-85., 13.], [-120., 11.]],
                                 per_vehicle_cost=costs.tolist(), per_vehicle_finish=[9., 9., 9.], team_cost=float(costs.mean()),
                                 learned_action_steps=int(used), min_adopted_bank_checks=49 if used else None,
                                 max_adopted_failed_scenarios=0 if used else None,
                                 completed=True, collision=False, aborted=False, contract_violations=0,
                                 controller_ms=[1., 600.]))
    return rows, spec


def test_all_models_and_banks_kept_and_no_best_training_seed_selection():
    methods = study.methods()
    assert len(methods) == 13
    for seed in (1802, 1812, 1822):
        assert {methods[f'{name}_{seed}']['bank_size'] for name in ('portfolio0', 'alternating', 'sampled', 'combined')} == {17, 33, 49}


def test_paired_report_counts_individual_harm_even_when_team_improves():
    rows, spec = example()
    result = study.summarize(rows, spec)['combined_1802']
    assert result['harmful_runs'] == result['bundle_harmful_adopted'] == 1
    assert result['bundle_adopted'] == 2
    assert result['runs'] == 16 and result['actual_adopted'] == 2
    assert result['all_mode_mean_gain_percent'] > 0
    assert result['deadline_misses_500ms'] == 16


def test_missing_duplicate_and_mismatched_pairs_are_rejected():
    rows, spec = example()
    for wrong in (rows[:-1], rows + [rows[0]]):
        with pytest.raises(ValueError, match='Missing or duplicate'):
            study.summarize(wrong, spec)
    wrong = copy.deepcopy(rows)
    wrong[1]['initial'][0][0] += 1
    with pytest.raises(AssertionError):
        study.summarize(wrong, spec)


def test_zero_adoption_cannot_hide_a_different_outcome_or_failure():
    rows, spec = example()
    wrong = copy.deepcopy(rows)
    wrong[-1]['per_vehicle_cost'][0] += .1
    with pytest.raises(AssertionError):
        study.summarize(wrong, spec)
    wrong = copy.deepcopy(rows)
    wrong[-1]['aborted'] = True
    with pytest.raises(ValueError, match='Zero-adoption failure'):
        study.summarize(wrong, spec)


def test_fewer_checks_or_any_failed_path_cannot_count_as_strict_admission():
    rows, spec = example()
    for key, value in (('min_adopted_bank_checks', 33), ('max_adopted_failed_scenarios', 1)):
        wrong = copy.deepcopy(rows)
        wrong[1][key] = value
        with pytest.raises(ValueError, match='Missing full-bank'):
            study.summarize(wrong, spec)


def test_seed_disjointness_uses_protocol_metadata_and_detects_overlap(tmp_path, monkeypatch):
    root = tmp_path / 'artifacts' / 'old'
    root.mkdir(parents=True)
    (root / 'protocol.json').write_text(json.dumps({'spec': {'seed_base': 100, 'n_initials': 4}}))
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    monkeypatch.setattr(study, 'STUDY', tmp_path / 'new')
    with pytest.raises(ValueError, match='overlaps'):
        study.independent_ranges(102, 4)
    assert len(study.independent_ranges(104, 4)) == 1
