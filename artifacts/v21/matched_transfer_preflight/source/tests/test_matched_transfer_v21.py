import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import study_matched_transfer_v21 as study
from preflight_matched_transfer_v21 import preflight_cases
from coopgate.matched_shift_v21 import matched_plant_parameters
from coopgate.shifted_sumo_v19 import PLANTS, SCENARIOS


def fixture():
    spec = dict(seed_base=63400001, n_initials_per_scenario=2, shards=3, scenarios=list(SCENARIOS),
                plants=list(PLANTS), methods=study.methods(), bootstrap_stream=21213, bootstrap_samples=50)
    rows = []
    for shard in range(spec['shards']):
        for gi, scenario, i, seed, plant, name in study.schedule(spec, shard):
            cost = 11. if plant == 'gain_drag' else 10.
            rows.append(dict(global_initial_index=gi, scenario=scenario, initial_index=i, seed=seed,
                plant_name=plant, controller=name, initial=[[-80., 12.], [-85., 13.], [-120., 11.]],
                plant=matched_plant_parameters(seed, plant), innovation_tape_sha256=str(seed),
                external_innovation_radius=.1, per_vehicle_cost=[cost] * 3, per_vehicle_finish=[cost] * 3,
                completed=True, aborted=False, collision=False, contract_violations=0,
                team_cost=cost, learned_action_steps=0, trajectory_sha256=f'{seed}/{plant}',
                max_actual_error=.1, error_contract_exceeded_substeps=0, controller_ms=[1.],
                forecast_error_count=0, generation_error_count=0))
    return rows, spec


def test_complete_methods_include_all_models_old_controls_and_both_new_margins():
    entries = study.methods()
    assert len(entries) == 19 and len(study.methods(True)) == 22
    for seed in study.MODELS:
        for prefix in ('raw', 'nominal', 'portfolio0', 'repair', 'phase05', 'phase10'):
            assert entries[f'{prefix}_{seed}']['seed'] == seed
        assert entries[f'phase05_{seed}']['phase_tolerance_seconds'] == .05
        assert entries[f'phase10_{seed}']['phase_tolerance_seconds'] == .1


def test_grid_rejects_duplicate_missing_and_substituted_initials():
    rows, spec = fixture()
    assert len(study.validate_grid(rows, spec)) == 5 * 2 * 4 * 19
    bad_seed = copy.deepcopy(rows)
    bad_seed[0]['seed'] += 1
    for bad in (rows[:-1], rows + [rows[0]], bad_seed):
        with pytest.raises(ValueError, match='transfer case'): study.validate_grid(bad, spec)
    with pytest.raises(ValueError, match='overlap'):
        list(study.cases({**spec, 'n_initials_per_scenario': 1001}))


def test_paired_plant_contrast_uses_the_same_initials_and_retains_baseline_failure():
    rows, spec = fixture()
    for row in rows:
        if row['controller'] == 'phase05_1802' and row['plant_name'] in ('nominal', 'gain_drag'):
            improvement = .3 if row['plant_name'] == 'gain_drag' else .2
            row['team_cost'] -= improvement
            row['per_vehicle_cost'] = [row['team_cost']] * 3
            row['learned_action_steps'] = 1
        if row['controller'] in ('phase10_1802', 'repair_1802') and row['plant_name'] in ('nominal', 'gain_drag'):
            row['learned_action_steps'] = 1
        if row['scenario'] == 'dense' and row['initial_index'] == 1 and row['plant_name'] == 'lag_strong':
            row.update(completed=False, aborted=True, error_contract_exceeded_substeps=1, max_actual_error=1.)
    result = study.summarize(rows, spec)
    contrast = next(c for c in result['paired_plant_contrasts'] if c['scenario'] == 'dense' and
                    c['plant'] == 'gain_drag' and c['method'] == 'phase05_1802')
    assert contrast['mean_seconds'] == pytest.approx(.1)
    assert contrast['pointwise_bootstrap95'] == pytest.approx([.1, .1])
    assert contrast['independent_initials'] == 2
    cell = next(c for c in result['cells'] if c['scenario'] == 'dense' and c['plant'] == 'lag_strong' and c['method'] == 'phase10_1812')
    assert cell['adopted'] == 0 and cell['harmful'] == 1 and cell['aborted'] == 1
    assert cell['baseline_failed_runs'] == 1 and cell['mean_gain_seconds'] == 0


def test_mismatched_noise_and_zero_adoption_outcome_changes_are_rejected():
    rows, spec = fixture()
    row = next(r for r in rows if r['controller'] == 'phase05_1802')
    row['innovation_tape_sha256'] = 'changed'
    with pytest.raises(ValueError, match='Paired external'): study.summarize(rows, spec)
    row['innovation_tape_sha256'] = str(row['seed'])
    row['per_vehicle_cost'][0] -= .1
    with pytest.raises(AssertionError, match='Zero-adoption'): study.summarize(rows, spec)


def test_overlap_audit_includes_offset_scenario_blocks(tmp_path, monkeypatch):
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    archive = tmp_path / 'artifacts' / 'old'
    archive.mkdir(parents=True)
    (archive / 'protocol.json').write_text(json.dumps(dict(spec=dict(seed_base=63400001,
        n_initials_per_scenario=32, scenarios=list(SCENARIOS)))))
    spec = dict(seed_base=63403001, n_initials_per_scenario=2, scenarios=['nominal'])
    with pytest.raises(ValueError, match='overlap'): study.disjoint_protocols(spec)


def test_preflight_exercises_all_plants_reference_variants_and_a_known_failure():
    old = dict(seed_base=62100001, n_initials_per_scenario=32, scenarios=list(SCENARIOS))
    failure = dict(global_initial_index=40, scenario='dense', initial_index=8, seed=62101009,
                   plant_name='lag_strong', controller='baseline')
    entries = preflight_cases(old, failure)
    assert len(entries) == len(set(entries)) == 105
    assert study.identity(failure) in entries
    for plant in PLANTS:
        assert len([e for e in entries if e[1] == 'dense' and e[2] == 0 and e[4] == plant]) == 22
    assert {e[1] for e in entries} == set(SCENARIOS)
