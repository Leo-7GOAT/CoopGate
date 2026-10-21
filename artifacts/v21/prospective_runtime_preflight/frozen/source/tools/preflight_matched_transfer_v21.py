"""Actual SUMO parity, all-plant integration and retained-failure preflight."""
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from coopgate.disturbance_v9 import DisturbedMergeEnv, disturbance_tape
from coopgate.joint_planner_v15 import JointRolloutGate
from coopgate.matched_shift_v21 import MatchedShiftedMergeEnv
from coopgate.shifted_sumo_v19 import PLANTS, SCENARIOS
from experiment_ego_v13 import run_episode
from probe_disturbed_gate_v19 import digest, save
from recover_checkpoint_v19 import atomic_bytes
from start_final_v19 import require_committed, verify_manifest
from study_matched_transfer_v21 import (PREFLIGHT, ROOT, factories, identity, independent_review,
                                       methods, record, source_records, verify_record)
from study_phase_repair_v21 import MODELS, REFERENCE_KEYS
from verify_development_v14 import trace_audit


OLD = ROOT / 'artifacts/v19/shifted_study'


class NominalNoiseReference(DisturbedMergeEnv):
    """The original additive simulator, independent of the actuator wrapper."""
    def reset_disturbed(self, seed, scenario, mode, radius):
        result = super().reset_disturbed(seed, scenario, mode, radius)
        self.tape = disturbance_tape(seed, mode, .1, round(self.c.horizon / self.c.dt))
        return result


def preflight_cases(old_spec, selected_failure):
    result = []
    primary_index = old_spec['scenarios'].index('dense')
    primary = (primary_index * old_spec['n_initials_per_scenario'], 'dense', 0,
               old_spec['seed_base'] + primary_index * 1000)
    for plant in PLANTS:
        result.extend((*primary, plant, name) for name in methods(True))
    for si, scenario in enumerate(SCENARIOS):
        if scenario == 'dense': continue
        for plant in PLANTS:
            result.append((si * old_spec['n_initials_per_scenario'], scenario, 0,
                           old_spec['seed_base'] + si * 1000, plant, 'baseline'))
    failure = identity(selected_failure)
    if failure not in result: result.append(failure)
    return result


def compare_trajectory(left, right, keys=('command', 'x', 'disturbance')):
    if len(left) != len(right): raise ValueError('Paired trajectory length differs')
    for actual, expected in zip(left, right):
        for key in keys: np.testing.assert_equal(actual[key], expected[key], err_msg='Trajectory differs: ' + key)


def run():
    torch.set_num_threads(1)
    if PREFLIGHT.exists(): raise FileExistsError('Preserve existing integration evidence')
    verify_manifest(OLD)
    old_protocol = json.loads((OLD / 'protocol.json').read_text())
    old_spec = old_protocol['spec']
    archived = json.loads((OLD / 'episodes.json').read_text())
    old_lookup = {(r['scenario'], r['initial_index'], r['plant_name'], r['controller']): r for r in archived}
    failure = next(r for r in archived if r['controller'] == 'baseline' and not r['completed'])
    selected_cases = preflight_cases(old_spec, failure)
    sources, models = source_records()
    require_committed([ROOT / name for name in sources])
    spec = dict(methods=methods(True), all_traces_saved=True, audit_initial_indices=[])
    protocol = dict(spec=spec, models=models)
    plan = dict(purpose='Integration/parity only; all initial states already used in V19',
        cases=selected_cases, additive_reference_scenarios=['nominal', 'dense'],
        expected_actual_sumo_runs=len(selected_cases) + 2, source_sha256=sources, models=models,
        old_manifest_sha256=digest(OLD / 'sha256.json'), methods=spec['methods'],
        selected_failure=list(identity(failure)), failure_selection='First baseline noncompletion in the completed sorted V19 archive',
        external_innovation_radius=.1, physical_error_radius=.5, new_study_initials_used=False,
        additional_independent_initials=0, risk_certified=False, phase_study_outcomes_read=False)
    PREFLIGHT.mkdir(parents=True)
    save(PREFLIGHT / 'plan.json', plan)
    for name in sources:
        target = PREFLIGHT / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    fs, rows, paths = factories(protocol), [], {}
    for plant in PLANTS:
        with MatchedShiftedMergeEnv(plant=plant) as env:
            for case in (entry for entry in selected_cases if entry[4] == plant):
                gi, scenario, index, seed, _, name = case
                row, trace, initial = run_episode(env, seed, scenario, 'uniform', fs[name], True)
                row = record(row, trace, initial, case, spec)
                target = PREFLIGHT / f'trace_{gi}_{plant}_{name}.json'
                save(target, trace)
                paths[identity(row)] = target
                rows.append(row)
                atomic_bytes(PREFLIGHT / 'episodes.json', (json.dumps(rows, indent=2) + '\n').encode('utf-8'))
                verify_record(trace, row, spec['methods'][name])
                print(json.dumps(dict(actual_sumo_runs=len(rows), expected_actual_sumo_runs=plan['expected_actual_sumo_runs'])), flush=True)
    lookup = {(r['scenario'], r['initial_index'], r['plant_name'], r['controller']): r for r in rows}
    reference_pairs, old_pairs = 0, 0
    parity_keys = ('initial', 'per_vehicle_finish', 'per_vehicle_variation', 'completed', 'collision',
                   'contract_violations', 'aborted', 'abort_reason', 'max_actual_error',
                   'error_contract_exceeded_substeps', 'learned_action_steps', 'plant')
    for row in rows:
        key = (row['scenario'], row['initial_index'], row['plant_name'], row['controller'])
        if row['plant_name'] == 'nominal' or key not in old_lookup: continue
        old = old_lookup[key]
        for field in parity_keys: np.testing.assert_equal(row[field], old[field], err_msg='Old plant parity: ' + field)
        old_path = OLD / f"shard_{old['global_initial_index'] % old_spec['shards']}" / (
            f"trace_{row['scenario']}_{row['initial_index']}_{row['plant_name']}_{row['controller']}.json")
        compare_trajectory(json.loads(paths[identity(row)].read_text()), json.loads(old_path.read_text()),
                           ('command', 'x', 'disturbance', 'innovation', 'actuator_states'))
        old_pairs += 1
    for plant in PLANTS:
        for seed in MODELS:
            original, reference = lookup['dense', 0, plant, f'repair_{seed}'], lookup['dense', 0, plant, f'reference_{seed}']
            for key in REFERENCE_KEYS: np.testing.assert_equal(original[key], reference[key])
            compare_trajectory(json.loads(paths[identity(original)].read_text()), json.loads(paths[identity(reference)].read_text()),
                ('command', 'x', 'disturbance', 'innovation', 'actuator_states', 'composition', 'repair_weight', 'selected_bank_differences'))
            reference_pairs += 1
    common_inputs = {}
    for row in rows:
        key = row['scenario'], row['initial_index']
        if key in common_inputs:
            previous = common_inputs[key]
            np.testing.assert_array_equal(row['initial'], previous['initial'])
            if row['innovation_tape_sha256'] != previous['innovation_tape_sha256']:
                raise ValueError('Cross-plant noise tapes differ')
        common_inputs[key] = row
    reference_rows = []
    with NominalNoiseReference() as env:
        for scenario in ('nominal', 'dense'):
            matched = lookup[scenario, 0, 'nominal', 'baseline']
            row, trace, initial = run_episode(env, matched['seed'], scenario, 'uniform', lambda x: JointRolloutGate(), True)
            row.update(initial=initial.tolist(), controller='additive_reference')
            save(PREFLIGHT / f'reference_additive_{scenario}.json', trace)
            reference_rows.append(row)
            save(PREFLIGHT / 'additive_reference_episodes.json', reference_rows)
            trace_audit(trace, row)
            for key in ('initial', 'per_vehicle_finish', 'per_vehicle_variation', 'completed', 'collision', 'contract_violations', 'aborted', 'abort_reason'):
                np.testing.assert_equal(row[key], matched[key], err_msg='Independent additive reference: ' + key)
            matched_trace = json.loads(paths[identity(matched)].read_text())
            compare_trajectory(trace, matched_trace, ('command', 'x'))
            # The actuator wrapper records (u+w)-u; the additive simulator
            # records w directly. Subtraction roundoff is not a noise change.
            np.testing.assert_allclose([t['disturbance'] for t in trace],
                                       [t['disturbance'] for t in matched_trace], atol=1e-12, rtol=0)
            print(json.dumps(dict(actual_sumo_runs=len(rows) + len(reference_rows), expected_actual_sumo_runs=plan['expected_actual_sumo_runs'])), flush=True)
    review = independent_review(rows, protocol, lambda row: json.loads(paths[identity(row)].read_text()))
    reproduced = next(r for r in rows if identity(r) == identity(failure))
    if reproduced['completed'] or not reproduced['aborted'] or reproduced['abort_reason'] != failure['abort_reason']:
        raise ValueError('The archived baseline failure was not retained')
    if len(rows) + len(reference_rows) != plan['expected_actual_sumo_runs']:
        raise ValueError('Incomplete actual integration grid')
    save(PREFLIGHT / 'audit.json', dict(all_checks_pass=True, actual_sumo_runs=len(rows) + len(reference_rows),
        expected_actual_sumo_runs=plan['expected_actual_sumo_runs'], matched_runs=len(rows),
        old_exact_trajectory_pairs=old_pairs, original_reference_pairs=reference_pairs,
        independent_additive_pairs=2, independently_reviewed=review, baseline_failure_reproduced=True,
        completed_runs=sum(r['completed'] for r in rows + reference_rows),
        aborted_runs=sum(r['aborted'] for r in rows + reference_rows),
        additional_independent_initials=0, new_study_initials_used=False, risk_certified=False,
        plan_sha256=digest(PREFLIGHT / 'plan.json')))
    save(PREFLIGHT / 'sha256.json', {p.relative_to(PREFLIGHT).as_posix(): digest(p) for p in PREFLIGHT.rglob('*') if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps(dict(preflight_passed=True, actual_sumo_runs=len(rows) + len(reference_rows))), flush=True)


if __name__ == '__main__': run()
