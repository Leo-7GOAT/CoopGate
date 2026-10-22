"""Frozen coverage and actual SUMO execution of complete-profile deployment.

Coverage is an audit, not another independent efficacy/risk sample. Realized
outcomes choose which cases to audit but are never supplied to the selector.
The selector sees calibration and forecasts constructed from observed state.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from coopgate.disturbance_v9 import DisturbedMergeEnv
from coopgate.episode_selection_v19 import physical_failure, select_deployment_profile
from coopgate.observable_initial_v19 import ObservableInitialMergeEnv
from compare_established_v19 import complete_study, read_plan
from experiment_ego_v13 import run_episode
from probe_disturbed_gate_v19 import digest, save
from start_final_v19 import check_launch, require_committed, verify_manifest
from study_gate_v19 import factories, read_protocol, regime_for
from verify_development_v14 import trace_audit

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'artifacts/v19/comparison_plan'
CAL = ROOT / 'artifacts/v19/comparison_calibration'
FINAL = ROOT / 'artifacts/v19/final_confirmation'
COMPARISON = ROOT / 'artifacts/v19/final_comparison'
REPLAY_PLAN = ROOT / 'artifacts/v19/comparison_replay_plan'


def freeze():
    if any(FINAL.glob('shard_*')):
        raise ValueError('Replay coverage must be frozen before final-test execution')
    plan = read_plan(PLAN)
    _, final = read_protocol(FINAL)
    sources = list((ROOT / 'src/coopgate').glob('*.py')) + [Path(__file__),
               ROOT / 'tools/experiment_ego_v13.py', ROOT / 'tools/verify_development_v14.py',
               ROOT / 'tools/study_gate_v19.py', ROOT / 'tools/compare_established_v19.py',
               ROOT / 'tools/probe_disturbed_gate_v19.py', ROOT / 'tools/start_final_v19.py']
    record = dict(
        purpose='Actual whole-profile deployment replay, not additional independent performance samples',
        comparison_plan_sha256=digest(PLAN / 'plan.json'), final_protocol_sha256=digest(FINAL / 'protocol.json'),
        methods=plan['final_test']['reported_methods'], training_seeds=plan['seeds'], modes=final['spec']['modes'],
        coverage=['First initial for every selected adapter/training-seed/profile in each mode',
                  'First actual learning admission for every adapter/training-seed/mode',
                  'First harmful learning admission for every adapter/training-seed/mode',
                  'All selected physical failures or noncompletions, including zero-adoption failures'],
        ordering='Initial index, adapter order, training seed, mode order; duplicate identical audit entries are merged',
        runtime='Select in the factory after actual SUMO reset and before the first actual transition; actual seed/mode/outcomes are unavailable to the selection interface',
        forecasts='All ten clean profiles from observed initial only; cache by exact observed initial within this replay audit; record full preparation cost and every reuse',
        expected_parity='Same selected profile and episode metrics; commands/states also compared wherever original component traces were saved; every new trace physically reconstructed',
        source_sha256={p.resolve().relative_to(ROOT).as_posix(): digest(p) for p in sources})
    REPLAY_PLAN.mkdir(parents=True, exist_ok=False)
    save(REPLAY_PLAN / 'plan.json', record)
    for source in record['source_sha256']:
        target = REPLAY_PLAN / 'source' / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source, target)
    save(REPLAY_PLAN / 'sha256.json', {p.relative_to(REPLAY_PLAN).as_posix(): digest(p)
         for p in REPLAY_PLAN.rglob('*') if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps(dict(plan_sha256=digest(REPLAY_PLAN / 'plan.json'), final_test_inspected=False)), flush=True)


def coverage_cases(selections, lookup, modes, adapters, seeds):
    """Include failures even if no learned action was ever executed."""
    chosen = {}
    seen_profile, seen_adoption, seen_harm = set(), set(), set()
    ordered = sorted(selections, key=lambda r: (r['initial_index'], adapters.index(r['adapter']), seeds.index(r['training_seed'])))
    for selection in ordered:
        i, adapter, seed, profile = (selection[key] for key in ('initial_index', 'adapter', 'training_seed', 'profile'))
        for mode in modes:
            row, baseline = lookup[i, mode, profile], lookup[i, mode, 'baseline']
            adopted = row['learned_action_steps'] > 0
            delta = np.column_stack((row['per_vehicle_cost'], row['per_vehicle_finish'])) - np.column_stack(
                (baseline['per_vehicle_cost'], baseline['per_vehicle_finish']))
            failure = physical_failure(row)
            harmful = failure or bool(np.any(delta > 1e-9))
            group, profile_group = (adapter, seed, mode), (adapter, seed, mode, profile)
            reasons = []
            if profile_group not in seen_profile:
                reasons.append('first_selected_profile')
                seen_profile.add(profile_group)
            if adopted and group not in seen_adoption:
                reasons.append('first_learning_admission')
                seen_adoption.add(group)
            if adopted and harmful and group not in seen_harm:
                reasons.append('first_harmful_learning_admission')
                seen_harm.add(group)
            if failure:
                reasons.append('physical_failure_or_noncompletion')
            if reasons:
                key = (i, adapter, seed, mode)
                if key in chosen:
                    raise ValueError('Duplicate complete-profile selection')
                chosen[key] = dict(**selection, mode=mode, reasons=reasons)
    return list(chosen.values())


class ObservedForecasts:
    def __init__(self, fs, output):
        self.fs, self.output = fs, Path(output)
        self.cache, self.preparations = {}, []
        self.actual_sumo_runs = 0

    def get(self, initial):
        # This key includes exactly the observable kinematics, not an episode ID
        # or any actual disturbance data. All other predictor inputs are frozen.
        state = np.asarray(initial, dtype=float)
        key = tuple(state.ravel())
        if key in self.cache:
            record = self.cache[key]
            record['uses'] += 1
            return record, True
        index, rows = len(self.preparations), {}
        started = time.perf_counter()
        with ObservableInitialMergeEnv(state) as shadow:
            for name, factory in self.fs.items():
                row, trace, observed = run_episode(shadow, 0, 'ignored', 'clean', factory, True)
                np.testing.assert_allclose(observed, state, atol=1e-8, rtol=0)
                row['initial'] = observed.tolist()
                trace_audit(trace, row)
                rows[name] = row
                save(self.output / f'shadow_{index}_{name}.json', trace)
                self.actual_sumo_runs += 1
        record = dict(index=index, initial=state.tolist(), rows=rows, uses=1,
                      full_preparation_wall_seconds=time.perf_counter() - started)
        self.cache[key] = record
        self.preparations.append(record)
        save(self.output / 'preparations.json', self.preparations)
        return record, False


def deployment_factory(adapter, seed, plan, calibration, fs, forecasts, decisions):
    def factory(initial):
        started = time.perf_counter()
        preparation, reused = (forecasts.get(initial) if adapter == 'conformal' else (None, False))
        choice = select_deployment_profile(adapter, seed, plan, calibration,
                                           preparation['rows'] if preparation else None)
        decisions.append(dict(adapter=adapter, training_seed=seed, profile=choice,
                              observed_initial=np.asarray(initial).tolist(),
                              shadow_preparation_index=preparation['index'] if preparation else None,
                              reused_shadow_forecast=reused,
                              selection_wall_seconds=time.perf_counter() - started))
        return fs[choice](initial)
    return factory


def assert_outcomes(row, reference):
    for field in ('per_vehicle_finish', 'per_vehicle_variation', 'completed', 'collision', 'aborted', 'contract_violations'):
        np.testing.assert_array_equal(row[field], reference[field], err_msg=field)


def run(output):
    torch.set_num_threads(1)
    check_launch()
    verify_manifest(REPLAY_PLAN)
    require_committed([REPLAY_PLAN / 'plan.json', REPLAY_PLAN / 'sha256.json'])
    replay_plan = json.loads((REPLAY_PLAN / 'plan.json').read_text())
    for path, expected in replay_plan['source_sha256'].items():
        assert digest(REPLAY_PLAN / 'source' / path) == digest(ROOT / path) == expected, path
    assert replay_plan['comparison_plan_sha256'] == digest(PLAN / 'plan.json')
    assert replay_plan['final_protocol_sha256'] == digest(FINAL / 'protocol.json')
    plan = read_plan(PLAN)
    verify_manifest(CAL)
    verify_manifest(COMPARISON)
    calibration = json.loads((CAL / 'calibration.json').read_text())
    protocol, lookup, names, _, _ = complete_study(FINAL, 'final_confirmation')
    comparison = json.loads((COMPARISON / 'comparison.json').read_text())
    assert comparison['calibration_sha256'] == digest(CAL / 'calibration.json')
    assert comparison['final_episodes_sha256'] == digest(FINAL / 'episodes.json')
    assert comparison['plan_sha256'] == digest(PLAN / 'plan.json')
    selections = json.loads((COMPARISON / 'selections.json').read_text())
    expected = {(i, adapter, seed) for i in range(protocol['spec']['n_initials'])
                for adapter in replay_plan['methods'] for seed in plan['seeds']}
    assert {(r['initial_index'], r['adapter'], r['training_seed']) for r in selections} == expected
    assert len(selections) == len(expected)
    for selection in selections:
        assert selection['seed'] == protocol['spec']['seed_base'] + selection['initial_index']
        assert selection['profile'] in ('baseline', *names)
    cases = coverage_cases(selections, lookup, replay_plan['modes'], replay_plan['methods'], plan['seeds'])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'cases.json', cases)
    fs, rows, decisions, parity = factories(protocol), [], [], []
    warm = np.array([[-80., 12.], [-85., 13.], [-120., 11.]])
    for factory in fs.values():
        factory(warm).select(warm, np.zeros(3))
    forecasts = ObservedForecasts(fs, output)
    with DisturbedMergeEnv() as env:
        for case_index, case in enumerate(cases):
            factory = deployment_factory(case['adapter'], case['training_seed'], plan, calibration, fs, forecasts, decisions)
            i = case['initial_index']
            row, trace, observed = run_episode(env, case['seed'], regime_for(protocol['spec'], i), case['mode'], factory, True)
            choice = decisions[-1]['profile']
            assert choice == case['profile'], 'Runtime selection differs from frozen paired comparison'
            reference = lookup[i, case['mode'], choice]
            np.testing.assert_array_equal(observed, reference['initial'])
            assert_outcomes(row, reference)
            row['initial'] = observed.tolist()
            trace_audit(trace, row)
            method = protocol['spec']['methods'][choice]
            used = sum(t.get('nominal_joint_action_admitted', method['kind'] == 'raw') for t in trace)
            assert used == reference['learned_action_steps']
            assert len(trace) == reference['control_steps']
            assert [int(t.get('composition', 0)) for t in trace] == reference['chosen_compositions']
            # Outcome/trace references are inspected only after the actual
            # episode. They cannot influence selection or execution above.
            original_trace = FINAL / f"shard_{i % protocol['spec']['shards']}" / f"trace_{i}_{case['mode']}_{choice}.json"
            trace_compared = original_trace.exists()
            if trace_compared:
                archived = json.loads(original_trace.read_text())
                assert len(archived) == len(trace)
                for actual, expected_step in zip(trace, archived):
                    np.testing.assert_array_equal(actual['command'], expected_step['command'])
                    np.testing.assert_allclose(actual['x'], expected_step['x'], atol=1e-8, rtol=0)
            preparation_index = decisions[-1]['shadow_preparation_index']
            if preparation_index is not None:
                preparation = forecasts.preparations[preparation_index]
                for name, shadow in preparation['rows'].items():
                    assert_outcomes(shadow, lookup[i, 'clean', name])
            row.update(case_index=case_index, initial_index=i, adapter=case['adapter'], training_seed=case['training_seed'],
                       selected_profile=choice, learned_action_steps=used, initial=observed.tolist())
            rows.append(row)
            parity.append(dict(case_index=case_index, decision_steps=len(trace), archived_trace_compared=trace_compared))
            save(output / f'trace_{case_index}.json', trace)
            save(output / 'episodes.json', rows)
            save(output / 'decisions.json', decisions)
            save(output / 'preparations.json', forecasts.preparations)
            print(json.dumps(dict(complete_replays=len(rows), planned_replays=len(cases), shadow_sumo_runs=forecasts.actual_sumo_runs)), flush=True)
    report = dict(
        purpose='Frozen-selection and real whole-profile execution identity audit',
        actual_deployment_replays=len(rows), actual_shadow_sumo_runs=forecasts.actual_sumo_runs,
        coverage_cases=len(cases), all_selected_profiles_match=True, all_episode_metrics_match=True,
        new_trajectories_physically_reconstructed=len(rows), parity=parity,
        whole_profile_replay_requirement_met=True,
        count_scope='Coverage audit only; these replays are not additional independent risk/efficacy samples',
        runtime_inputs='Calibration plus observed-initial clean forecasts only; no actual mode or future outcomes',
        risk_certified_by_this_audit=False, hard_real_time_certified=False,
        preparation_cost_scope='Full ten-profile shadow preparation includes startup and resets; all exact-initial cache reuse is recorded. A fresh deployment must pay the full preparation cost.',
        replay_plan_sha256=digest(REPLAY_PLAN / 'plan.json'),
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in
                       (PLAN / 'plan.json', CAL / 'calibration.json', FINAL / 'protocol.json',
                        FINAL / 'episodes.json', COMPARISON / 'comparison.json', COMPARISON / 'selections.json')})
    save(output / 'audit.json', report)
    shutil.copy2(__file__, output / Path(__file__).name)
    save(output / 'sha256.json', {p.relative_to(output).as_posix(): digest(p)
         for p in output.rglob('*') if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps({key: value for key, value in report.items() if key != 'parity'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('freeze')
    p = sub.add_parser('run')
    p.add_argument('output')
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze()
    else:
        run(args.output)
