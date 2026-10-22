"""Calibrate established adapters and evaluate untouched paired final episodes.

All adapters have access to the same frozen nine complete controller profiles.
LTT is also the proposed gate's validation tool, not a competing new theorem.
Conformal and mean-SPI adapters have different targets, explicitly preserved.
"""
import argparse
from collections import Counter
import json
import shutil
from pathlib import Path

import numpy as np

from coopgate.comparison_v19 import (
    conditional_counts, conformal_episode_bound, conformal_profile, select_fixed_replicates,
)
from coopgate.joint_admission_v15 import JointBound
from certify_gate_family_v19 import certify
from probe_disturbed_gate_v19 import digest, save

ROOT = Path(__file__).resolve().parents[1]


def complete_study(directory, role):
    directory = Path(directory)
    protocol = json.loads((directory / 'protocol.json').read_text())
    spec = protocol['spec']
    if spec.get('data_role') != role or spec['sampling'] != 'iid_mixture':
        raise ValueError('Wrong data role or non-IID initial sampling')
    # A missing manifest or missing shards is an incomplete study, never usable
    # calibration. The caller must not monitor partial outcome-dependent bounds.
    for name, expected in json.loads((directory / 'sha256.json').read_text()).items():
        assert digest(directory / name) == expected, name
    summary = json.loads((directory / 'summary.json').read_text())
    assert summary['episodes_sha256'] == digest(directory / 'episodes.json')
    assert summary['protocol_sha256'] == digest(directory / 'protocol.json')
    rows = json.loads((directory / 'episodes.json').read_text())
    lookup = {(r['initial_index'], r['disturbance'], r['controller']): r for r in rows}
    n, modes = spec['n_initials'], spec['modes']
    names = [name for name in spec['methods'] if name != 'baseline']
    expected = {(i, mode, name) for i in range(n) for mode in modes for name in spec['methods']}
    assert set(lookup) == expected and len(lookup) == len(rows) == spec['expected_actual_sumo_runs']
    values = np.empty((n, len(modes), len(names), 3, 2))
    failed = np.zeros(values.shape[:3], bool)
    for i in range(n):
        initial = lookup[i, modes[0], 'baseline']['initial']
        for mi, mode in enumerate(modes):
            base = lookup[i, mode, 'baseline']
            np.testing.assert_array_equal(base['initial'], initial)
            for ci, name in enumerate(names):
                row = lookup[i, mode, name]
                np.testing.assert_array_equal(row['initial'], initial)
                values[i, mi, ci] = np.column_stack((row['per_vehicle_cost'], row['per_vehicle_finish'])) - np.column_stack(
                    (base['per_vehicle_cost'], base['per_vehicle_finish']))
                failed[i, mi, ci] = bool(not row['completed'] or row['collision'] or row['aborted'] or row['contract_violations'])
    return protocol, lookup, names, values, failed


def freeze(study, output):
    study, output = Path(study), Path(output)
    protocol = json.loads((study / 'protocol.json').read_text())
    spec = protocol['spec']
    assert spec['data_role'] == 'risk_validation' and spec['sampling'] == 'iid_mixture'
    sources = [Path(__file__), ROOT / 'src/coopgate/comparison_v19.py',
               ROOT / 'src/coopgate/joint_admission_v15.py', ROOT / 'src/coopgate/risk_comparators_v19.py',
               ROOT / 'tools/certify_gate_family_v19.py', ROOT / 'tools/probe_disturbed_gate_v19.py']
    plan = dict(
        purpose='Established episode-profile adapters, frozen during calibration collection and before final-test evaluation',
        calibration_protocol_sha256=digest(study / 'protocol.json'),
        profiles=[name for name in spec['methods'] if name != 'baseline'],
        profile_definitions=spec['methods'], models=protocol['models'], seeds=sorted(protocol['models']),
        nominal_predictor='Complete clean SUMO closed loop for each fixed profile and the baseline, from the same observed initial; computed without observing the actual future disturbance. Clean component executions may be reused as these forecasts.',
        conformal=dict(alpha=.05, component_scales=[[1., 1.]] * 3, minimum_team_gain=1e-9,
                       score='Maximum residual over all four modes, nine profiles, vehicles and metrics; +inf for any physical failure or noncompletion in the bundle',
                       target='Marginal coverage across the calibration set and a new initial bundle, not adoption-conditioned risk'),
        ltt=dict(source='Frozen two-target Bonferroni certification of the same nine profiles',
                 selection='For each fixed training seed, choose the certified profile with highest calibration actual-mode mean gain; baseline if none'),
        mean_spi=dict(source='Same nine-profile paired Gaussian simultaneous mean bands',
                      selection='For each fixed training seed choose the positive-lower-bound profile with highest calibration actual-mode mean gain; baseline if none',
                      target='Asymptotic positive population mean improvement, not conditional harm control'),
        final_test=dict(seed_base=62200001, n_initials=512, sampling='iid_mixture', modes=spec['modes'],
                        mode_selection_stream=19417, bootstrap_seed=19423, bootstrap_samples=5000,
                        reported_methods=['raw', 'nominal', 'portfolio0', 'conformal', 'ltt', 'mean_spi'],
                        actual_adoption_minimum=.20, mean_gain_percent_minimum=.5),
        comparison_scope='All methods share the same frozen profile family and paired data. This compares deployment rules and guarantees; it is not a claim that the proposed online gate outperforms LTT itself. No cross-training-seed selection.',
        evaluation='Select an entire independently executed component trajectory. No trajectory splice or post-outcome switching. Record selected profiles for separate actual SUMO replay verification.',
        timing='Report selected-profile step timings and the sum of all clean shadow controller/setup times for conformal initialization. The latter excludes simulator IPC/startup and is only a lower bound on full preparation time.',
        source_sha256={p.relative_to(ROOT).as_posix(): digest(p) for p in sources})
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'plan.json', plan)
    for source in plan['source_sha256']:
        destination = output / 'source' / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source, destination)
    save(output / 'sha256.json', {p.relative_to(output).as_posix(): digest(p)
         for p in output.rglob('*') if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps(dict(plan_sha256=digest(output / 'plan.json'), profiles=plan['profiles'])), flush=True)


def read_plan(directory):
    directory = Path(directory)
    plan = json.loads((directory / 'plan.json').read_text())
    for path, expected in json.loads((directory / 'sha256.json').read_text()).items():
        assert digest(directory / path) == expected, path
    for path, expected in plan['source_sha256'].items():
        assert digest(ROOT / path) == expected, 'Frozen comparison source changed: ' + path
    return plan


def calibrate(plan_directory, study, output):
    plan = read_plan(plan_directory)
    study, output = Path(study), Path(output)
    assert digest(study / 'protocol.json') == plan['calibration_protocol_sha256']
    protocol, lookup, names, differences, failed = complete_study(study, 'risk_validation')
    assert names == plan['profiles'] and protocol['models'] == plan['models']
    spec = protocol['spec']
    predicted = differences[:, spec['modes'].index('clean')]
    bound, conformal = conformal_episode_bound(differences, predicted, failed, names, plan['conformal']['alpha'])
    output.mkdir(parents=True, exist_ok=False)
    certify(study, output / 'conditional_risk.json')
    risk = json.loads((output / 'conditional_risk.json').read_text())
    means = {r['candidate']: r['mean_gain'] for r in risk['mean_improvement_comparator']['candidates']}
    mean_eligible = [r['candidate'] for r in risk['mean_improvement_comparator']['candidates'] if r['mean_improvement_selected']]
    result = dict(
        plan_sha256=digest(Path(plan_directory) / 'plan.json'),
        calibration_protocol_sha256=digest(study / 'protocol.json'),
        calibration_episodes_sha256=digest(study / 'episodes.json'),
        calibration_seed_base=spec['seed_base'], calibration_n_initials=spec['n_initials'],
        names=names, models=protocol['models'], conformal=conformal,
        ltt_selection=select_fixed_replicates(names, risk['both_targets_certified'], means, plan['seeds']),
        mean_spi_selection=select_fixed_replicates(names, mean_eligible, means, plan['seeds']),
        both_targets_certified=risk['both_targets_certified'],
        conditional_risk_sha256=digest(output / 'conditional_risk.json'),
        final_test_inspected=False)
    save(output / 'calibration.json', result)
    save(output / 'sha256.json', {p.name: digest(p) for p in output.iterdir() if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps(result), flush=True)


def evaluate(plan_directory, calibration_directory, study, output):
    plan = read_plan(plan_directory)
    calibration_directory, study, output = Path(calibration_directory), Path(study), Path(output)
    for name, expected in json.loads((calibration_directory / 'sha256.json').read_text()).items():
        assert digest(calibration_directory / name) == expected, name
    cal = json.loads((calibration_directory / 'calibration.json').read_text())
    assert cal['plan_sha256'] == digest(Path(plan_directory) / 'plan.json')
    protocol, lookup, names, differences, failed = complete_study(study, 'final_confirmation')
    spec, final = protocol['spec'], plan['final_test']
    for field in ('seed_base', 'n_initials', 'sampling', 'modes'):
        assert spec[field] == final[field], field
    assert names == cal['names'] == plan['profiles'] and protocol['models'] == cal['models']
    assert spec['methods'] == plan['profile_definitions']
    train_ids = set(range(cal['calibration_seed_base'], cal['calibration_seed_base'] + cal['calibration_n_initials']))
    test_ids = list(range(spec['seed_base'], spec['seed_base'] + spec['n_initials']))
    assert not train_ids.intersection(test_ids)
    n, modes, seeds = spec['n_initials'], spec['modes'], plan['seeds']
    clean_index = modes.index('clean')
    q = np.inf if cal['conformal']['infinite_q'] else cal['conformal']['q']
    bound = JointBound(tuple(names), ((1., 1.),) * 3, q, cal['calibration_n_initials'], plan['conformal']['alpha'])
    actual_modes = np.array([np.random.default_rng(np.random.SeedSequence([seed, final['mode_selection_stream']])).integers(len(modes)) for seed in test_ids])
    bootstrap = np.random.default_rng(final['bootstrap_seed']).integers(n, size=(final['bootstrap_samples'], n))
    selections, reports = [], {}
    shadow_times = [sum(lookup[i, 'clean', name]['setup_ms'] + sum(lookup[i, 'clean', name]['controller_ms'])
                       for name in ('baseline', *names)) for i in range(n)]
    for method in final['reported_methods']:
        per_seed, gains = {}, []
        for seed in seeds:
            picked = []
            for i in range(n):
                if method == 'conformal':
                    choice = conformal_profile(bound, differences[i, clean_index], names, seed, failed[i, clean_index])
                    if not lookup[i, 'clean', 'baseline']['completed']:
                        choice = 'baseline'
                elif method in ('ltt', 'mean_spi'):
                    choice = cal[method + '_selection'][seed]
                else:
                    choice = f'{method}_{seed}'
                picked.append(choice)
                selections.append(dict(initial_index=i, seed=test_ids[i], training_seed=seed, adapter=method, profile=choice))
            a, h, gain, completed = [], [], [], []
            bundle_a, bundle_h, times = [], [], []
            for i, choice in enumerate(picked):
                cluster_a, cluster_h = [], []
                for mode in modes:
                    row, base = lookup[i, mode, choice], lookup[i, mode, 'baseline']
                    delta = np.column_stack((row['per_vehicle_cost'], row['per_vehicle_finish'])) - np.column_stack(
                        (base['per_vehicle_cost'], base['per_vehicle_finish']))
                    adopted = row['learned_action_steps'] > 0
                    harmful = bool(np.any(delta > 1e-9) or not row['completed'] or row['collision'] or row['aborted'] or row['contract_violations'])
                    cluster_a.append(adopted)
                    cluster_h.append(harmful)
                    if mode == modes[actual_modes[i]]:
                        a.append(adopted)
                        h.append(harmful)
                        gain.append(float(-delta[:, 0].mean()))
                        completed.append(row['completed'])
                        times.extend(row['controller_ms'])
                bundle_a.append(any(cluster_a))
                bundle_h.append(any(cluster_h))
            gain = np.array(gain)
            gains.append(gain)
            base_mean = np.mean([lookup[i, modes[actual_modes[i]], 'baseline']['team_cost'] for i in range(n)])
            per_seed[seed] = dict(
                actual_random_mode=conditional_counts(np.array(a), np.array(h)),
                worst_mode_bundle=conditional_counts(np.array(bundle_a), np.array(bundle_h)),
                completed=sum(completed), runs=n, selected_profiles=dict(Counter(picked)),
                mean_team_gain_seconds=float(gain.mean()), mean_team_gain_percent=float(100 * gain.mean() / base_mean),
                mean_team_gain_ci95=np.quantile(gain[bootstrap].mean(1), [.025, .975]).tolist(),
                controller_p95_ms=float(np.percentile(times, 95)) if times else None,
                controller_p99_ms=float(np.percentile(times, 99)) if times else None,
                controller_max_ms=float(max(times)) if times else None,
                steps_over_500ms=sum(t > 500 for t in times), total_control_steps=len(times))
            if method == 'conformal':
                per_seed[seed]['shadow_initialization_controller_time_lower_bound_ms'] = dict(
                    p50=float(np.median(shadow_times)), p95=float(np.percentile(shadow_times, 95)), maximum=float(max(shadow_times)))
        family_gain = np.array(gains).mean(0)
        reports[method] = dict(per_training_seed=per_seed, independent_initials=n,
                              mean_team_gain_seconds=float(family_gain.mean()),
                              mean_team_gain_ci95=np.quantile(family_gain[bootstrap].mean(1), [.025, .975]).tolist())
    report = dict(purpose='Independent paired final comparison of frozen whole-profile deployment rules',
                  results=reports, newly_executed_adapter_sumo_runs=0,
                  underlying_actual_component_sumo_runs=spec['expected_actual_sumo_runs'],
                  whole_profile_replays_still_required=True,
                  inherited_conditional_risk_certificates=cal['both_targets_certified'],
                  conformal_guarantee='Marginal bundle coverage; no adoption-conditioned risk certificate',
                  mean_spi_guarantee='Asymptotic population mean improvement; no adoption-conditioned risk certificate',
                  count_scope='Every reported rate is within one fixed training replicate and has n independent initial IDs. Cross-seed aggregation is used only for mean gains.',
                  actual_modes={str(seed): modes[m] for seed, m in zip(test_ids, actual_modes)},
                  plan_sha256=digest(Path(plan_directory) / 'plan.json'),
                  calibration_sha256=digest(calibration_directory / 'calibration.json'),
                  final_protocol_sha256=digest(study / 'protocol.json'), final_episodes_sha256=digest(study / 'episodes.json'))
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'comparison.json', report)
    save(output / 'selections.json', selections)
    save(output / 'sha256.json', {p.name: digest(p) for p in output.iterdir() if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps(dict(methods=list(reports), independent_initials=n, whole_profile_replays_still_required=True)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('freeze')
    p.add_argument('study')
    p.add_argument('output')
    p = sub.add_parser('calibrate')
    p.add_argument('plan')
    p.add_argument('study')
    p.add_argument('output')
    p = sub.add_parser('evaluate')
    p.add_argument('plan')
    p.add_argument('calibration')
    p.add_argument('study')
    p.add_argument('output')
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze(args.study, args.output)
    elif args.command == 'calibrate':
        calibrate(args.plan, args.study, args.output)
    else:
        evaluate(args.plan, args.calibration, args.study, args.output)
