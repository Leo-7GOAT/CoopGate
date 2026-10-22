"""Independently reconstruct the frozen comparison and its joint requirements.

This reporter changes no candidate, selection, sample size, threshold, or CI.
It reads finalized studies only. Synthetic checks do not count as SUMO evidence.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.stats import beta, binom

from compare_established_v19 import complete_study, read_plan
from probe_disturbed_gate_v19 import digest, save
from start_final_v19 import check_launch, require_committed, verify_manifest

ROOT = Path(__file__).resolve().parents[1]
V19 = ROOT / 'artifacts/v19'
REPORT_PLAN = V19 / 'final_report_plan'
TARGETS = ('actual_random_mode', 'worst_mode_bundle')


def same(actual, expected, name):
    """Check output identities, not a tolerance-based change to admission."""
    if actual is None or expected is None or isinstance(expected, (bool, str)):
        if actual != expected:
            raise ValueError('Report mismatch: ' + name)
    elif isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError('Report keys mismatch: ' + name)
        for key, value in expected.items():
            same(actual[key], value, name + '/' + str(key))
    elif isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            raise ValueError('Report sequence mismatch: ' + name)
        for index, value in enumerate(expected):
            same(actual[index], value, name + '/' + str(index))
    else:
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12, err_msg=name)


def physical_failure(row):
    return bool(not row['completed'] or row['collision'] or row['aborted'] or row['contract_violations'])


def observation(row, base):
    cost = np.asarray(row['per_vehicle_cost'], float)
    finish = np.asarray(row['per_vehicle_finish'], float)
    base_cost = np.asarray(base['per_vehicle_cost'], float)
    base_finish = np.asarray(base['per_vehicle_finish'], float)
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in (cost, finish, base_cost, base_finish)):
        raise ValueError('Finite three-vehicle outcomes required')
    same(row['team_cost'], cost.mean(), 'row team cost')
    same(base['team_cost'], base_cost.mean(), 'baseline team cost')
    adopted = row['learned_action_steps'] > 0
    failed = physical_failure(row)
    harmful = bool(np.any(cost - base_cost > 1e-9) or np.any(finish - base_finish > 1e-9) or failed)
    # A zero-adoption episode must be compared with the actual baseline,
    # including baseline failures. Equality is not an absolute safety claim.
    fallback_match = True
    if not adopted:
        for field in ('per_vehicle_cost', 'per_vehicle_finish', 'per_vehicle_variation',
                      'completed', 'collision', 'aborted', 'contract_violations'):
            fallback_match &= bool(np.array_equal(row[field], base[field]))
    return dict(adopted=adopted, harmful=harmful, failed=failed,
                gain=float(np.mean(base_cost - cost)), baseline_cost=float(base_cost.mean()),
                fallback_outcomes_match=bool(fallback_match))


def counts(adopted, harmful):
    adopted, harmful = list(adopted), list(harmful)
    if len(adopted) != len(harmful) or not adopted:
        raise ValueError('Nonempty paired episode indicators required')
    n = sum(bool(a) for a in adopted)
    k = sum(bool(a) and bool(h) for a, h in zip(adopted, harmful))
    return dict(adopted=n, harmful_adopted=k, conditional_harm=k / n if n else None,
                adoption_fraction=n / len(adopted))


def exact_bound(n, k, level, alpha):
    if not 0 <= k <= n or not 0 < level < 1 or not 0 < alpha < 1:
        raise ValueError('Invalid conditional-binomial design')
    upper = float(beta.ppf(1 - level, k + 1, n - k)) if n > k else 1.
    p = float(binom.cdf(k, n, alpha)) if n else 1.
    return upper, p, bool(n > 0 and p <= level)


def modes_for(spec, stream):
    return [spec['modes'][int(np.random.default_rng(np.random.SeedSequence(
        [spec['seed_base'] + i, stream])).integers(len(spec['modes'])))]
        for i in range(spec['n_initials'])]


def audit_risk(lookup, spec, risk):
    names = [name for name in spec['methods'] if name != 'baseline']
    config = spec['risk_validation']
    actual_modes = modes_for(spec, config['mode_selection_stream'])
    same(risk['alpha'], config['alpha'], 'risk alpha')
    same(risk['total_delta'], config['delta'], 'total delta')
    same(risk['independent_initials'], spec['n_initials'], 'risk independent initials')
    same(risk['actual_random_modes'], {str(spec['seed_base'] + i): mode for i, mode in enumerate(actual_modes)},
         'calibration actual modes')
    level = config['delta'] / (len(names) * len(TARGETS))
    reconstructed, certified = {}, []
    for name in names:
        arrays = {target: ([], []) for target in TARGETS}
        for i, actual_mode in enumerate(actual_modes):
            bundle = [observation(lookup[i, mode, name], lookup[i, mode, 'baseline']) for mode in spec['modes']]
            actual = bundle[spec['modes'].index(actual_mode)]
            arrays[TARGETS[0]][0].append(actual['adopted'])
            arrays[TARGETS[0]][1].append(actual['harmful'])
            arrays[TARGETS[1]][0].append(any(r['adopted'] for r in bundle))
            arrays[TARGETS[1]][1].append(any(r['harmful'] for r in bundle))
        results = {}
        for target, (adopted, harmful) in arrays.items():
            count = counts(adopted, harmful)
            upper, p, passed = exact_bound(count['adopted'], count['harmful_adopted'], level, config['alpha'])
            table = risk['targets'][target]
            same(table['independent_initials'], spec['n_initials'], target + ' sample size')
            same(table['delta'], config['delta'] / 2, target + ' error budget')
            expected = dict(candidate=name, adopted=count['adopted'], harmful=count['harmful_adopted'],
                            empirical_conditional_risk=count['conditional_harm'], binomial_pvalue=p,
                            threshold_upper=upper, certified=passed, tested=True)
            recorded = next(r for r in table['candidates'] if r['candidate'] == name)
            same(recorded, expected, target + '/' + name)
            results[target] = expected
        reconstructed[name] = results
        if all(r['certified'] for r in results.values()):
            certified.append(name)
    same(risk['both_targets_certified'], certified, 'simultaneously certified identities')
    return reconstructed


def expected_selection(method, seed, i, lookup, names, calibration):
    if method in ('raw', 'nominal', 'portfolio0'):
        return method + '_' + seed
    if method in ('ltt', 'mean_spi'):
        return calibration[method + '_selection'][seed]
    if method != 'conformal':
        raise ValueError('Unknown frozen adapter')
    if calibration['conformal']['infinite_q'] or not lookup[i, 'clean', 'baseline']['completed']:
        return 'baseline'
    # Reconstruct using only the observed-initial clean forecast. The actual
    # mode, actual outcome and selected future trajectory are never arguments.
    q, options = calibration['conformal']['q'], []
    base = lookup[i, 'clean', 'baseline']
    for name in names:
        row = lookup[i, 'clean', name]
        if not name.endswith('_' + seed) or physical_failure(row):
            continue
        upper_cost = np.asarray(row['per_vehicle_cost']) - base['per_vehicle_cost'] + q
        upper_time = np.asarray(row['per_vehicle_finish']) - base['per_vehicle_finish'] + q
        if np.all(upper_cost <= 0) and np.all(upper_time <= 0) and upper_cost.mean() < -1e-9:
            options.append((float(upper_cost.mean()), name))
    # min by value keeps the frozen family ordering for equal forecast gains.
    return min(options, key=lambda pair: pair[0])[1] if options else 'baseline'


def certificate_for(method, profiles, certified):
    """Only a calibration-fixed complete profile inherits its certificate.

    State-dependent choice among individually certified profiles does not in
    general preserve their adoption-conditioned risks. Conformal has its own
    marginal coverage statement, not the LTT conditional guarantee.
    """
    unique = set(profiles)
    if method == 'conformal':
        return dict(certified=False, reason='state_dependent_selection_requires_own_conditional_calibration')
    if unique == {'baseline'}:
        return dict(certified=False, reason='baseline_only_no_learning_conditional_certificate')
    if len(unique) != 1:
        raise ValueError('A calibration-fixed adapter changed profiles across test states')
    name = next(iter(unique))
    return dict(certified=name in certified,
                reason='fixed_profile_has_both_certificates' if name in certified else 'fixed_profile_not_certified')


def target_check(metrics, certificate, design):
    conditions = dict(
        both_conditional_risk_targets=certificate['certified'],
        actual_adoption=metrics['actual_random_mode']['adoption_fraction'] >= design['actual_adoption_minimum'],
        mean_gain=metrics['mean_team_gain_percent'] >= design['mean_gain_percent_minimum'],
        positive_paired_gain_interval=metrics['mean_team_gain_ci95'][0] > 0.)
    return dict(conditions=conditions, all_conditions_met=all(conditions.values()))


def audit_final(lookup, spec, plan, calibration, comparison, selections):
    n, seeds, methods, names = spec['n_initials'], plan['seeds'], plan['final_test']['reported_methods'], plan['profiles']
    actual_modes = modes_for(spec, plan['final_test']['mode_selection_stream'])
    same(comparison['actual_modes'], {str(spec['seed_base'] + i): mode for i, mode in enumerate(actual_modes)},
         'final actual modes')
    selected = {(r['initial_index'], r['adapter'], r['training_seed']): r for r in selections}
    expected = {(i, method, seed) for i in range(n) for method in methods for seed in seeds}
    if set(selected) != expected or len(selections) != len(expected):
        raise ValueError('Missing, duplicate or extra complete-profile selections')
    bootstrap = np.random.default_rng(plan['final_test']['bootstrap_seed']).integers(
        n, size=(plan['final_test']['bootstrap_samples'], n))
    results = {}
    for method in methods:
        per_seed, family_gains = {}, []
        for seed in seeds:
            observations, bundles, profiles, times, completed = [], [], [], [], 0
            for i, mode in enumerate(actual_modes):
                record = selected[i, method, seed]
                same(record['seed'], spec['seed_base'] + i, 'selection initial seed')
                profile = expected_selection(method, seed, i, lookup, names, calibration)
                same(record['profile'], profile, 'whole-profile choice')
                profiles.append(profile)
                bundles.append([observation(lookup[i, m, profile], lookup[i, m, 'baseline']) for m in spec['modes']])
                observations.append(bundles[-1][spec['modes'].index(mode)])
                row = lookup[i, mode, profile]
                completed += bool(row['completed'])
                times.extend(row['controller_ms'])
            gain = np.array([r['gain'] for r in observations])
            family_gains.append(gain)
            base_mean = np.mean([r['baseline_cost'] for r in observations])
            metrics = dict(
                actual_random_mode=counts([r['adopted'] for r in observations], [r['harmful'] for r in observations]),
                worst_mode_bundle=counts([any(r['adopted'] for r in b) for b in bundles],
                                         [any(r['harmful'] for r in b) for b in bundles]),
                completed=completed, runs=n, selected_profiles=dict(Counter(profiles)),
                mean_team_gain_seconds=float(gain.mean()), mean_team_gain_percent=float(100 * gain.mean() / base_mean),
                mean_team_gain_ci95=np.quantile(gain[bootstrap].mean(1), [.025, .975]).tolist(),
                controller_p95_ms=float(np.percentile(times, 95)) if times else None,
                controller_p99_ms=float(np.percentile(times, 99)) if times else None,
                controller_max_ms=float(max(times)) if times else None,
                steps_over_500ms=sum(t > 500 for t in times), total_control_steps=len(times))
            recorded = comparison['results'][method]['per_training_seed'][seed]
            for field, value in metrics.items():
                same(recorded[field], value, method + '/' + seed + '/' + field)
            if method == 'conformal':
                shadows = [sum(lookup[i, 'clean', name]['setup_ms'] + sum(lookup[i, 'clean', name]['controller_ms'])
                               for name in ('baseline', *names)) for i in range(n)]
                same(recorded['shadow_initialization_controller_time_lower_bound_ms'],
                     dict(p50=float(np.median(shadows)), p95=float(np.percentile(shadows, 95)), maximum=float(max(shadows))),
                     'all-profile preparation lower bound')
            certificate = certificate_for(method, profiles, calibration['both_targets_certified'])
            physical = dict(actual_failed_episodes=sum(r['failed'] for r in observations),
                            actual_baseline_failed_episodes=sum(physical_failure(lookup[i, mode, 'baseline'])
                                                               for i, mode in enumerate(actual_modes)),
                            actual_zero_adoption_episodes=sum(not r['adopted'] for r in observations),
                            zero_adoption_failures=sum(r['failed'] and not r['adopted'] for r in observations),
                            zero_adoption_outcome_mismatches=[i for i, r in enumerate(observations)
                                                            if not r['fallback_outcomes_match']],
                            any_mode_failed_initials=sum(any(r['failed'] for r in b) for b in bundles))
            # Stratified values describe scope; neither mixture certificate nor
            # the mixture CI is promoted to a subgroup-specific guarantee.
            strata = {}
            for scene in sorted({lookup[i, actual_modes[i], 'baseline']['scenario'] for i in range(n)}):
                subset = [r for i, r in enumerate(observations) if lookup[i, actual_modes[i], 'baseline']['scenario'] == scene]
                strata[scene] = dict(independent_initials=len(subset),
                                     actual_random_mode=counts([r['adopted'] for r in subset], [r['harmful'] for r in subset]),
                                     mean_team_gain_percent=float(100 * np.mean([r['gain'] for r in subset]) /
                                                                  np.mean([r['baseline_cost'] for r in subset])),
                                     physical_failures=sum(r['failed'] for r in subset), subgroup_risk_certified=False)
            per_seed[seed] = dict(metrics=metrics, conditional_certificate=certificate,
                                  target_check=target_check(metrics, certificate, plan['final_test']),
                                  physical_outcomes=physical, descriptive_regimes=strata)
        # Keep the same initial together across models. Three fixed trained
        # models are not three times as many independent scene samples.
        family_gain = np.asarray(family_gains).mean(0)
        family = dict(independent_initials=n, mean_team_gain_seconds=float(family_gain.mean()),
                      mean_team_gain_ci95=np.quantile(family_gain[bootstrap].mean(1), [.025, .975]).tolist())
        for field, value in family.items():
            same(comparison['results'][method][field], value, method + '/family/' + field)
        results[method] = dict(per_training_seed=per_seed, fixed_model_average=family,
                               all_replicates_meet_targets=all(r['target_check']['all_conditions_met'] for r in per_seed.values()),
                               passing_training_seeds=[seed for seed, r in per_seed.items() if r['target_check']['all_conditions_met']])
    return results


def development_preflight(output):
    """Exercise the independent reducer on completed, public transfer data.

    This is post-hoc accounting, not an additional experiment or calibration.
    It never reads the live calibration shards or the future final split.
    """
    study, analysis_dir = V19 / 'shifted_study', V19 / 'shifted_analysis'
    verify_manifest(study)
    verify_manifest(analysis_dir)
    summary = json.loads((study / 'summary.json').read_text())
    same(summary['episodes_sha256'], digest(study / 'episodes.json'), 'completed transfer episodes')
    rows = json.loads((study / 'episodes.json').read_text())
    same(len(rows), summary['actual_sumo_runs'], 'completed transfer sample count')
    analysis = json.loads((analysis_dir / 'analysis.json').read_text())
    lookup = {(r['scenario'], r['initial_index'], r['plant_name'], r['controller']): r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError('Duplicated transfer records')
    reconstructed, failures, zero_adoption = [], [], []
    paired = 0
    for cell in analysis['cells']:
        models = []
        for reference in cell['per_model']:
            observations, originals = [], []
            name = cell['family'] + '_' + reference['seed']
            for i in range(cell['independent_initials']):
                key = (cell['scenario'], i, cell['plant'])
                row, base = lookup[*key, name], lookup[*key, 'baseline']
                item = observation(row, base)
                originals.append(row)
                observations.append(item)
                paired += 1
                record = dict(scenario=cell['scenario'], initial_index=i, plant=cell['plant'],
                              controller=name, failed=item['failed'], fallback_outcomes_match=item['fallback_outcomes_match'])
                if not item['adopted']:
                    zero_adoption.append(record)
                if item['failed']:
                    failures.append(record)
            actual = dict(seed=reference['seed'], adopted=sum(r['adopted'] for r in observations),
                          harmful=sum(r['harmful'] for r in observations),
                          harmful_adopted=sum(r['adopted'] and r['harmful'] for r in observations),
                          completed=sum(r['completed'] for r in originals),
                          mean_team_gain_seconds=float(np.mean([r['gain'] for r in observations])))
            same(actual, reference, 'transfer cell ' + name)
            models.append(actual)
        reconstructed.append(dict(scenario=cell['scenario'], plant=cell['plant'], family=cell['family'], per_model=models))
    report = dict(purpose='Post-hoc independent final-report reducer preflight on completed transfer outcomes',
                  synthetic=False, additional_sumo_runs=0, calibration_outcomes_read=False, final_outcomes_read=False,
                  risk_certified=False, source_study_runs=len(rows), reconstructed_cells=len(reconstructed),
                  reconstructed_model_cells=sum(len(c['per_model']) for c in reconstructed), paired_controller_outcomes=paired,
                  all_recorded_model_cell_values_match=True, zero_adoption_episodes=len(zero_adoption),
                  zero_adoption_outcome_mismatches=[r for r in zero_adoption if not r['fallback_outcomes_match']],
                  zero_adoption_physical_failures=[r for r in zero_adoption if r['failed']],
                  all_nonbaseline_physical_failures=failures, baseline_physical_failures=sum(
                      physical_failure(r) for r in rows if r['controller'] == 'baseline'),
                  cells=reconstructed,
                  scope='Correlated fixed-model transfer records; outcome equality only, not all command trajectories or absolute fallback safety.',
                  source_sha256=digest(__file__),
                  input_sha256={path.relative_to(V19).as_posix(): digest(path) for path in
                                (study / 'episodes.json', study / 'summary.json', study / 'protocol.json', analysis_dir / 'analysis.json')})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'audit.json', report)
    shutil.copy2(__file__, output / Path(__file__).name)
    save(output / 'sha256.json', {p.name: digest(p) for p in output.iterdir() if p.name != 'sha256.json'})
    print(json.dumps({key: report[key] for key in ('reconstructed_cells', 'reconstructed_model_cells',
                     'paired_controller_outcomes', 'zero_adoption_episodes', 'baseline_physical_failures')}
                     | dict(zero_adoption_outcome_mismatches=len(report['zero_adoption_outcome_mismatches']),
                            zero_adoption_physical_failures=len(report['zero_adoption_physical_failures']),
                            risk_certified=False, additional_sumo_runs=0)), flush=True)


def freeze():
    if (V19 / 'comparison_calibration/calibration.json').exists() or (V19 / 'final_launch.json').exists():
        raise ValueError('Freeze this reporting plan before calibration selection or final execution')
    source = Path(__file__).resolve()
    plan = read_plan(V19 / 'comparison_plan')
    record = dict(purpose='Independent final accounting and conjunction of the original risk/usefulness requirements',
                  comparison_plan_sha256=digest(V19 / 'comparison_plan/plan.json'),
                  risk_protocol_sha256=digest(V19 / 'risk_validation/protocol.json'),
                  final_protocol_sha256=digest(V19 / 'final_confirmation/protocol.json'),
                  source_path=source.relative_to(ROOT).as_posix(), source_sha256=digest(source),
                  thresholds={key: plan['final_test'][key] for key in ('actual_adoption_minimum', 'mean_gain_percent_minimum')},
                  replicate_reporting='Report every fixed model separately and their paired initial-cluster average. The all-replicates flag requires every model to pass; no best-seed substitution.',
                  certificates='Recompute original calibration counts/bounds. A state-dependent selector cannot inherit a fixed-profile conditional certificate.',
                  interval_scope='Original percentile bootstrap intervals and Gaussian bands retain their approximate/asymptotic scope; no new finite-sample efficacy claim.',
                  physical_scope='Retain all physical failures, including zero-adoption and baseline failures. Equal fallback outcomes do not imply absolute safety.',
                  nonclaims=['new controller', 'new certification method', 'changed original criteria', 'real-time feasibility', 'goal completion'],
                  calibration_outcomes_read=False, final_outcomes_read=False)
    REPORT_PLAN.mkdir(parents=True, exist_ok=False)
    save(REPORT_PLAN / 'plan.json', record)
    shutil.copy2(source, REPORT_PLAN / source.name)
    save(REPORT_PLAN / 'sha256.json', {p.name: digest(p) for p in REPORT_PLAN.iterdir() if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps(dict(report_plan_sha256=digest(REPORT_PLAN / 'plan.json'), outcomes_read=False)), flush=True)


def run(output):
    # Check all prerequisites before creating an output directory. Never accept
    # a partially written shard or a hash manifest alone as a completed study.
    verify_manifest(REPORT_PLAN)
    report_plan = json.loads((REPORT_PLAN / 'plan.json').read_text())
    require_committed([REPORT_PLAN / 'plan.json', REPORT_PLAN / 'sha256.json'])
    same(digest(__file__), report_plan['source_sha256'], 'frozen reporter source')
    check_launch()
    for key, path in (('comparison_plan_sha256', 'comparison_plan/plan.json'),
                      ('risk_protocol_sha256', 'risk_validation/protocol.json'),
                      ('final_protocol_sha256', 'final_confirmation/protocol.json')):
        same(digest(V19 / path), report_plan[key], key)
    for folder in ('comparison_calibration', 'final_comparison', 'comparison_replay'):
        verify_manifest(V19 / folder)
    plan = read_plan(V19 / 'comparison_plan')
    calibration = json.loads((V19 / 'comparison_calibration/calibration.json').read_text())
    risk = json.loads((V19 / 'comparison_calibration/conditional_risk.json').read_text())
    comparison = json.loads((V19 / 'final_comparison/comparison.json').read_text())
    replay = json.loads((V19 / 'comparison_replay/audit.json').read_text())
    same(calibration['both_targets_certified'], risk['both_targets_certified'], 'calibration certificate identities')
    same(comparison['inherited_conditional_risk_certificates'], risk['both_targets_certified'], 'comparison certificate identities')
    same(calibration['conditional_risk_sha256'], digest(V19 / 'comparison_calibration/conditional_risk.json'), 'risk file binding')
    for key, path in (('calibration_sha256', 'comparison_calibration/calibration.json'),
                      ('final_episodes_sha256', 'final_confirmation/episodes.json'),
                      ('plan_sha256', 'comparison_plan/plan.json')):
        same(comparison[key], digest(V19 / path), key)
    for path, expected in replay['source_sha256'].items():
        same(digest(ROOT / path.replace('\\', '/')), expected, 'replay binding')
    for flag in ('whole_profile_replay_requirement_met', 'all_selected_profiles_match', 'all_episode_metrics_match'):
        same(replay[flag], True, flag)
    same(replay['actual_deployment_replays'], replay['new_trajectories_physically_reconstructed'], 'reconstructed replays')
    if replay['actual_deployment_replays'] < 1:
        raise ValueError('Actual replay evidence required')
    risk_protocol, risk_lookup, _, _, _ = complete_study(V19 / 'risk_validation', 'risk_validation')
    risk_results = audit_risk(risk_lookup, risk_protocol['spec'], risk)
    protocol, lookup, _, _, _ = complete_study(V19 / 'final_confirmation', 'final_confirmation')
    selections = json.loads((V19 / 'final_comparison/selections.json').read_text())
    results = audit_final(lookup, protocol['spec'], plan, calibration, comparison, selections)
    dependencies = ['final_report_plan/plan.json', 'comparison_plan/plan.json',
                    'risk_validation/episodes.json', 'risk_validation/summary.json',
                    'comparison_calibration/calibration.json', 'comparison_calibration/conditional_risk.json',
                    'final_confirmation/episodes.json', 'final_confirmation/summary.json',
                    'final_comparison/comparison.json', 'final_comparison/selections.json', 'comparison_replay/audit.json']
    report = dict(purpose=report_plan['purpose'], calibration_reconstruction=risk_results, results=results,
                  original_report_values_reproduced=True, real_whole_profile_replays_verified=True,
                  newly_executed_sumo_runs=0, real_time_certified=False,
                  strict_gate_all_replicates_meet_original_targets=results['portfolio0']['all_replicates_meet_targets'],
                  all_replicates_interpretation=report_plan['replicate_reporting'],
                  goal_completion_claimed=False, source_sha256=report_plan['source_sha256'],
                  input_sha256={path: digest(V19 / path) for path in dependencies})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'report.json', report)
    save(output / 'sha256.json', {'report.json': digest(output / 'report.json')})
    print(json.dumps(dict(strict_gate_all_replicates_meet_original_targets=report['strict_gate_all_replicates_meet_original_targets'],
                          original_report_values_reproduced=True, goal_completion_claimed=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('freeze')
    p = sub.add_parser('preflight')
    p.add_argument('output')
    p = sub.add_parser('run')
    p.add_argument('output')
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze()
    elif args.command == 'preflight':
        development_preflight(args.output)
    else:
        run(args.output)
