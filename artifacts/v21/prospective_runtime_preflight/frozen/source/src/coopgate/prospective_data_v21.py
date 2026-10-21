"""Common complete paired observations for all prospective statistical adapters.

No selection, fitting, or sample-size change happens here. Four disturbance
modes and every fixed profile belong to one independent initial cluster.
"""
from numbers import Integral
from collections import Counter

import numpy as np

from .comparison_v19 import conditional_counts, conformal_profile
from .joint_admission_v15 import JointBound


MODES = ('clean', 'uniform', 'bias', 'pulse')


def integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(name + ' must be an integer in its declared range')
    return int(value)


def paired_observations(rows, *, profiles, independent_initials, seed_base, methods):
    """Build finite per-car cost/time differences without removing failures.

    Supports streamed records in any order, but requires the entire common grid.
    Cost accounting is verified against independently provided arrival and
    command-variation fields. Every controller/mode must share initial state.
    """
    n = integer(independent_initials, 'Independent initial count', 1)
    seed_base = integer(seed_base, 'Initial seed base', 1)
    names = tuple(profiles)
    if not names or len(set(names)) != len(names) or 'baseline' in names:
        raise ValueError('Unique non-baseline common profiles required')
    controllers = ('baseline', *names)
    if set(methods) != set(controllers):
        raise ValueError('Every method must belong to the same frozen candidate family')
    shape = (n, len(MODES), len(controllers))
    seen = np.zeros(shape, dtype=bool)
    used = np.zeros(shape, dtype=bool)
    failed = np.zeros(shape, dtype=bool)
    completed = np.zeros(shape, dtype=bool)
    metrics = np.zeros((*shape, 3, 2))
    timings = np.empty(shape, dtype=object)
    setup = np.zeros(shape)
    initials = np.zeros((n, 3, 2))
    initial_seen = np.zeros(n, dtype=bool)
    scenarios = [None] * n
    for row in rows:
        i = integer(row['initial_index'], 'Initial index')
        if i >= n or row['seed'] != seed_base + i:
            raise ValueError('Observation is outside the frozen initial sample')
        if row['disturbance'] not in MODES or row['controller'] not in controllers:
            raise ValueError('Unexpected disturbance or candidate profile')
        m, j = MODES.index(row['disturbance']), controllers.index(row['controller'])
        if seen[i, m, j]:
            raise ValueError('Repeated case cannot become an independent observation')
        initial = np.asarray(row['initial'], dtype=float)
        if initial.shape != (3, 2) or not np.isfinite(initial).all():
            raise ValueError('Finite observed three-vehicle initial state required')
        if initial_seen[i]:
            if not np.array_equal(initials[i], initial) or scenarios[i] != row['scenario']:
                raise ValueError('Methods or disturbance modes do not share the same initial scene')
        else:
            initials[i], scenarios[i], initial_seen[i] = initial, row['scenario'], True
        finish, variation, cost = (np.asarray(row[k], dtype=float) for k in
                                  ('per_vehicle_finish', 'per_vehicle_variation', 'per_vehicle_cost'))
        if any(v.shape != (3,) or not np.isfinite(v).all() for v in (finish, variation, cost)):
            raise ValueError('All three vehicles require finite capped outcomes, including failures')
        if (np.any(finish < 0) or np.any(variation < 0)
                or not np.allclose(cost, finish + .05 * variation, atol=1e-10, rtol=0)
                or not np.isclose(float(row['team_cost']), cost.mean(), atol=1e-10, rtol=0)):
            raise ValueError('Cost does not match registered arrival and control variation')
        steps = integer(row['learned_action_steps'], 'Learning action count')
        if steps > integer(row['control_steps'], 'Executed control steps'):
            raise ValueError('Learning count exceeds executed decisions')
        if j == 0 and steps:
            raise ValueError('Baseline cannot adopt a learning action')
        elapsed = np.asarray(row['controller_ms'], dtype=float)
        initialization = float(row['setup_ms'])
        if (elapsed.shape != (row['control_steps'],) or not np.isfinite(elapsed).all()
                or np.any(elapsed < 0) or not np.isfinite(initialization) or initialization < 0):
            raise ValueError('Every actual control call and setup time must be retained')
        metrics[i, m, j] = np.column_stack((cost, finish))
        timings[i, m, j], setup[i, m, j] = elapsed, initialization
        used[i, m, j] = steps > 0
        completed[i, m, j] = bool(row['completed'])
        failed[i, m, j] = bool(not row['completed'] or row['collision'] or row['aborted'] or row['contract_violations'])
        seen[i, m, j] = True
    if not seen.all():
        raise ValueError('Incomplete common initial/mode/controller grid')
    differences = metrics[:, :, 1:] - metrics[:, :, :1]
    for j, name in enumerate(names):
        if methods[name]['kind'] == 'raw':
            continue
        fallback = ~used[:, :, j + 1]
        if (np.any(np.abs(differences[:, :, j][fallback]) > 1e-9)
                or np.any(failed[:, :, j + 1][fallback] != failed[:, :, 0][fallback])):
            raise ValueError('A zero-adoption gate differs from its paired baseline')
    adoption = used[:, :, 1:]
    failure = failed[:, :, 1:]
    harmful = failure | np.any(differences > 1e-9, axis=(-1, -2))
    return dict(names=names, adopted=adoption, harmful=harmful, differences=differences, failed=failure,
                baseline_metrics=metrics[:, :, 0], baseline_failed=failed[:, :, 0],
                completed=completed[:, :, 1:], baseline_completed=completed[:, :, 0],
                controller_timings=timings, setup_timings=setup,
                initials=initials, scenarios=tuple(scenarios), independent_initials=n,
                actual_controller_runs=int(seen.sum()))


def actual_mode_indices(independent_initials, seed_base, stream):
    n = integer(independent_initials, 'Independent initial count', 1)
    seed_base = integer(seed_base, 'Initial seed base', 1)
    stream = integer(stream, 'Mode selection stream')
    return np.asarray([np.random.default_rng(np.random.SeedSequence([seed_base + i, stream])).integers(4)
                       for i in range(n)])


def evaluate_adapters(data, calibration, *, family, seeds, seed_base, mode_stream,
                      bootstrap_seed, bootstrap_samples=5000, conformal_alpha=.05):
    """Evaluate all common whole-profile adapters on one complete final grid.

    LTT and Mean SPI use their fixed calibration choices. Conformal uses only
    the clean complete-profile prediction and forecast-failure flags for that
    initial. The collection/deployment layer must separately verify that these
    clean predictions are available from observed initial state alone.
    """
    names, n = tuple(data['names']), data['independent_initials']
    seeds = tuple(map(str, seeds))
    if set(names) != {f'{prefix}_{seed}' for seed in seeds for prefix in ('raw', 'nominal', family)}:
        raise ValueError('Every method must use the same complete three-profile family per model')
    if set(calibration['ltt_selection']) != set(seeds) or set(calibration['mean_spi_selection']) != set(seeds):
        raise ValueError('Missing fixed-model calibration choice')
    for selector in ('ltt_selection', 'mean_spi_selection'):
        for seed, name in calibration[selector].items():
            if name != 'baseline' and (name not in names or not name.endswith('_' + seed)):
                raise ValueError('Calibrated selection crosses models or introduces a new profile')
    modes = actual_mode_indices(n, seed_base, mode_stream)
    draws = np.random.default_rng(bootstrap_seed).integers(n, size=(integer(bootstrap_samples, 'Bootstrap count', 1), n))
    conformal = calibration['conformal']
    q = np.inf if conformal['infinite_q'] else conformal['q']
    bound = JointBound(names, ((1., 1.),) * 3, q, conformal['calibration_initials'], conformal_alpha)
    d, a, h, f = (data[k] for k in ('differences', 'adopted', 'harmful', 'failed'))
    baseline_cost = data['baseline_metrics'][np.arange(n), modes, :, 0].mean(-1).mean()
    shadow_times = [sum(data['setup_timings'][i, 0, j] + data['controller_timings'][i, 0, j].sum()
                        for j in range(len(names) + 1)) for i in range(n)]
    reports, selections = {}, []
    for method in ('raw', 'nominal', family, 'conformal', 'ltt', 'mean_spi'):
        members, gains = {}, []
        for seed in seeds:
            choices = []
            for i in range(n):
                if method == 'conformal':
                    choice = conformal_profile(bound, d[i, 0], names, seed, f[i, 0])
                    if not data['baseline_completed'][i, 0]:
                        choice = 'baseline'
                elif method in ('ltt', 'mean_spi'):
                    choice = calibration[method + '_selection'][seed]
                else:
                    choice = f'{method}_{seed}'
                choices.append(choice)
                selections.append(dict(initial_index=i, seed=seed_base + i, training_seed=seed, adapter=method, profile=choice))
            used, harmful, completed, gain, bundle_a, bundle_h, times = [], [], [], [], [], [], []
            for i, name in enumerate(choices):
                if name == 'baseline':
                    cluster_a = np.zeros(4, dtype=bool)
                    cluster_h = data['baseline_failed'][i]
                    gain.append(0.)
                    completed.append(data['baseline_completed'][i, modes[i]])
                    column = 0
                else:
                    j = names.index(name)
                    cluster_a, cluster_h = a[i, :, j], h[i, :, j]
                    gain.append(float(-d[i, modes[i], j, :, 0].mean()))
                    completed.append(data['completed'][i, modes[i], j])
                    column = j + 1
                used.append(cluster_a[modes[i]])
                harmful.append(cluster_h[modes[i]])
                bundle_a.append(cluster_a.any())
                bundle_h.append(cluster_h.any())
                times.extend(data['controller_timings'][i, modes[i], column])
            gain = np.asarray(gain)
            gains.append(gain)
            members[seed] = dict(actual_random_mode=conditional_counts(np.asarray(used), np.asarray(harmful)),
                worst_mode_bundle=conditional_counts(np.asarray(bundle_a), np.asarray(bundle_h)),
                completed=int(sum(completed)), runs=n, selected_profiles=dict(Counter(choices)),
                mean_team_gain_seconds=float(gain.mean()), mean_team_gain_percent=float(100 * gain.mean() / baseline_cost),
                mean_team_gain_ci95=np.quantile(gain[draws].mean(1), [.025, .975]).tolist(),
                controller_p95_ms=float(np.percentile(times, 95)) if times else None,
                controller_p99_ms=float(np.percentile(times, 99)) if times else None,
                controller_max_ms=float(max(times)) if times else None,
                steps_over_500ms=int(sum(t > 500 for t in times)), total_control_steps=len(times))
            if method == 'conformal':
                members[seed]['shadow_initialization_controller_time_lower_bound_ms'] = dict(
                    p50=float(np.median(shadow_times)), p95=float(np.percentile(shadow_times, 95)), maximum=float(max(shadow_times)))
        gains = np.asarray(gains).mean(0)
        reports[method] = dict(per_training_seed=members, independent_initials=n, mean_team_gain_seconds=float(gains.mean()),
                              mean_team_gain_ci95=np.quantile(gains[draws].mean(1), [.025, .975]).tolist())
    return dict(results=reports, selections=selections, actual_modes=modes.tolist(),
                new_sumo_runs=0, new_risk_certification=False, actual_deployment_replays_still_required=True)
