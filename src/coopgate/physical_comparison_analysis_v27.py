"""Descriptive, paired comparisons with a common actual-command adoption event.

These functions do not select controllers or issue statistical certificates.
Initials, not modes, actors, plants or repeated executions, are sampling units.
"""
from numbers import Integral

import numpy as np


def count(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError('Nonnegative integer required: ' + name)
    return int(value)


def annotated(row, common_steps):
    steps = count(common_steps, 'common_learning_steps')
    if steps > count(row['control_steps'], 'control_steps'):
        raise ValueError('Adoption exceeds executed decisions')
    count(row['learned_action_steps'], 'original_learning_steps')
    result = dict(row, common_learning_steps=steps,
                  original_learning_steps=row['learned_action_steps'])
    validate_outcomes(result)
    return result


def validate_outcomes(row):
    for name in ('per_vehicle_finish', 'per_vehicle_variation', 'per_vehicle_cost'):
        values = np.asarray(row[name], dtype=float)
        if values.shape != (3,) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError('Three finite nonnegative outcomes required: ' + name)
    expected = np.asarray(row['per_vehicle_finish']) + .05 * np.asarray(row['per_vehicle_variation'])
    if (not np.allclose(row['per_vehicle_cost'], expected, rtol=0, atol=1e-12)
            or not np.isfinite(row['team_cost'])
            or abs(row['team_cost'] - expected.mean()) > 1e-12):
        raise ValueError('Recorded cost must retain every vehicle and capped failure outcome')
    for name in ('completed', 'collision', 'aborted'):
        if not isinstance(row[name], (bool, np.bool_)):
            raise ValueError('Boolean execution flag required: ' + name)
    count(row['contract_violations'], 'contract_violations')
    steps = count(row['control_steps'], 'control_steps')
    for name in ('common_learning_steps', 'original_learning_steps'):
        if count(row[name], name) > steps:
            raise ValueError('Adoption exceeds executed decisions')
    if row['controller'] == 'baseline' and row['common_learning_steps']:
        raise ValueError('The shared baseline cannot adopt itself as learning')


def failed(row):
    return bool(not row['completed'] or row['collision'] or row['aborted'] or row['contract_violations'])


def harmful(row, baseline):
    return bool(failed(row)
                or np.any(np.asarray(row['per_vehicle_cost']) - baseline['per_vehicle_cost'] > 1e-9)
                or np.any(np.asarray(row['per_vehicle_finish']) - baseline['per_vehicle_finish'] > 1e-9))


def identity(study, row):
    if study == 'phase':
        return row['initial_index'], row['disturbance'], row['controller']
    if study == 'transfer':
        return (row['global_initial_index'], row['scenario'], row['initial_index'], row['seed'],
                row['plant_name'], row['controller'])
    raise ValueError('Unknown study')


def expected_cases(study, spec, methods):
    if len(methods) != len(set(methods)) or 'baseline' not in methods:
        raise ValueError('Unique complete methods including baseline required')
    if study == 'phase':
        return {(i, mode, name) for i in range(spec['n_initials'])
                for mode in spec['modes'] for name in methods}
    if study == 'transfer':
        n = spec['n_initials_per_scenario']
        return {(si*n+i, scene, i, spec['seed_base']+si*1000+i, plant, name)
                for si, scene in enumerate(spec['scenarios']) for i in range(n)
                for plant in spec['plants'] for name in methods}
    raise ValueError('Unknown study')


def equal_fields(row, reference, keys):
    for key in keys:
        try:
            np.testing.assert_equal(row[key], reference[key])
        except AssertionError as exc:
            raise ValueError('Paired external inputs differ: ' + key) from exc


def validate_grid(study, rows, spec, methods):
    lookup = {identity(study, row): row for row in rows}
    if len(lookup) != len(rows) or set(lookup) != expected_cases(study, spec, methods):
        raise ValueError('Missing, duplicated or substituted method/model/initial/condition')
    for case, row in lookup.items():
        validate_outcomes(row)
        base = lookup[(*case[:-1], 'baseline')]
        equal_fields(row, base, ('seed', 'scenario', 'disturbance', 'radius', 'initial'))
        initial = np.asarray(row['initial'], dtype=float)
        if initial.shape != (3, 2) or not np.isfinite(initial).all():
            raise ValueError('Three finite paired initial states required')
        if study == 'phase':
            if row['seed'] != spec['seed_base'] + case[0]:
                raise ValueError('Phase seed differs from frozen initial')
            common = lookup[(case[0], spec['modes'][0], 'baseline')]
            equal_fields(row, common, ('seed', 'scenario', 'initial', 'radius'))
        else:
            equal_fields(row, base, ('plant', 'external_innovation_radius',
                                    'physical_error_radius', 'innovation_tape_sha256'))
            common = lookup[(*case[:4], spec['plants'][0], 'baseline')]
            equal_fields(row, common, ('seed', 'scenario', 'initial', 'disturbance', 'radius',
                                       'external_innovation_radius', 'physical_error_radius',
                                       'innovation_tape_sha256'))
            if (row['disturbance'] != spec['mode']
                    or row['external_innovation_radius'] != spec['external_innovation_radius']
                    or row['physical_error_radius'] != spec['physical_error_radius']):
                raise ValueError('Transfer noise or physical contract differs')
            error = row['max_actual_error']
            if not np.isfinite(error) or error < 0:
                raise ValueError('Finite nonnegative realized error required')
            count(row['error_contract_exceeded_substeps'], 'error_contract_exceeded_substeps')
    return lookup


def actual_modes(spec):
    return [spec['modes'][int(np.random.default_rng(np.random.SeedSequence(
        [spec['seed_base']+i, spec['mode_selection_stream']])).integers(len(spec['modes'])))]
        for i in range(spec['n_initials'])]


def bootstrap_draws(n, samples, stream):
    if n < 1 or samples < 1:
        raise ValueError('Nonempty initial-cluster bootstrap required')
    return np.random.default_rng(stream).integers(n, size=(samples, n))


def mean_interval(values, draws):
    values = np.asarray(values, dtype=float)
    draws = np.asarray(draws)
    if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
            or draws.ndim != 2 or draws.shape[1] != len(values) or not len(draws)
            or draws.dtype.kind not in 'iu' or np.any((draws < 0) | (draws >= len(values)))):
        raise ValueError('Finite paired values and matching initial-cluster draws required')
    return np.quantile(values[draws].mean(axis=1), [.025, .975]).tolist()


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def event_counts(adopted, harms):
    adopted, harms = np.asarray(adopted), np.asarray(harms)
    if adopted.dtype != np.bool_ or harms.dtype != np.bool_ or adopted.shape != harms.shape:
        raise ValueError('Matching Boolean adoption and harm arrays required')
    a, h, ah = int(adopted.sum()), int(harms.sum()), int((adopted & harms).sum())
    return dict(adopted=a, harmful=h, harmful_adopted=ah,
                empirical_conditional_risk=ratio(ah, a))


def bundle_counts(adopted, harms):
    adopted, harms = np.asarray(adopted), np.asarray(harms)
    if adopted.ndim != 2 or adopted.shape != harms.shape:
        raise ValueError('Initial-by-mode arrays required')
    # The original bundle target is (any adoption) AND (any harm), possibly
    # in different modes. It is deliberately not any(adoption AND harm).
    return event_counts(adopted.any(axis=1), harms.any(axis=1))


def diagnostics(rows):
    return dict(executions=len(rows), completed=sum(r['completed'] for r in rows),
                failed=sum(failed(r) for r in rows), aborted=sum(r['aborted'] for r in rows),
                collisions=sum(r['collision'] for r in rows),
                contract_violation_events=sum(r['contract_violations'] for r in rows))


def paired_metrics(rows, bases, draws):
    if not rows or len(rows) != len(bases):
        raise ValueError('All paired outcomes including failures required')
    gains = np.asarray([b['team_cost']-r['team_cost'] for r, b in zip(rows, bases, strict=True)])
    adopted = np.asarray([r['common_learning_steps'] > 0 for r in rows])
    original = np.asarray([r['original_learning_steps'] > 0 for r in rows])
    harms = np.asarray([harmful(r, b) for r, b in zip(rows, bases, strict=True)])
    baseline_mean = float(np.mean([b['team_cost'] for b in bases]))
    car_cost = np.asarray([np.asarray(b['per_vehicle_cost'])-r['per_vehicle_cost']
                           for r, b in zip(rows, bases, strict=True)])
    car_time = np.asarray([np.asarray(b['per_vehicle_finish'])-r['per_vehicle_finish']
                           for r, b in zip(rows, bases, strict=True)])
    result = dict(independent_initials=len(rows), mean_gain_seconds=float(gains.mean()),
        gain_pointwise_bootstrap95=mean_interval(gains, draws),
        mean_gain_percent=ratio(100*gains.mean(), baseline_mean),
        baseline_mean_team_cost=baseline_mean, **event_counts(adopted, harms),
        adoption_rate=float(adopted.mean()), original_adopted=int(original.sum()),
        changed_episode_adoption_events=int((adopted != original).sum()),
        common_learning_steps=sum(r['common_learning_steps'] for r in rows),
        original_learning_steps=sum(r['original_learning_steps'] for r in rows),
        control_steps=sum(r['control_steps'] for r in rows),
        baseline_failed=sum(failed(b) for b in bases), **diagnostics(rows),
        per_vehicle_mean_cost_gain=car_cost.mean(axis=0).tolist(),
        per_vehicle_mean_arrival_gain=car_time.mean(axis=0).tolist(),
        per_vehicle_cost_regressions=(car_cost < -1e-9).sum(axis=0).tolist(),
        per_vehicle_arrival_regressions=(car_time < -1e-9).sum(axis=0).tolist())
    if 'max_actual_error' in rows[0]:
        result.update(maximum_actual_error=max(r['max_actual_error'] for r in rows),
                      exceeded_error_runs=sum(r['error_contract_exceeded_substeps'] > 0 for r in rows))
    return result


def paired_contrast(primary, comparator, bases, draws):
    if len(primary) != len(comparator) or len(primary) != len(bases) or not primary:
        raise ValueError('Complete matched primary/comparator/base triples required')
    delta = np.asarray([c['team_cost']-p['team_cost'] for p, c in zip(primary, comparator, strict=True)])
    pe = event_counts(np.asarray([r['common_learning_steps'] > 0 for r in primary]),
                      np.asarray([harmful(r, b) for r, b in zip(primary, bases, strict=True)]))
    ce = event_counts(np.asarray([r['common_learning_steps'] > 0 for r in comparator]),
                      np.asarray([harmful(r, b) for r, b in zip(comparator, bases, strict=True)]))
    risks = (pe['empirical_conditional_risk'], ce['empirical_conditional_risk'])
    return dict(independent_initials=len(primary), primary=primary[0]['controller'],
        comparator=comparator[0]['controller'], mean_increment_seconds=float(delta.mean()),
        increment_pointwise_bootstrap95=mean_interval(delta, draws),
        positive_cost_increment_favors='primary', primary_events=pe, comparator_events=ce,
        adoption_rate_difference=(pe['adopted']-ce['adopted'])/len(primary),
        empirical_conditional_risk_difference=None if None in risks else risks[0]-risks[1],
        negative_risk_difference_favors='primary',
        failed_primary=sum(failed(r) for r in primary), failed_comparator=sum(failed(r) for r in comparator),
        failed_baseline=sum(failed(r) for r in bases))


def analyze_phase(rows, spec, methods, models, rules, samples, stream):
    lookup = validate_grid('phase', rows, spec, methods)
    modes, n = actual_modes(spec), spec['n_initials']
    draws = bootstrap_draws(n, samples, stream)
    selected = {name: [lookup[i, modes[i], name] for i in range(n)] for name in methods}
    results, subgroups = [], []
    for name in methods:
        a = np.asarray([[lookup[i, mode, name]['common_learning_steps'] > 0 for mode in spec['modes']] for i in range(n)])
        h = np.asarray([[harmful(lookup[i, mode, name], lookup[i, mode, 'baseline']) for mode in spec['modes']] for i in range(n)])
        all_modes = [lookup[i, mode, name] for i in range(n) for mode in spec['modes']]
        results.append(dict(method=name, actual=paired_metrics(selected[name], selected['baseline'], draws),
                            four_mode_bundle=bundle_counts(a, h), all_mode_diagnostics=diagnostics(all_modes)))
        for scene in sorted({r['scenario'] for r in selected['baseline']}):
            indices = [i for i, r in enumerate(selected['baseline']) if r['scenario'] == scene]
            subgroups.append(dict(method=name, scenario=scene, **paired_metrics(
                [selected[name][i] for i in indices], [selected['baseline'][i] for i in indices],
                bootstrap_draws(len(indices), samples, stream))))
    contrasts = [dict(model=str(model), **paired_contrast(selected[f'phase05_{model}'],
        selected[f'{rule}_{model}'], selected['baseline'], draws)) for model in models for rule in rules]
    return dict(actual_mode_by_initial=modes, independent_initials=n, executions=len(rows),
                methods=results, scenario_subgroups=subgroups, primary_contrasts=contrasts)


def analyze_transfer(rows, spec, methods, models, rules, samples, stream):
    lookup = validate_grid('transfer', rows, spec, methods)
    n = spec['n_initials_per_scenario']
    draws = bootstrap_draws(n, samples, stream)
    cells, contrasts = [], []
    for si, scene in enumerate(spec['scenarios']):
        for plant in spec['plants']:
            selected = {name: [lookup[(si*n+i, scene, i, spec['seed_base']+si*1000+i, plant, name)]
                               for i in range(n)] for name in methods}
            for name in methods:
                cells.append(dict(scenario=scene, plant=plant, method=name,
                                  **paired_metrics(selected[name], selected['baseline'], draws)))
            for model in models:
                for rule in rules:
                    contrasts.append(dict(scenario=scene, plant=plant, model=str(model), **paired_contrast(
                        selected[f'phase05_{model}'], selected[f'{rule}_{model}'], selected['baseline'], draws)))
    return dict(independent_initials=n*len(spec['scenarios']), initials_per_cell=n,
                executions=len(rows), cells=cells, primary_contrasts=contrasts)
