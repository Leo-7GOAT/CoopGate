"""Prospective whole-family selection and fixed-sample validation rules.

Development selects one complete family before new calibration. A summable
error budget prevents repeated failed attempts from becoming test-until-pass.
These are standard sample splitting, intersection-union and union-bound tools.
"""
from numbers import Integral

import numpy as np
from scipy.stats import beta

from .comparison_v19 import conformal_episode_bound, select_fixed_replicates
from .family_risk_v20 import MODELS, TARGETS, family_certificate
from .risk_comparators_v19 import conditional_ltt, paired_mean_improvement_bands


FAMILIES = ('phase05', 'phase10', 'repair')
MODES = ('clean', 'uniform', 'bias', 'pulse')
CALIBRATION_N = 4096
CONFIRMATION_N = 1024
ADOPTION_MIN = .20
GAIN_PERCENT_MIN = .5
RISK_LIMIT = .05
TOTAL_ATTEMPT_ERROR = .05


def positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(name + ' must be a positive integer')
    return int(value)


def count_pair(k, n, maximum):
    if (any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in (k, n)) or
            not 0 <= k <= n <= maximum):
        raise ValueError('Invalid initial-cluster adoption/harm counts')
    return int(k), int(n)


def attempt_error(attempt, total_error=TOTAL_ATTEMPT_ERROR):
    attempt = positive_integer(attempt, 'Attempt index')
    if not np.isfinite(total_error) or not 0 < total_error < 1:
        raise ValueError('Invalid total error budget')
    return float(total_error / (attempt * (attempt + 1)))


def select_development_family(results, independent_initials):
    """Select all three fixed models together; return no choice if none qualify.

    Pointwise development upper scores rank candidates only. They do not certify
    any policy and are never reused as calibration or final-test evidence.
    """
    independent_initials = positive_integer(independent_initials, 'Independent initials')
    required = {f'{family}_{seed}' for family in FAMILIES for seed in MODELS}
    if not required <= set(results): raise ValueError('Every candidate family and fixed model is required')
    table = {}
    for family in FAMILIES:
        scores, gains, members = [], [], {}
        for seed in MODELS:
            row = results[f'{family}_{seed}']
            pairs = {target: count_pair(row[f'{target}_harmful_adopted'], row[f'{target}_adopted'], independent_initials)
                     for target in TARGETS}
            rate = pairs['actual'][1] / independent_initials
            gain = float(row['actual_mean_gain_percent'])
            interval = np.asarray(row['actual_mean_gain_pointwise_bootstrap95'], dtype=float)
            if (interval.shape != (2,) or not np.isfinite(interval).all() or interval[0] > interval[1] or
                    not np.isfinite([gain, row['actual_adoption_rate']]).all() or
                    not np.isclose(rate, row['actual_adoption_rate'], atol=1e-14, rtol=0)):
                raise ValueError('Invalid development summary or adoption denominator')
            if pairs['bundle'][1] < pairs['actual'][1] or pairs['bundle'][0] < pairs['actual'][0]:
                raise ValueError('Bundle must include the realized actual-mode events')
            failed = row['failed_runs']
            if isinstance(failed, bool) or not isinstance(failed, Integral) or not 0 <= failed <= 4 * independent_initials:
                raise ValueError('Invalid failure count')
            empirical_acceptable = all(n > 0 and k / n <= RISK_LIMIT for k, n in pairs.values())
            eligible = bool(rate >= ADOPTION_MIN and gain >= GAIN_PERCENT_MIN and interval[0] > 0
                            and failed == 0 and empirical_acceptable)
            upper_scores = {target: float(beta.ppf(.95, k + 1, n - k)) if k < n else 1.
                            for target, (k, n) in pairs.items()}
            scores.extend(upper_scores.values())
            gains.append(gain)
            members[str(seed)] = dict(development_eligible=eligible, actual_adoption_rate=rate,
                mean_gain_percent=gain, pointwise_gain_interval=interval.tolist(), failed_runs=int(failed),
                development_ranking_scores=upper_scores, empirical_risk_screen_passed=empirical_acceptable)
        table[family] = dict(eligible=all(m['development_eligible'] for m in members.values()),
            members=members, worst_development_upper_score=max(scores), minimum_model_gain_percent=min(gains))
    eligible = [name for name in FAMILIES if table[name]['eligible']]
    choice = min(eligible, key=lambda name: (table[name]['worst_development_upper_score'],
                 -table[name]['minimum_model_gain_percent'], FAMILIES.index(name))) if eligible else None
    return dict(selected_family=choice, candidates=table, risk_certified=False,
        selection_rule='All three models pass development usefulness, completion and empirical risk screens; minimize worst six-target development upper score, then maximize minimum model gain, then fixed family order',
        calibration_outcomes_used=False, cross_model_family_splicing=False)


def profile_names(family):
    if family not in FAMILIES: raise ValueError('Unknown predeclared family')
    return tuple(f'{prefix}_{seed}' for seed in MODELS for prefix in ('raw', 'nominal', family))


def validate_observations(names, adopted, harmful, differences, failed):
    names = tuple(names)
    a, h, d, f = map(np.asarray, (adopted, harmful, differences, failed))
    if a.ndim != 3 or a.shape[0] < 1 or a.shape[1:] != (4, len(names)) or len(set(names)) != len(names):
        raise ValueError('One observation cluster per independent initial and all four modes required')
    if any(x.shape != a.shape or x.dtype != np.bool_ for x in (a, h, f)):
        raise ValueError('Boolean adoption, harm and failure arrays must match')
    if d.shape != (*a.shape, 3, 2) or not np.isfinite(d).all():
        raise ValueError('Finite paired per-car cost/time differences required')
    expected_harm = f | np.any(d > 1e-9, axis=(-1, -2))
    if not np.array_equal(expected_harm, h): raise ValueError('Harm must retain every car, metric and physical failure')
    return a, h, d, f


def target_counts(names, adopted, harmful, actual_modes, family):
    names, modes = tuple(names), np.asarray(actual_modes)
    a, h = np.asarray(adopted), np.asarray(harmful)
    if a.ndim != 3 or a.shape[1:] != (4, len(names)) or a.shape != h.shape or a.dtype != np.bool_ or h.dtype != np.bool_:
        raise ValueError('Matching Boolean initial/mode/profile arrays required')
    if modes.shape != (len(a),) or modes.dtype.kind not in 'iu' or np.any((modes < 0) | (modes > 3)):
        raise ValueError('Exactly one preassigned actual mode per initial required')
    if len(set(names)) != len(names): raise ValueError('Duplicate profile')
    counts = {}
    for seed in MODELS:
        name = f'{family}_{seed}'
        if name not in names: raise ValueError('Missing fixed-model family member')
        j = names.index(name)
        actual_a, actual_h = a[np.arange(len(a)), modes, j], h[np.arange(len(a)), modes, j]
        bundle_a, bundle_h = a[:, :, j].any(1), h[:, :, j].any(1)
        counts[f'{seed}:actual'] = (int((actual_a & actual_h).sum()), int(actual_a.sum()))
        counts[f'{seed}:bundle'] = (int((bundle_a & bundle_h).sum()), int(bundle_a.sum()))
    return counts


def calibrate_observations(names, adopted, harmful, differences, failed, *, family, seed_base,
                           attempt, mode_stream=22107, mean_simulations=50000):
    """Use one fixed calibration set for the primary claim and all adapters."""
    if tuple(names) != profile_names(family): raise ValueError('Different common candidate family')
    a, h, d, f = validate_observations(names, adopted, harmful, differences, failed)
    if len(a) != CALIBRATION_N: raise ValueError('The frozen calibration sample size must be complete')
    seed_base = positive_integer(seed_base, 'Calibration seed base')
    ids = list(range(seed_base, seed_base + len(a)))
    modes = np.asarray([np.random.default_rng(np.random.SeedSequence([seed, mode_stream])).integers(4) for seed in ids])
    actual_a, actual_h = a[np.arange(len(a)), modes], h[np.arange(len(a)), modes]
    gains = -d[np.arange(len(a)), modes, :, :, 0].mean(-1)
    counts = target_counts(names, a, h, modes, family)
    primary = family_certificate(counts, RISK_LIMIT, attempt_error(attempt))
    # The comparison procedures have their own predeclared statement. They
    # cannot replace a failed primary claim or select another training seed.
    ltt_actual = conditional_ltt(ids, names, actual_a, actual_h, alpha=.05, delta=.025)
    ltt_bundle = conditional_ltt(ids, names, a.any(1), h.any(1), alpha=.05, delta=.025)
    ltt_eligible = [name for j, name in enumerate(names) if ltt_actual['candidates'][j]['certified']
                    and ltt_bundle['candidates'][j]['certified']]
    mean = paired_mean_improvement_bands(ids, names, gains, delta=.05, simulations=mean_simulations, seed=22113)
    means = {r['candidate']: r['mean_gain'] for r in mean['candidates']}
    mean_eligible = [r['candidate'] for r in mean['candidates'] if r['mean_improvement_selected']]
    _, conformal = conformal_episode_bound(d, d[:, 0], f, names, alpha=.05)
    return dict(primary_family=family, primary_certificate=primary, primary_attempt=int(attempt),
        total_primary_attempt_error=TOTAL_ATTEMPT_ERROR, attempt_error=attempt_error(attempt),
        initial_count=len(a), common_profiles=list(names), common_calibration_budget=True,
        ltt_actual=ltt_actual, ltt_bundle=ltt_bundle,
        ltt_selection=select_fixed_replicates(names, ltt_eligible, means, MODELS),
        mean_spi=mean, mean_spi_selection=select_fixed_replicates(names, mean_eligible, means, MODELS),
        conformal=conformal, final_confirmation_required=True,
        primary_can_be_replaced_by_comparator=False, actual_modes=modes.tolist())


def validate_attempt_history(entries, next_attempt):
    """A new attempt requires every previous calibration attempt to be recorded."""
    next_attempt = positive_integer(next_attempt, 'Next attempt')
    if len(entries) != next_attempt - 1: raise ValueError('Missing prior attempt records')
    for index, record in enumerate(entries, 1):
        if record['attempt'] != index or record['allocated_error'] != attempt_error(index):
            raise ValueError('Attempt order or error allocation changed')
        if record['status'] not in ('rejected', 'completed', 'aborted_after_data_collection'):
            raise ValueError('An active/unrecorded attempt cannot be replaced')
        if not record.get('frozen_protocol_sha256') or not record.get('terminal_evidence_sha256'):
            raise ValueError('Every attempt requires immutable protocol and terminal evidence')
    return dict(attempt=next_attempt, allocated_error=attempt_error(next_attempt),
                cumulative_allocated_error=sum(attempt_error(i) for i in range(1, next_attempt + 1)),
                total_error_budget=TOTAL_ATTEMPT_ERROR)
