"""Joint attempt-2 evidence gate for the fixed recovery-controller family.

The V38 conditional-harm certificate is necessary but not sufficient for a
deployment claim: a controller can have low conditional harm while almost
never being adopted, or while having no useful paired cost gain.  This module
keeps the three frozen actors indivisible and requires every actor to pass:

* mixture and strong-lag conditional harm risk at most five percent;
* mixture adoption probability at least twenty percent;
* mean paired team-cost gain greater than one half percent; and
* zero observed baseline or candidate execution failures.

All inferential components use the already allocated second-attempt level.
Because the scientific alternative is the intersection of every fixed
component, the family test is an intersection-union test: its p-value is the
maximum constituent p-value and no across-component Bonferroni split is used.
Strong-lag risk and IID confirmation adoption use exact binomial tests.
Balanced calibration mixture risk uses simultaneous stratum bounds on joint
harm and adoption; pooled adopted observations are not assumed binomial.
The paired-gain bound is
a one-sided paired Student-t bound; it is exact under IID normal cluster gains
and otherwise an explicitly asymptotic finite-variance statement.
"""
from collections import Counter

import numpy as np
from scipy.stats import beta, binom, t

from .recovery_risk_v38 import (CALIBRATION_N, CONFIRMATION_N, MEMBERS, MODELS,
    RISK_LIMIT, attempt_error)
from .stratified_risk import conditional_risk_certificate


ATTEMPT = 2
ATTEMPT_ALPHA = attempt_error(ATTEMPT)
ADOPTION_MIN = .20
GAIN_MIN_PERCENT = .5
PLANTS = ('nominal', 'gain_drag', 'lag_mild', 'lag_strong')
PLANT_COUNT = CALIBRATION_N // len(PLANTS)
MIN_GAIN_SD = 1e-12
FIXED_INFERENTIAL_MEMBERS = (
    *(f'risk:{name}' for name in MEMBERS),
    *(f'adoption:{model}' for model in MODELS),
    *(f'gain:{model}' for model in MODELS),
)


def _fixed_models(model_ids):
    values = tuple(model_ids)
    if values != MODELS:
        raise ValueError('The ordered three-model family must remain exactly fixed')
    return values


def _boolean_matrix(value, rows, name):
    array = np.asarray(value)
    if array.shape != (rows, len(MODELS)) or array.dtype != np.bool_:
        raise ValueError(f'{name} must be a complete Boolean initial/model matrix')
    return array


def _boolean_vector(value, rows, name):
    array = np.asarray(value)
    if array.shape != (rows,) or array.dtype != np.bool_:
        raise ValueError(f'{name} must be a complete Boolean initial vector')
    return array


def _balanced_plants(plants):
    values = np.asarray(plants)
    if values.shape != (CALIBRATION_N,) or values.dtype.kind not in 'USO':
        raise ValueError('The complete fixed calibration plant schedule is required')
    counts = Counter(values.tolist())
    expected = {name: PLANT_COUNT for name in PLANTS}
    if counts != expected:
        raise ValueError('Calibration must retain the predeclared balanced plant mixture')
    return values


def adoption_certificate(adopted, *, model_ids=MODELS):
    """Exact lower bounds; confirmation rows must be IID mixture draws.

    Unlike calibration, confirmation must NOT impose fixed per-plant counts.
    """
    _fixed_models(model_ids)
    array = _boolean_matrix(adopted, CONFIRMATION_N, 'Confirmation adoption')
    members = {}
    for column, model in enumerate(MODELS):
        successes = int(array[:, column].sum())
        p_value = float(binom.sf(successes - 1, CONFIRMATION_N, ADOPTION_MIN))
        lower = (float(beta.ppf(ATTEMPT_ALPHA, successes,
                                CONFIRMATION_N - successes + 1))
                 if successes else 0.)
        members[str(model)] = dict(adoptions=successes, initials=CONFIRMATION_N,
            observed_rate=successes / CONFIRMATION_N,
            pointwise_p_value=p_value, adoption_lower=lower,
            certified=bool(p_value <= ATTEMPT_ALPHA and lower >= ADOPTION_MIN))
    family_p = max(item['pointwise_p_value'] for item in members.values())
    minimum_lower = min(item['adoption_lower'] for item in members.values())
    return dict(members=members, threshold=ADOPTION_MIN, attempt=ATTEMPT,
        allocated_error=ATTEMPT_ALPHA, family_p_value=family_p,
        minimum_adoption_lower=minimum_lower,
        all_fixed_models_adoption_certified=bool(
            family_p <= ATTEMPT_ALPHA and minimum_lower >= ADOPTION_MIN),
        exact_binomial=True, intersection_union_test=True,
        constituent_bonferroni_used=False)


def _risk_certificate(adopted, harmful, failed, plants):
    a = _boolean_matrix(adopted, CALIBRATION_N, 'Calibration adoption')
    h = _boolean_matrix(harmful, CALIBRATION_N, 'Calibration harm')
    if np.any(failed & ~h):
        raise ValueError('Every controller failure must be retained as harm')
    members = {}
    for column, model in enumerate(MODELS):
        masks = [plants == plant for plant in PLANTS]
        n = [int(mask.sum()) for mask in masks]
        adopted_counts = [int((mask & a[:, column]).sum()) for mask in masks]
        harmful_counts = [int((mask & a[:, column] & h[:, column]).sum())
                          for mask in masks]
        mixture = conditional_risk_certificate(harmful_counts, adopted_counts,
            n, [.25] * len(PLANTS), alpha=ATTEMPT_ALPHA, risk_limit=RISK_LIMIT)
        members[f'{model}:mixture'] = dict(
            harmful_adoptions=sum(harmful_counts), adoptions=sum(adopted_counts),
            pointwise_p_value=mixture['pointwise_p_value'],
            pointwise_risk_upper=mixture['risk_upper'],
            conditional_risk_observed=bool(sum(adopted_counts)),
            method='stratified_joint_harm_over_adoption', stratified=mixture)
        k, count = harmful_counts[-1], adopted_counts[-1]
        p_value = float(binom.cdf(k, count, RISK_LIMIT)) if count else 1.
        upper = float(beta.isf(ATTEMPT_ALPHA, k + 1, count - k)) if k < count else 1.
        members[f'{model}:strong_lag'] = dict(harmful_adoptions=k, adoptions=count,
            pointwise_p_value=p_value, pointwise_risk_upper=upper,
            conditional_risk_observed=bool(count), method='within_stratum_exact_binomial')
    family_p = max(item['pointwise_p_value'] for item in members.values())
    upper = max(item['pointwise_risk_upper'] for item in members.values())
    return dict(members=members, family_p_value=family_p,
        maximum_conditional_risk_upper=upper, allocated_error=ATTEMPT_ALPHA,
        risk_limit=RISK_LIMIT, attempt=ATTEMPT,
        all_fixed_models_both_targets_certified=bool(
            family_p <= ATTEMPT_ALPHA and upper <= RISK_LIMIT),
        intersection_union_test=True, constituent_bonferroni_used=False,
        within_mixture_bound_bonferroni_used=True,
        calibration_iid_within_each_plant_required=True,
        statewise_probability_certified=False, out_of_distribution_risk_certified=False)


def paired_gain_certificate(paired_gain_percent, *, model_ids=MODELS):
    """One-sided paired-t lower bounds for fixed-model mean percentage gain.

    Each row must be one independent confirmation initial and each value must
    already be 100 * (baseline cost - candidate cost) / baseline cost
    for that same row, with a strictly positive baseline cost.
    A zero empirical variance is deliberately untestable rather than treated as
    infinite evidence.
    """
    _fixed_models(model_ids)
    gains = np.asarray(paired_gain_percent, dtype=float)
    if gains.shape != (CONFIRMATION_N, len(MODELS)) or not np.isfinite(gains).all():
        raise ValueError('Finite paired gains for every confirmation initial/model are required')
    critical = float(t.ppf(1. - ATTEMPT_ALPHA, CONFIRMATION_N - 1))
    members = {}
    for column, model in enumerate(MODELS):
        values = gains[:, column]
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        testable = bool(sd > MIN_GAIN_SD)
        if testable:
            standard_error = sd / np.sqrt(CONFIRMATION_N)
            statistic = (mean - GAIN_MIN_PERCENT) / standard_error
            p_value = float(t.sf(statistic, CONFIRMATION_N - 1))
            lower = float(mean - critical * standard_error)
        else:
            standard_error, p_value, lower = 0., 1., None
        members[str(model)] = dict(mean_gain_percent=mean, standard_deviation=sd,
            standard_error=float(standard_error), pointwise_p_value=p_value,
            gain_lower_percent=lower, testable=testable,
            certified=bool(testable and p_value <= ATTEMPT_ALPHA
                           and lower is not None and lower > GAIN_MIN_PERCENT))
    family_p = max(item['pointwise_p_value'] for item in members.values())
    finite_lowers = [item['gain_lower_percent'] for item in members.values()
                     if item['gain_lower_percent'] is not None]
    minimum_lower = min(finite_lowers) if len(finite_lowers) == len(MODELS) else None
    all_certified = all(item['certified'] for item in members.values())
    return dict(members=members, threshold_percent=GAIN_MIN_PERCENT,
        attempt=ATTEMPT, allocated_error=ATTEMPT_ALPHA, critical=critical,
        family_p_value=family_p, minimum_gain_lower_percent=minimum_lower,
        all_fixed_models_gain_certified=bool(all_certified),
        paired_by_independent_initial=True, exact_finite_sample=False,
        normal_iid_exact_otherwise_asymptotic=True,
        intersection_union_test=True, constituent_bonferroni_used=False)


def joint_deployment_certificate(calibration_adopted, calibration_harmful,
                                 calibration_failed, calibration_plants,
                                 confirmation_adopted, paired_gain_percent,
                                 confirmation_failed, *,
                                 calibration_baseline_failed,
                                 confirmation_baseline_failed,
                                 model_ids=MODELS):
    """Require risk, usefulness, adoption and complete execution jointly.

    Calibration and confirmation are separate frozen samples.  Confirmation
    cannot select, remove or reorder a model.  Observing no execution failures
    is an engineering acceptance condition, not a claim that population failure
    probability is mathematically zero.
    """
    _fixed_models(model_ids)
    plants = _balanced_plants(calibration_plants)
    calibration_failures = _boolean_matrix(
        calibration_failed, CALIBRATION_N, 'Calibration candidate failure')
    calibration_baseline = _boolean_vector(
        calibration_baseline_failed, CALIBRATION_N, 'Calibration baseline failure')
    confirmation_failures = _boolean_matrix(
        confirmation_failed, CONFIRMATION_N, 'Confirmation candidate failure')
    confirmation_baseline = _boolean_vector(
        confirmation_baseline_failed, CONFIRMATION_N, 'Confirmation baseline failure')

    risk = _risk_certificate(calibration_adopted,
        calibration_harmful, calibration_failures, plants)
    adoption = adoption_certificate(confirmation_adopted, model_ids=model_ids)
    gain = paired_gain_certificate(paired_gain_percent, model_ids=model_ids)

    p_values = {
        **{f'risk:{name}': item['pointwise_p_value']
           for name, item in risk['members'].items()},
        **{f'adoption:{name}': item['pointwise_p_value']
           for name, item in adoption['members'].items()},
        **{f'gain:{name}': item['pointwise_p_value']
           for name, item in gain['members'].items()},
    }
    if set(p_values) != set(FIXED_INFERENTIAL_MEMBERS):
        raise RuntimeError('The fixed joint inferential family changed')
    family_p = max(p_values.values())
    statistical = bool(
        risk['all_fixed_models_both_targets_certified']
        and adoption['all_fixed_models_adoption_certified']
        and gain['all_fixed_models_gain_certified']
        and family_p <= ATTEMPT_ALPHA)
    failure_count = int(calibration_failures.sum() + calibration_baseline.sum()
                        + confirmation_failures.sum() + confirmation_baseline.sum())
    zero_execution_failures = failure_count == 0
    return dict(attempt=ATTEMPT, allocated_error=ATTEMPT_ALPHA,
        total_attempt_error=.05, fixed_models=list(MODELS),
        fixed_inferential_members=list(FIXED_INFERENTIAL_MEMBERS),
        calibration_initials=CALIBRATION_N, confirmation_initials=CONFIRMATION_N,
        primary_risk=risk, adoption=adoption, paired_gain=gain,
        constituent_p_values=p_values, joint_family_p_value=family_p,
        all_statistical_components_certified=statistical,
        observed_execution_failures=failure_count,
        all_baseline_and_candidate_executions_complete=zero_execution_failures,
        joint_deployment_certified=bool(statistical and zero_execution_failures),
        intersection_union_test=True, constituent_bonferroni_used=False,
        controller_family_selection_permitted=False,
        confirmation_outcomes_used_for_selection=False,
        confirmation_iid_mixture_sampling_required=True,
        zero_observed_failures_is_not_zero_population_failure_risk=True)
