"""Failure-preserving analysis revision for the unstarted eight-selector study.

Controller decisions, samples, confidence levels and numerical bounds are the
frozen V39 definitions. Failed references are retained but cannot establish a
performance improvement; pre-adoption failures must not be rejected as drift.
"""
from dataclasses import asdict
from numbers import Integral

import numpy as np
from scipy.stats import t

from .fair_comparison_v39 import (METHODS, CANDIDATE_METHODS, DESCRIPTORS,
    EpisodeRecord, ProtocolSpec, fingerprint, _record_arrays, _exact_lower, _exact_upper)


def analysis_contract():
    return dict(version='v44-failure-preserving-fair-analysis',
        pre_adoption_failure='retain all costs and failure flags; exclude only from adoption denominator',
        zero_adoption_equality='require equality only when candidate and baseline both succeed',
        failed_baseline='retain capped costs; conservatively mark harm; relative improvement unassessable',
        ranking='unchanged risk/gain/adoption order among useful methods with zero method and baseline failures',
        timeout='any recorded evidence timeout remains a failure, even if baseline fallback finishes',
        uncertainty='unchanged V39 allocation and bounds; no new risk certificate',
        no_rows_deleted=True, no_controller_or_schedule_changes=True)


def failed(record):
    return bool(not record.completed or record.timed_out or record.collision
                or record.aborted or record.contract_violations)


def _group(records, spec):
    if not isinstance(spec, ProtocolSpec):
        raise ValueError('A fully hash-bound pre-registered protocol is required')
    records = tuple(records)
    if not records:
        raise ValueError('A complete matched record grid is required')
    by_case, identities = {}, {}
    for record in records:
        if not isinstance(record, EpisodeRecord) or record.method not in METHODS:
            raise ValueError('Only pre-registered episode records are allowed')
        if (record.protocol_sha256 != fingerprint(spec) or record.cost_contract != spec.cost_contract
                or not np.isclose(record.timeout_ms, spec.budget.decision_timeout_ms, atol=0., rtol=0.)):
            raise ValueError('Every method must share the frozen protocol, timeout and cost')
        if any(not isinstance(getattr(record, name), (bool, np.bool_))
               for name in ('adopted', 'completed', 'timed_out', 'collision', 'aborted')):
            raise ValueError('Adoption, completion and failure flags must be Boolean')
        if (isinstance(record.contract_violations, (bool, np.bool_))
                or not isinstance(record.contract_violations, Integral) or record.contract_violations < 0):
            raise ValueError('Contract violations must be a nonnegative integer')
        _record_arrays(record)
        if identities.setdefault(record.case.case_id, record.case) != record.case:
            raise ValueError('One case ID cannot change scene, initial, dynamics or disturbance')
        methods = by_case.setdefault(record.case, {})
        if record.method in methods:
            raise ValueError('Repeated method/case record')
        methods[record.method] = record
    if any(set(methods) != set(METHODS) for methods in by_case.values()):
        raise ValueError('Every case must contain the complete comparator family exactly once')
    return by_case


def summarize_matched_records(records, spec):
    by_case = _group(records, spec)
    clusters = {}
    counts = {m: dict(failed_cases=0, baseline_failed_cases=0, failed_before_adoption_cases=0,
        timeout_cases=0, collision_cases=0, aborted_cases=0, incomplete_cases=0,
        contract_violation_cases=0) for m in METHODS}
    for case, methods in by_case.items():
        baseline = methods['always_baseline']
        if baseline.adopted:
            raise ValueError('Always-baseline cannot contain a learning adoption')
        base_cost, base_arrival = _record_arrays(baseline)
        base_failed = failed(baseline)
        values = {}
        for method in METHODS:
            record = methods[method]
            cost, arrival = _record_arrays(record)
            failure = failed(record)
            if (method != 'always_baseline' and not record.adopted and not failure and not base_failed
                    and (not np.allclose(cost, base_cost, atol=1e-9, rtol=0)
                         or not np.allclose(arrival, base_arrival, atol=1e-9, rtol=0))):
                raise ValueError('A successful zero-adoption comparator must equal its paired baseline')
            harm = bool(base_failed or failure or np.any(cost > base_cost+1e-9)
                        or np.any(arrival > base_arrival+1e-9))
            values[method] = dict(adopted=bool(record.adopted), harmful=harm, failed=failure,
                baseline_failed=base_failed, failed_before_adoption=failure and not record.adopted,
                gain=float(baseline.team_cost-record.team_cost), baseline_cost=float(baseline.team_cost))
            updates = dict(failed_cases=failure, baseline_failed_cases=base_failed,
                failed_before_adoption_cases=failure and not record.adopted,
                timeout_cases=record.timed_out, collision_cases=record.collision,
                aborted_cases=record.aborted, incomplete_cases=not record.completed,
                contract_violation_cases=record.contract_violations > 0)
            for name, value in updates.items():
                counts[method][name] += int(value)
        clusters.setdefault(case.independent_cluster, []).append(values)

    risk_error = spec.family_error*spec.risk_error_share/len(CANDIDATE_METHODS)
    adoption_error = spec.family_error*spec.adoption_error_share/len(CANDIDATE_METHODS)
    gain_error = spec.family_error*spec.gain_error_share/len(CANDIDATE_METHODS)
    summaries = {}
    for method in METHODS:
        rows = []
        for cluster in clusters.values():
            selected = [v[method] for v in cluster]
            rows.append({**{key: any(r[key] for r in selected) for key in
                ('adopted', 'harmful', 'failed', 'baseline_failed', 'failed_before_adoption')},
                'gain': float(np.mean([r['gain'] for r in selected])),
                'baseline_cost': float(np.mean([r['baseline_cost'] for r in selected]))})
        adopted = np.array([r['adopted'] for r in rows], dtype=bool)
        harmful = np.array([r['harmful'] for r in rows], dtype=bool)
        gains = np.array([r['gain'] for r in rows])
        baseline_costs = np.array([r['baseline_cost'] for r in rows])
        if np.any(baseline_costs <= 0.):
            raise ValueError('Positive capped baseline cost is required for percentage gain')
        n, a, h = len(rows), int(adopted.sum()), int((adopted & harmful).sum())
        percentages = 100.*gains/baseline_costs
        sd = float(percentages.std(ddof=1)) if n > 1 else 0.
        critical = float(t.ppf(1.-gain_error, n-1)) if n > 1 else None
        lower = None if n < 2 or sd <= 1e-12 else float(percentages.mean()-critical*sd/np.sqrt(n))
        adoption_lower, risk_upper = _exact_lower(a, n, adoption_error), _exact_upper(h, a, risk_error)
        useful = bool(method != 'always_baseline' and adoption_lower >= spec.adoption_minimum
                      and lower is not None and lower >= spec.gain_minimum_percent)
        base_failures = sum(r['baseline_failed'] for r in rows)
        failures = sum(r['failed'] for r in rows)
        summaries[method] = dict(independent_clusters=n, adopted_clusters=a,
            harmful_adopted_clusters=h, adoption_rate=float(a/n),
            simultaneous_adoption_lower=adoption_lower,
            empirical_conditional_harm=float(h/a) if a else None,
            simultaneous_conditional_harm_upper=risk_upper,
            mean_paired_team_gain_s=float(gains.mean()),
            mean_paired_cluster_gain_percent=float(percentages.mean()),
            ratio_of_means_gain_percent=float(100.*gains.mean()/baseline_costs.mean()),
            paired_percent_standard_deviation=sd, paired_percent_t_critical=critical,
            simultaneous_paired_t_gain_lower_percent=lower,
            usefulness_constraints_passed=useful, descriptor=asdict(DESCRIPTORS[method]),
            **counts[method], failed_clusters=failures, baseline_failed_clusters=base_failures,
            failed_before_adoption_clusters=sum(r['failed_before_adoption'] for r in rows),
            joint_harm_adoption_rate=float(h/n), failure_rate=float(failures/n),
            relative_improvement_assessable=base_failures == 0,
            comparison_eligible=bool(useful and not failures and not base_failures))
    eligible = [m for m in CANDIDATE_METHODS if summaries[m]['comparison_eligible']]
    winner = min(eligible, key=lambda m:(summaries[m]['simultaneous_conditional_harm_upper'],
        -summaries[m]['simultaneous_paired_t_gain_lower_percent'],
        -summaries[m]['simultaneous_adoption_lower'], METHODS.index(m))) if eligible else None
    return dict(protocol_sha256=fingerprint(spec), analysis_contract=analysis_contract(),
        primary_metric=spec.primary_metric, family_error=spec.family_error,
        error_allocation=dict(risk_family=spec.family_error*spec.risk_error_share,
            adoption_family=spec.family_error*spec.adoption_error_share,
            gain_family=spec.family_error*spec.gain_error_share,
            candidate_methods=len(CANDIDATE_METHODS), risk_per_method=risk_error,
            adoption_per_method=adoption_error, gain_per_method=gain_error),
        independent_clusters=len(clusters), cases=len(by_case), methods=summaries,
        adoption_constraint=spec.adoption_minimum, gain_constraint_percent=spec.gain_minimum_percent,
        always_baseline_eligible=False, selected_method=winner,
        exact_bounds=('conditional harm and adoption use exact one-sided binomial bounds; '
            'gain uses a Bonferroni one-sided paired t bound on per-cluster percentage '
            'gains, exact under IID normal percentage gains and otherwise diagnostic'),
        full_reproduction_and_adapters_explicit=True, deleted_records=0,
        zero_observed_failures_is_not_zero_population_failure_risk=True)
