"""Convert COMPLETE paired episode records to the unchanged V39 certificate.

No finite-bank margin is interpreted as a probability. Baseline failures never
remove a sample: they conservatively impute harm and block deployment.
"""
import numpy as np
from .recovery_risk_v39 import MODELS, joint_deployment_certificate

METHODS = ('baseline', *(str(model) for model in MODELS))


def episode_outcome(row):
    costs = np.asarray(row['per_vehicle_cost'], dtype=float)
    arrivals = np.asarray(row['per_vehicle_finish'], dtype=float)
    if (costs.shape != (3,) or arrivals.shape != (3,) or not np.isfinite(costs).all()
            or not np.isfinite(arrivals).all() or np.any(costs < 0) or np.any(arrivals < 0)
            or np.any(arrivals > 25.+1e-9) or not np.isfinite(row['team_cost'])
            or abs(float(costs.mean())-row['team_cost']) > 1e-9 or row['team_cost'] <= 0):
        raise ValueError('Finite positive paired capped cost/arrival accounting required')
    if type(row['learned_action_steps']) is not int or not 0 <= row['learned_action_steps'] <= 50:
        raise ValueError('Invalid actual adoption count')
    failed = bool(not row['completed'] or row['aborted'] or row['collision'] or row['contract_violations'])
    return dict(adopted=row['learned_action_steps'] > 0, failed=failed, costs=costs,
                arrivals=arrivals, team_cost=float(row['team_cost']))


def paired_cluster(rows):
    if set(rows) != set(METHODS):
        raise ValueError('One baseline and all three fixed models are required')
    base = episode_outcome(rows['baseline'])
    if base['adopted']:
        raise ValueError('Baseline cannot adopt a learned command')
    result = dict(baseline_failed=base['failed'], adopted=[], harmful=[], failed=[],
                  gain_percent=[], baseline_unassessable=[])
    for model in METHODS[1:]:
        actual = episode_outcome(rows[model])
        # A candidate can fail during planning before adopting a command. That
        # is a real failure outcome, not an excuse to reject/drop its sample.
        if not actual['adopted'] and not actual['failed'] and not base['failed']:
            if (not np.allclose(actual['costs'], base['costs'], atol=1e-9, rtol=0)
                    or not np.allclose(actual['arrivals'], base['arrivals'], atol=1e-9, rtol=0)
                    or any(rows[model][key] != rows['baseline'][key] for key in
                           ('completed', 'aborted', 'collision', 'contract_violations'))):
                raise ValueError('Zero adoption differs from the paired baseline')
        harmful = bool(actual['failed'] or base['failed']
            or np.any(actual['costs'] > base['costs']+1e-9)
            or np.any(actual['arrivals'] > base['arrivals']+1e-9))
        result['adopted'].append(actual['adopted'])
        result['harmful'].append(harmful)
        result['failed'].append(actual['failed'])
        result['gain_percent'].append(100*(base['team_cost']-actual['team_cost'])/base['team_cost'])
        result['baseline_unassessable'].append(base['failed'])
    return result


def stage_arrays(cases, records):
    """Reject missing/duplicate/foreign rows before examining paired effects."""
    expected = {(c['cluster_id'], method) for c in cases for method in METHODS}
    seen, rows = set(), {}
    for record in records:
        key = (record['cluster_id'], record['method'])
        if key not in expected or key in seen:
            raise ValueError('Duplicate or foreign episode')
        seen.add(key)
        rows[key] = record['row']
    if seen != expected or len(expected) != len(cases)*4:
        raise ValueError('Every independent cluster and fixed model must be complete')
    outcomes = []
    for case in cases:
        selected = {method: rows[case['cluster_id'], method] for method in METHODS}
        for method, row in selected.items():
            if (row['seed'] != case['seed'] or row['scenario'] != case['scenario']
                    or row['disturbance'] != case['disturbance'] or row['plant_name'] != case['plant']
                    or row['model'] != ('none' if method == 'baseline' else method)
                    or row['rule'] != ('baseline' if method == 'baseline' else 'bank_performance')):
                raise ValueError('Recorded pairing identity changed')
        outcomes.append(paired_cluster(selected))
    return {name: np.asarray([o[name] for o in outcomes], dtype=float if name == 'gain_percent' else bool)
            for name in ('baseline_failed', 'adopted', 'harmful', 'failed', 'gain_percent', 'baseline_unassessable')}


def certificate_from_complete_records(plan, calibration_records, confirmation_records):
    """Caller must first verify native terminal receipts, seals and full replay."""
    calibration = stage_arrays(plan['stages']['calibration'], calibration_records)
    confirmation = stage_arrays(plan['stages']['confirmation'], confirmation_records)
    result = joint_deployment_certificate(
        calibration['adopted'], calibration['harmful'], calibration['failed'],
        [c['plant'] for c in plan['stages']['calibration']],
        confirmation['adopted'], confirmation['gain_percent'], confirmation['failed'],
        calibration_baseline_failed=calibration['baseline_failed'],
        confirmation_baseline_failed=confirmation['baseline_failed'])
    result.update(baseline_failed_pairs_imputed_harm=int(calibration['baseline_unassessable'].sum()
        + confirmation['baseline_unassessable'].sum()), baseline_failure_samples_removed=0,
        gain_retains_capped_failure_costs=True)
    return result
