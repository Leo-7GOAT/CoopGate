"""Complete-trajectory accounting from same-prefix baseline advantages.

The telescoping identity is an algebraic consistency check, not a new learning
algorithm or a probabilistic certificate. Incomplete branches remain explicit.
"""
import numpy as np


def failed(row):
    return bool(not row['completed'] or row['aborted'] or row['collision'] or row['contract_violations'])


def audit_identity(source, baseline, admissions):
    steps = [a['step'] for a in admissions]
    if (len(steps) != source['learned_action_steps'] or len(set(steps)) != len(steps)
            or any(type(k) is not int or not 0 <= k < source['control_steps'] for k in steps)):
        raise ValueError('Every recorded admission is required exactly once')
    observed = np.column_stack((source['per_vehicle_cost'],source['per_vehicle_finish']))-np.column_stack(
        (baseline['per_vehicle_cost'],baseline['per_vehicle_finish']))
    if observed.shape != (3,2) or not np.isfinite(observed).all():
        raise ValueError('Complete finite original per-vehicle outcomes required')
    reasons = []
    if failed(source): reasons.append('source_episode_incomplete_or_unsafe')
    if failed(baseline): reasons.append('initial_baseline_incomplete_or_unsafe')
    if any(not a['pair']['both_complete'] for a in admissions): reasons.append('local_continuation_incomplete')
    if reasons:
        return dict(assessable=False,reasons=reasons,admissions=len(admissions),
            observed_capped_difference=observed.tolist(),identity_matches=None,
            interpretation='Incomplete costs are retained but do not authorize a performance identity or improvement claim.')
    local = [np.asarray(a['pair']['difference'],dtype=float) for a in admissions]
    if any(a.shape != (3,2) or not np.isfinite(a).all() for a in local):
        raise ValueError('Every complete local vector must contain three costs and three times')
    total = sum(local,np.zeros((3,2)))
    positive = sum((np.maximum(a,0.) for a in local),np.zeros((3,2)))
    negative = sum((np.minimum(a,0.) for a in local),np.zeros((3,2)))
    error = float(np.max(np.abs(total-observed)))
    return dict(assessable=True,reasons=[],admissions=len(admissions),
        observed_difference=observed.tolist(),summed_local_difference=total.tolist(),
        positive_local_contributions=positive.tolist(),negative_local_contributions=negative.tolist(),
        max_absolute_error=error,identity_matches=error <= 1e-8,
        all_local_nonworsening=bool(all(np.all(a <= 1e-9) for a in local)),
        episode_nonworsening=bool(np.all(observed <= 1e-9)),
        episode_mean_cost_difference=float(observed[:,0].mean()))
