"""Deploy frozen complete profiles using calibration and nominal forecasts only.

Actual episode IDs, disturbance labels and realized outcomes are deliberately
absent from this interface. The selected controller is fixed for the episode.
"""
import numpy as np

from .comparison_v19 import conformal_profile
from .joint_admission_v15 import JointBound


def validate_calibration(plan, calibration):
    names, seeds = plan['profiles'], plan['seeds']
    if calibration['names'] != names or calibration['models'] != plan['models']:
        raise ValueError('Calibration/profile identity mismatch')
    for rule in ('ltt_selection', 'mean_spi_selection'):
        selected = calibration[rule]
        if set(selected) != set(seeds):
            raise ValueError('Every fixed training replicate must be retained')
        for seed, name in selected.items():
            if name != 'baseline' and (name not in names or not name.endswith('_' + seed)):
                raise ValueError('A selector cannot cross training replicates')


def profile_metrics(row):
    finish = np.asarray(row['per_vehicle_finish'], dtype=float)
    variation = np.asarray(row['per_vehicle_variation'], dtype=float)
    if finish.shape != (3,) or variation.shape != (3,):
        raise ValueError('Three-vehicle complete-profile metrics required')
    result = np.column_stack((finish + .05 * variation, finish))
    if not np.isfinite(result).all():
        raise ValueError('Finite capped complete-profile forecasts required')
    return result


def physical_failure(row):
    return bool(not row['completed'] or row['collision'] or row['aborted'] or row['contract_violations'])


def select_deployment_profile(adapter, training_seed, plan, calibration, nominal_forecasts=None):
    """Return a whole-profile name, never an assembled vehicle trajectory."""
    validate_calibration(plan, calibration)
    seed = str(training_seed)
    if seed not in plan['seeds']:
        raise ValueError('Unknown fixed training replicate')
    names = plan['profiles']
    if adapter in ('raw', 'nominal', 'portfolio0'):
        name = adapter + '_' + seed
        if name not in names:
            raise ValueError('Missing frozen complete profile')
        return name
    if adapter in ('ltt', 'mean_spi'):
        return calibration[adapter + '_selection'][seed]
    if adapter != 'conformal':
        raise ValueError('Unknown frozen deployment adapter')
    if nominal_forecasts is None or set(nominal_forecasts) != {'baseline', *names}:
        raise ValueError('Conformal deployment requires the full nominal forecast family')
    baseline = nominal_forecasts['baseline']
    if not baseline['completed']:
        return 'baseline'
    prediction = np.stack([profile_metrics(nominal_forecasts[name]) - profile_metrics(baseline) for name in names])
    failed = np.array([physical_failure(nominal_forecasts[name]) for name in names], dtype=bool)
    q = np.inf if calibration['conformal']['infinite_q'] else calibration['conformal']['q']
    bound = JointBound(tuple(names), ((1., 1.),) * 3, q,
                       calibration['calibration_n_initials'], plan['conformal']['alpha'])
    return conformal_profile(bound, prediction, names, seed, failed)
