"""Complete-profile prediction and adapter choices from current observations.

The predictor accepts an observed state, fixed controller protocol and family.
It has no input for the real episode seed, scenario, noise mode or future trace.
Calibration identities and whole-profile choices remain frozen across deployment.
"""
import time

import numpy as np

from coopgate.comparison_v19 import conformal_profile
from coopgate.joint_admission_v15 import JointBound
from coopgate.observable_initial_v19 import ObservableInitialMergeEnv
from coopgate.prospective_validation_v21 import MODELS, profile_names
from experiment_ego_v13 import run_episode
from study_matched_transfer_v21 import factories


def observed_state(value):
    state = np.asarray(value, dtype=float)
    if state.shape != (3, 2) or not np.isfinite(state).all():
        raise ValueError('Finite observed three-vehicle positions and speeds required')
    return state.copy()


def metrics(row):
    finish, variation, cost = [np.asarray(row[key], dtype=float) for key in
                              ('per_vehicle_finish', 'per_vehicle_variation', 'per_vehicle_cost')]
    if (any(v.shape != (3,) or not np.isfinite(v).all() for v in (finish, variation, cost))
            or np.any(finish < 0) or np.any(variation < 0)
            or not np.allclose(cost, finish + .05 * variation, atol=1e-10, rtol=0)):
        raise ValueError('Prediction must preserve finite per-car registered metrics, including failures')
    return np.column_stack((cost, finish))


def failed(row):
    return bool(not row['completed'] or row['collision'] or row['aborted'] or row['contract_violations'])


def prediction_from_rows(initial, family, rows):
    state = observed_state(initial)
    names = profile_names(family)
    if set(rows) != {'baseline', *names}:
        raise ValueError('Complete common prediction family required')
    for name, row in rows.items():
        if (row['controller'] != name or row['seed'] != 1 or row['scenario'] != 'unused'
                or row['disturbance'] != 'clean'):
            raise ValueError('Only fixed-engine observable-initial predictions may enter this interface')
        np.testing.assert_allclose(observed_state(row['initial']), state, atol=1e-8, rtol=0)
    base = metrics(rows['baseline'])
    return dict(family=family, names=list(names), observed_initial=state.tolist(),
        differences=[(metrics(rows[name])-base).tolist() for name in names],
        failed=[failed(rows[name]) for name in names], baseline_completed=bool(rows['baseline']['completed']),
        baseline_metrics=base.tolist(), baseline_failed=failed(rows['baseline']),
        inputs='Observed state and frozen controllers; fixed nominal internal engine only')


def forecast(initial, protocol, family, on_record=None):
    """Run nominal copies from observations; persist each role before analysis.

    on_record(name, row, trace) may atomically retain completed predictions.
    Full wall time includes simulator setup and all ten predictions; controller
    and setup timings in the records are additionally retained unchanged.
    """
    state = observed_state(initial)
    names = profile_names(family)
    if set(protocol['spec']['methods']) != {'baseline', *names}:
        raise ValueError('Predictor protocol must contain exactly the shared ten controllers')
    started = time.perf_counter()
    fs = factories(protocol)
    rows = {}
    with ObservableInitialMergeEnv(state) as env:
        for name in ('baseline', *names):
            row, trace, observed = run_episode(env, 1, 'unused', 'clean', fs[name], True)
            row.update(controller=name, initial=observed.tolist())
            if on_record is not None:
                on_record(name, row, trace)
            rows[name] = row
    result = prediction_from_rows(state, family, rows)
    result.update(full_prediction_wall_ms=(time.perf_counter()-started)*1000., actual_prediction_runs=len(rows))
    return result


def choose_profiles(prediction, calibration, family):
    """Match all six frozen adapters without reading realized outcomes."""
    names = profile_names(family)
    if (prediction['family'] != family or tuple(prediction['names']) != names
            or calibration['primary_family'] != family or tuple(calibration['common_profiles']) != names):
        raise ValueError('Prediction and calibration must use the same complete family')
    observed_state(prediction['observed_initial'])
    differences = np.asarray(prediction['differences'], dtype=float)
    failures = np.asarray(prediction['failed'])
    if (differences.shape != (len(names), 3, 2) or not np.isfinite(differences).all()
            or failures.shape != (len(names),) or failures.dtype != np.bool_):
        raise ValueError('Finite whole-profile predictions and Boolean failure flags required')
    seeds = tuple(map(str, MODELS))
    for selector in ('ltt_selection', 'mean_spi_selection'):
        choices = calibration[selector]
        if set(choices) != set(seeds):
            raise ValueError('Every fixed training model requires a calibrated choice')
        for seed, name in choices.items():
            if name != 'baseline' and (name not in names or not name.endswith('_'+seed)):
                raise ValueError('Calibrated choice cannot cross models or introduce profiles')
    conformal = calibration['conformal']
    q = np.inf if conformal['infinite_q'] else float(conformal['q'])
    bound = JointBound(names, ((1., 1.),)*3, q, conformal['calibration_initials'], .05)
    output = {}
    for method in ('raw', 'nominal', family, 'conformal', 'ltt', 'mean_spi'):
        output[method] = {}
        for seed in seeds:
            if method == 'conformal':
                choice = (conformal_profile(bound, differences, names, seed, failures)
                          if prediction['baseline_completed'] else 'baseline')
            elif method in ('ltt', 'mean_spi'):
                choice = calibration[method+'_selection'][seed]
            else:
                choice = method+'_'+seed
            output[method][seed] = choice
    return output


def verify_clean_identity(prediction, clean_rows):
    """Audit only: check predicted metrics against independent clean execution.

    Realized rows are deliberately confined to this verifier, never the selector.
    """
    names = tuple(prediction['names'])
    if set(clean_rows) != {'baseline', *names}:
        raise ValueError('Complete independent clean reference required')
    for name, row in clean_rows.items():
        if row['controller'] != name or row['disturbance'] != 'clean':
            raise ValueError('Different reference profile or mode')
        np.testing.assert_allclose(row['initial'], prediction['observed_initial'], atol=1e-8, rtol=0)
    base = metrics(clean_rows['baseline'])
    np.testing.assert_allclose(prediction['baseline_metrics'], base, atol=1e-10, rtol=0)
    np.testing.assert_allclose(prediction['differences'], [metrics(clean_rows[n])-base for n in names], atol=1e-10, rtol=0)
    np.testing.assert_equal(prediction['failed'], [failed(clean_rows[n]) for n in names])
    np.testing.assert_equal(prediction['baseline_completed'], bool(clean_rows['baseline']['completed']))
    np.testing.assert_equal(prediction['baseline_failed'], failed(clean_rows['baseline']))
