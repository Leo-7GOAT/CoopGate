"""Outcome-free power sensitivity for the unchanged, unstarted V42 design.

Risk power is Monte Carlo under explicit hypothetical Bernoulli populations.
Adoption power is exact binomial; gain power assumes IID normal paired gains.
The all-model bound uses a union bound, never model/endpoint independence.
No formal outcome, checkpoint, partial training output, or episode DB is read.
"""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.stats import beta, binom, nct, t

from coopgate.recovery_risk_v39 import (
    ADOPTION_MIN, ATTEMPT_ALPHA, CALIBRATION_N, CONFIRMATION_N,
    GAIN_MIN_PERCENT, MODELS, PLANT_COUNT, RISK_LIMIT,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'artifacts/v42/joint_risk/plan.json'
PLAN_SHA = 'b1e2d54020c95fb6e15c2355040cf2d22631b5e853687be9a54afee58aa7fab6'
OUT = ROOT/'artifacts/v42/joint_risk_design_power'
DRAWS = 20000
SEED = 425501
MC_ALPHA = .01
SOURCES = (
    'tools/analyze_joint_risk_power.py', 'tests/test_joint_risk_design_power.py',
    'docs/joint_risk_design_power.md', 'src/coopgate/recovery_risk_v39.py',
    'src/coopgate/recovery_risk_v38.py', 'src/coopgate/stratified_risk.py',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_committed(path):
    name = Path(path).relative_to(ROOT).as_posix()
    if subprocess.check_output(['git', 'show', 'HEAD:'+name], cwd=ROOT) != Path(path).read_bytes():
        raise ValueError('Commit unchanged input first: '+name)


def risk_passes(adopted, harmful):
    """Vectorize the two fixed risk bounds, without numerical p inversion.

    These are hypothetical counts, not real controller observations. The frozen
    certificate additionally inverts its nested bounds; tests check agreement
    away from numerical equality. No scientific execution code is replaced.
    """
    a, k = np.asarray(adopted), np.asarray(harmful)
    if (a.ndim != 2 or a.shape[1] != 4 or k.shape != a.shape
            or a.dtype.kind not in 'iu' or k.dtype.kind not in 'iu'
            or np.any(k < 0) or np.any(k > a) or np.any(a > PLANT_COUNT)):
        raise ValueError('Four complete integer stratum counts are required')
    delta = ATTEMPT_ALPHA/8
    counts = np.arange(PLANT_COUNT+1)
    joint_upper = np.ones(len(counts))
    joint_upper[:-1] = beta.isf(delta, counts[:-1]+1, PLANT_COUNT-counts[:-1])
    adopt_lower = np.zeros(len(counts))
    adopt_lower[1:] = beta.ppf(delta, counts[1:], PLANT_COUNT-counts[1:]+1)
    numerator = joint_upper[k].mean(axis=1)
    denominator = adopt_lower[a].mean(axis=1)
    mixture = (denominator > 0) & (numerator <= RISK_LIMIT*denominator)
    lag_upper = np.ones(len(a))
    valid = k[:, -1] < a[:, -1]
    lag_upper[valid] = beta.isf(ATTEMPT_ALPHA, k[valid, -1]+1,
                                a[valid, -1]-k[valid, -1])
    strong_lag = (a[:, -1] > 0) & (lag_upper <= RISK_LIMIT)
    return mixture, strong_lag, mixture & strong_lag


def hypothetical_populations():
    rows = []
    for adoption in (.20, .25, .40, .60):
        for risk in (0., .005, .01, .02, .05):
            rows.append(dict(name=f'homogeneous_a{adoption:g}_r{risk:g}',
                adoption=[adoption]*4, conditional_harm=[risk]*4))
    # Same marginal mixture adoption can hide a difficult strong-lag stratum.
    rows += [dict(name='lag_adoption_sparse', adoption=[.60, .60, .60, .10],
                  conditional_harm=[.005]*4),
             dict(name='lag_harm_concentrated', adoption=[.60]*4,
                  conditional_harm=[.001, .001, .001, .03])]
    return rows


def monte_carlo_interval(successes, draws, alpha):
    lower = float(beta.ppf(alpha/2, successes, draws-successes+1)) if successes else 0.
    upper = float(beta.isf(alpha/2, successes+1, draws-successes)) if successes < draws else 1.
    return dict(lower=lower, upper=upper)


def adoption_power(probability):
    counts = np.arange(CONFIRMATION_N+1)
    lower = np.zeros(len(counts))
    lower[1:] = beta.ppf(ATTEMPT_ALPHA, counts[1:], CONFIRMATION_N-counts[1:]+1)
    accepted = (binom.sf(counts-1, CONFIRMATION_N, ADOPTION_MIN) <= ATTEMPT_ALPHA) & (lower >= ADOPTION_MIN)
    minimum = int(np.flatnonzero(accepted)[0])
    assert np.array_equal(accepted, counts >= minimum)
    return dict(minimum_adoptions=minimum,
                power=float(binom.sf(minimum-1, CONFIRMATION_N, probability)))


def normal_gain_power(mean, standard_deviation):
    if not np.isfinite(standard_deviation) or standard_deviation <= 0:
        raise ValueError('A strictly positive hypothetical gain SD is required')
    critical = t.ppf(1-ATTEMPT_ALPHA, CONFIRMATION_N-1)
    noncentrality = (mean-GAIN_MIN_PERCENT)*np.sqrt(CONFIRMATION_N)/standard_deviation
    return float(nct.sf(critical, CONFIRMATION_N-1, noncentrality))


def all_model_power_bounds(risk_interval, adoption, gain):
    # Three models have the stated marginals; dependence is unrestricted.
    lower = max(0., 1-len(MODELS)*((1-risk_interval['lower'])+(1-adoption)+(1-gain)))
    upper = min(risk_interval['upper'], adoption, gain)
    return dict(lower=float(lower), upper=float(upper),
                endpoint_or_model_independence_assumed=False)


def analyze():
    if OUT.exists():
        raise FileExistsError('Preserve the original design-power report')
    if digest(PLAN) != PLAN_SHA:
        raise ValueError('The original V42 plan changed')
    # Formal study has not been bound or started. Older studies remain unread.
    if any((PLAN.parent/name).exists() for name in ('binding.json', 'calibration', 'confirmation')):
        raise ValueError('This analysis must precede V42 formal binding')
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    contract = plan['contract']
    assert contract['alpha'] == ATTEMPT_ALPHA
    assert contract['risk_limit'] == RISK_LIMIT
    assert contract['models'] == list(MODELS)
    assert contract['calibration']['initials'] == CALIBRATION_N == 4*PLANT_COUNT
    assert contract['confirmation']['initials'] == CONFIRMATION_N
    assert contract['adoption_lower_minimum'] == ADOPTION_MIN
    assert contract['paired_mean_gain_lower_strict_minimum_percent'] == GAIN_MIN_PERCENT
    for name in SOURCES:
        require_committed(ROOT/name)
    require_committed(PLAN)
    populations = hypothetical_populations()
    # Simultaneous Monte Carlo uncertainty applies to 22 risk-power estimates,
    # not to controller risk. It consumes no formal experiment alpha budget.
    per_row_alpha = MC_ALPHA/len(populations)
    risk_rows, joint_rows = [], []
    for index, population in enumerate(populations):
        rng = np.random.default_rng(SEED+index)
        a = rng.binomial(PLANT_COUNT, population['adoption'], size=(DRAWS, 4))
        k = rng.binomial(a, population['conditional_harm'])
        mixture, lag, joint = risk_passes(a, k)
        successes = int(joint.sum())
        interval = monte_carlo_interval(successes, DRAWS, per_row_alpha)
        adoption = adoption_power(float(np.mean(population['adoption'])))
        risk_rows.append(dict(**population, seed=SEED+index, draws=DRAWS,
            joint_risk_passes=successes, estimated_joint_risk_power=successes/DRAWS,
            mixture_passes=int(mixture.sum()), strong_lag_passes=int(lag.sum()),
            monte_carlo_interval=interval, confirmation_adoption=adoption))
        for mean in (.75, 1., 1.5):
            for sd in (1., 3., 5.):
                gain = normal_gain_power(mean, sd)
                joint_rows.append(dict(population=population['name'],
                    hypothetical_mean_gain_percent=mean, hypothetical_gain_sd_percent=sd,
                    single_model_gain_power=gain,
                    all_three_models_all_components_power_bounds=all_model_power_bounds(
                        interval, adoption['power'], gain)))
    report = dict(plan_sha256=PLAN_SHA, source_sha256={n:digest(ROOT/n) for n in SOURCES},
        design_only=True, formal_outcomes_read=False, additional_sumo_runs=0,
        formal_plan_or_budget_changed=False, mc_draws_per_population=DRAWS,
        mc_simultaneous_coverage=1-MC_ALPHA, risk_power=risk_rows, joint_power_bounds=joint_rows,
        hypotheses='IID rows within each fixed plant; independent simulated calibration strata; all three models have each displayed marginal scenario.',
        gain_assumption='IID normal paired percentage gains with the displayed positive SD.',
        execution_assumption='Population execution failure probability is assumed zero only for this design-power calculation; not inferred from any data.',
        limitations='Unknown gain/adoption and between-model dependence handled only by a conservative union bound; often vacuous. This is not an actual certificate or an efficacy forecast. No sample extension, model selection or threshold adjustment follows automatically.')
    OUT.mkdir(parents=True, exist_ok=False)
    with (OUT/'report.json').open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(report_sha256=digest(OUT/'report.json'), populations=len(risk_rows),
                         joint_scenarios=len(joint_rows), formal_outcomes_read=False)))


if __name__ == '__main__':
    analyze()
