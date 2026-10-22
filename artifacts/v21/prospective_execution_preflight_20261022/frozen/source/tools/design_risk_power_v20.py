"""Prospective fixed-sample power scenarios, without reading experiment outcomes."""
import json
import math
from pathlib import Path
import shutil

import numpy as np
from scipy.stats import beta, binom

from probe_disturbed_gate_v19 import digest, save

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'artifacts/v20/risk_power_design'


def exact_cutoffs(maximum, risk_limit, level):
    ns = np.arange(maximum + 1)
    quantiles = binom.ppf(level, ns, risk_limit).astype(int)
    cutoffs = quantiles - (binom.cdf(quantiles, ns, risk_limit) > level)
    cutoffs[0] = -1  # Conditional risk is undefined without adoption.
    valid = cutoffs >= 0
    if np.any(binom.cdf(cutoffs[valid], ns[valid], risk_limit) > level):
        raise ValueError('Invalid exact certification cutoff')
    if np.any(binom.cdf(cutoffs[1:] + 1, ns[1:], risk_limit) <= level):
        raise ValueError('Cutoff is not the largest passing integer')
    np.testing.assert_array_less(beta.ppf(1 - level, cutoffs[valid] + 1, ns[valid] - cutoffs[valid]),
                                 np.full(valid.sum(), risk_limit + 1e-12))
    np.testing.assert_array_less(np.full(valid.sum(), risk_limit - 1e-12),
                                 beta.ppf(1 - level, cutoffs[valid] + 2, ns[valid] - cutoffs[valid] - 1))
    return cutoffs


def single_target_power(initials, adoption_probability, true_risk, cutoffs):
    ns = np.arange(initials + 1)
    value = float(binom.pmf(ns, initials, adoption_probability) @ binom.cdf(cutoffs[:initials + 1], ns, true_risk))
    # Finite probability summation may exceed one by a few machine epsilons.
    # Reject a substantive discrepancy; clip only the declared rounding band.
    if not np.isfinite(value) or value < -1e-12 or value > 1. + 1e-12:
        raise ValueError('Probability summation outside numerical roundoff')
    return float(np.clip(value, 0., 1.))


def joint_lower_bound(marginal_powers):
    powers = np.asarray(marginal_powers, dtype=float)
    if powers.ndim != 1 or not len(powers) or not np.isfinite(powers).all() or np.any((powers < 0.) | (powers > 1.)):
        raise ValueError('Valid marginal powers required')
    return float(max(0., 1. - np.sum(1. - powers)))


def run():
    if OUTPUT.exists():
        raise FileExistsError('Preserve the prospective power design')
    # Retain the previous multiplicity budget as a design reference only.
    protocol_path = ROOT / 'artifacts/v19/risk_validation/protocol.json'
    protocol = json.loads(protocol_path.read_text())
    config = protocol['spec']['risk_validation']
    if config['method'] != 'bonferroni':
        raise ValueError('The reference familywise allocation changed')
    candidates, targets = len(protocol['spec']['methods']) - 1, len(config['targets'])
    alpha, delta = config['alpha'], config['delta']
    level = delta / (candidates * targets)
    initials_grid = (1024, 2048, 4096, 8192)
    adoption_grid, risk_grid = (.1, .2, .3), (0., .005, .01, .02, .03)
    cutoffs = exact_cutoffs(max(initials_grid), alpha, level)
    minimum = math.ceil(math.log(level) / math.log1p(-alpha))
    if int(np.flatnonzero(cutoffs >= 0)[0]) != minimum:
        raise ValueError('Zero-harm sample threshold changed')
    rows = []
    for n in initials_grid:
        for q in adoption_grid:
            for risk in risk_grid:
                power = single_target_power(n, q, risk, cutoffs)
                rows.append(dict(independent_initials=n, hypothetical_adoption_probability=q,
                                 hypothetical_conditional_risk=risk, single_target_power=power,
                                 all_six_risk_tests_power_lower_bound=joint_lower_bound([power] * 6),
                                 sumo_runs_if_four_modes_and_ten_controllers=40 * n))
            null = single_target_power(n, q, alpha, cutoffs)
            if null > level + 1e-12:
                raise ValueError('Null false-certification probability exceeds its budget')
            zero = single_target_power(n, q, 0., cutoffs)
            np.testing.assert_allclose(zero, binom.sf(minimum - 1, n, q), atol=1e-12, rtol=1e-12)
    recommendations = []
    for q in adoption_grid:
        for risk in risk_grid:
            options = [r for r in rows if r['hypothetical_adoption_probability'] == q and r['hypothetical_conditional_risk'] == risk
                       and r['all_six_risk_tests_power_lower_bound'] >= .9]
            recommendations.append(dict(hypothetical_adoption_probability=q, hypothetical_conditional_risk=risk,
                smallest_grid_size_with_90percent_six_test_lower_bound=options[0]['independent_initials'] if options else None))
    OUTPUT.mkdir(parents=True)
    report = dict(purpose='Prospective exact fixed-sample design scenarios; no future protocol or stopping rule selected',
                  reference_candidates=candidates, reference_targets=targets, risk_limit=alpha,
                  total_false_certification_budget=delta, per_test_level=level, zero_harm_minimum_adoptions=minimum,
                  rows=rows, grid_size_comparisons=recommendations,
                  six_test_scope='Three fixed policies and two targets each. The displayed lower bound assumes every marginal has the row-specified hypothetical adoption and conditional-risk probabilities. No target independence is assumed.',
                  excluded_scope='The bound concerns risk certification only, not passing independent-test adoption, gain or positive confidence-interval requirements.',
                  development_or_final_outcomes_read=False, risk_certified=False, sample_size_selected=False,
                  current_protocol_or_stopping_rule_changed=False, reference_protocol_sha256=digest(protocol_path),
                  probability_roundoff_clipping_tolerance=1e-12, source_sha256=digest(__file__))
    save(OUTPUT / 'analysis.json', report)
    shutil.copy2(__file__, OUTPUT / Path(__file__).name)
    save(OUTPUT / 'sha256.json', {p.name: digest(p) for p in OUTPUT.iterdir() if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps(dict(zero_harm_minimum_adoptions=minimum,
                          adoption_20percent=[r for r in rows if r['hypothetical_adoption_probability'] == .2
                                             and r['hypothetical_conditional_risk'] in (.005, .01, .02)],
                          grid_size_comparisons=recommendations)), flush=True)


if __name__ == '__main__':
    run()
