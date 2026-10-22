"""Exact design-only binomial power; never inspect calibration outcomes."""
import argparse
import json
import math
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, binom

from probe_disturbed_gate_v19 import digest, save

ROOT = Path(__file__).resolve().parents[1]


def analyze(output):
    protocol_path = ROOT / 'artifacts/v19/risk_validation/protocol.json'
    protocol = json.loads(protocol_path.read_text())
    spec, config = protocol['spec'], protocol['spec']['risk_validation']
    candidates = len(spec['methods']) - 1
    target_count = len(config['targets'])
    alpha, level, n_total = config['alpha'], config['delta'] / (candidates * target_count), spec['n_initials']
    assert config['method'] == 'bonferroni'
    ns = np.arange(n_total + 1)
    cutoffs = np.full(n_total + 1, -1, dtype=int)
    for n in range(1, n_total + 1):
        admitted = np.flatnonzero(binom.cdf(np.arange(n + 1), n, alpha) <= level)
        if len(admitted):
            k = cutoffs[n] = admitted[-1]
            # Check inversion against a separately evaluated beta upper bound,
            # including that the next integer count cannot pass.
            assert beta.ppf(1 - level, k + 1, n - k) <= alpha + 1e-12
            assert k + 1 == n or beta.ppf(1 - level, k + 2, n - k - 1) > alpha - 1e-12
    minimum = math.ceil(math.log(level) / math.log1p(-alpha))
    assert np.flatnonzero(cutoffs >= 0)[0] == minimum
    assert (1 - alpha) ** minimum <= level < (1 - alpha) ** (minimum - 1)
    table = [dict(adopted=n, maximum_harm_count=int(cutoffs[n]),
                  maximum_empirical_harm_rate=float(cutoffs[n] / n) if cutoffs[n] >= 0 else None)
             for n in (50, 100, minimum, 153, 205, 307, 410, 512, 614, 768, 1024)]
    curves = []
    for risk in (.01, .02, .03):
        probability_given_adoption_count = binom.cdf(cutoffs, ns, risk)
        for adoption in np.linspace(.05, .8, 76):
            power = float(binom.pmf(ns, n_total, adoption) @ probability_given_adoption_count)
            curves.append(dict(true_conditional_risk=risk, adoption_probability=float(adoption),
                               single_candidate_target_certification_probability=power))
    report = dict(
        purpose='Design-only exact binomial power for the already frozen protocol',
        independent_initials=n_total, candidates=candidates, risk_targets=target_count,
        risk_limit=alpha, total_confidence_failure_budget=config['delta'], per_candidate_target_level=level,
        minimum_zero_harm_adoptions=minimum, count_cutoffs=table, curves=curves,
        calibration_outcomes_read=False, risk_certified=False,
        scope='A hypothetical IID population with specified adoption probability and conditional risk; one fixed candidate/target at the allocated multiplicity level. Not power for simultaneous certification of both dependent targets or a prediction of actual outcomes.',
        protocol_changed=False, protocol_sha256=digest(protocol_path))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'analysis.json', report)
    fig, ax = plt.subplots(figsize=(8.1, 4.8), layout='constrained')
    for risk, color in zip((.01, .02, .03), ('#06766f', '#356cac', '#b55e31')):
        records = [r for r in curves if r['true_conditional_risk'] == risk]
        ax.plot([100 * r['adoption_probability'] for r in records],
                [r['single_candidate_target_certification_probability'] for r in records],
                color=color, label=f'True conditional risk {risk:.0%}')
    ax.set(xlabel='Population probability of actual learning adoption (%)',
           ylabel='Probability of certification', ylim=(0, 1.02), xlim=(5, 80))
    ax.grid(alpha=.2)
    ax.legend(loc='lower right')
    ax.set_title('Frozen design: 1,024 independent initials, 5% risk limit\n'
                 'One candidate / target; level 0.05 / (9 × 2). Hypothetical populations only.', fontsize=10)
    fig.savefig(output / 'certification_power.png', dpi=200)
    plt.close(fig)
    shutil.copy2(__file__, output / Path(__file__).name)
    save(output / 'sha256.json', {p.name: digest(p) for p in output.iterdir() if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps(dict(minimum_zero_harm_adoptions=minimum, count_cutoffs=table)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output')
    args = parser.parse_args()
    analyze(args.output)
