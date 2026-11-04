"""Post-pilot, pre-Monte-Carlo fixed sizing; no formal alpha allocation."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from scipy.stats import beta, binom, t

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/mappo_ctde_20261101'
PILOT = BASE / 'development/local_top_two_fresh_pilot_20261104'
ANALYSIS = BASE / 'development/local_top_two_fresh_analysis_20261104'
OUT = BASE / 'top_two_formal_power_20261104'
PROTOCOL = ROOT / 'docs/top_two_formal_power_20261104_protocol.md'
PLANTS = ('nominal', 'gain_drag', 'lag_mild', 'lag_strong')
SEEDS = ('4703', '4704', '4705')
GRID = sorted(((a, c) for a in (4096, 6144, 8192) for c in (1024, 1536)),
              key=lambda pair: (7 * pair[0] + 16 * pair[1], pair))
B = 4096
STREAM = 650401
ERROR = .05 / 42
STRESSES = (.003, .005, .01)
MECHANISMS = ('shared', 'independent')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, obj):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')


def register():
    OUT.mkdir(exist_ok=False)
    write(OUT / 'registration.json', dict(
        source_sha256=sha(__file__), protocol_sha256=sha(PROTOCOL),
        pilot_report_sha256=sha(PILOT / 'report.json'),
        analysis_report_sha256=sha(ANALYSIS / 'report.json'),
        paired_outcomes_sha256=sha(ANALYSIS / 'paired_outcomes.json'),
        pilot_effects_already_read=True, monte_carlo_results_not_read=True,
        new_formal_error_allocated=False, new_formal_runs=0,
        grid=GRID, replicates=B, stream=STREAM,
        extra_conditional_harm_stresses=STRESSES, mechanisms=MECHANISMS,
        selection_stress=.005, proposed_unused_error=ERROR))


def load():
    registration = read(OUT / 'registration.json')
    for key, path in [('source_sha256', __file__), ('protocol_sha256', PROTOCOL),
                      ('pilot_report_sha256', PILOT / 'report.json'),
                      ('analysis_report_sha256', ANALYSIS / 'report.json'),
                      ('paired_outcomes_sha256', ANALYSIS / 'paired_outcomes.json')]:
        if registration[key] != sha(path):
            raise ValueError('Registered sizing input changed: ' + str(path))
    for directory in (PILOT, ANALYSIS):
        owner = read(directory / 'owner_native_exit.json')
        if owner['actual_native_exit_code'] != 0 or owner['report_sha256'] != sha(directory / 'report.json'):
            raise ValueError('Actual completed owner required')
    report = read(PILOT / 'report.json')
    if not report['completed'] or report['actual_sumo_executions'] != 2240:
        raise ValueError('Whole fresh development pilot required')
    rows = read(ANALYSIS / 'paired_outcomes.json')
    groups = {}
    for row in rows:
        key = row['case']['cluster_id']
        item = groups.setdefault(key, dict(case=row['case'], models={}))
        if row['seed'] in item['models'] or row['case'] != item['case']:
            raise ValueError('Duplicate or inconsistent paired case')
        item['models'][row['seed']] = row['top_history']
    if len(groups) != 320 or len(rows) != 960:
        raise ValueError('All 320 independent development rows required')
    ordered = []
    for key in sorted(groups):
        item = groups[key]
        if set(item['models']) != set(SEEDS):
            raise ValueError('All three fixed model columns required')
        ordered.append(item)
    plants = np.array([x['case']['plant'] for x in ordered])
    adopt = np.array([[x['models'][s]['adopted'] for s in SEEDS] for x in ordered], bool)
    harm = np.array([[x['models'][s]['harm'] for s in SEEDS] for x in ordered], bool)
    gains = np.array([[x['models'][s]['gain'] for s in SEEDS] for x in ordered], float)
    failures = sum(v['execution_failures'] for v in report['results'].values())
    for i, seed in enumerate(SEEDS):
        q = report['results']['fresh:top_history:' + seed]
        if (int(adopt[:, i].sum()) != q['adopted'] or int((adopt[:, i] & harm[:, i]).sum()) != q['harmful_adopted']
                or not np.isclose(gains[:, i].mean(), q['mean_paired_percentage_gain'], atol=1e-12, rtol=0)):
            raise ValueError('Sizing outcomes differ from complete original data')
    if any((plants == p).sum() != 80 for p in PLANTS):
        raise ValueError('Original balanced pilot changed')
    return plants, adopt, harm, gains, failures


def calibration_bootstrap(plants, adopt, harm, n, rng):
    """Joint (A,H) pattern counts preserve complete-row bootstrap dependence."""
    codes = (adopt * np.array([1, 2, 4])).sum(1) + ((adopt & harm) * np.array([8, 16, 32])).sum(1)
    patterns = np.arange(64)
    ap = (patterns[:, None] & np.array([1, 2, 4])) != 0
    hp = (patterns[:, None] & np.array([8, 16, 32])) != 0
    counts = [rng.multinomial(n // 4, np.bincount(codes[plants == p], minlength=64) / 80, size=B)
              for p in PLANTS]
    adopted = np.array([v @ ap.astype(np.int64) for v in counts])
    harmful = np.array([v @ hp.astype(np.int64) for v in counts])
    return counts, ap, hp, adopted, harmful


def added_harms(counts, ap, hp, original, stress, mechanism, rng):
    result = original.copy()
    for plant, count in enumerate(counts):
        if mechanism == 'shared':
            shocks = rng.binomial(count, stress)
            result[plant] += shocks @ (ap & ~hp).astype(np.int64)
        elif mechanism == 'independent':
            for column in range(3):
                eligible = count[:, ap[:, column] & ~hp[:, column]].sum(1)
                result[plant, :, column] += rng.binomial(eligible, stress)
        else:
            raise ValueError('Unregistered dependence mechanism')
    return result


def gain_adoption_checks(counts, adopt, gains, n):
    ac = counts @ adopt.astype(np.int64)
    alo = np.where(ac > 0, beta.ppf(ERROR, ac, n - ac + 1), 0.)
    adoption_ok = np.all((alo >= .2 + 1e-12) & (binom.sf(ac - 1, n, .2) <= ERROR), axis=1)
    mean = counts @ gains / n
    variance = np.maximum(0., (counts @ (gains * gains) / n - mean * mean) * n / (n - 1))
    se = np.sqrt(variance / n)
    lower = mean - t.isf(ERROR, n - 1) * se
    statistic = np.divide(mean - .5, se, out=np.zeros_like(mean), where=se > 1e-12)
    gain_ok = np.all((variance > 1e-24) & (lower > .5) & (t.sf(statistic, n - 1) <= ERROR), axis=1)
    return adoption_ok, gain_ok


def risk_checks(adopted, harmful, n):
    # Same four-stratum, eight-component construction as the frozen certificate.
    per = n // 4
    upper = np.where(harmful < per, beta.isf(ERROR / 8, harmful + 1, per - harmful), 1.)
    lower = np.where(adopted > 0, beta.ppf(ERROR / 8, adopted, per - adopted + 1), 0.)
    denominator = lower.mean(0)
    mixture = np.divide(upper.mean(0), denominator, out=np.ones_like(denominator), where=denominator > 0)
    strong = np.where(harmful[-1] < adopted[-1], beta.isf(ERROR, harmful[-1] + 1, adopted[-1] - harmful[-1]), 1.)
    return np.all((mixture <= .05 - 1e-12) & (strong <= .05 - 1e-12)
                  & (binom.cdf(harmful[-1], adopted[-1], .05) <= ERROR), axis=1)


def run():
    plants, adopt, harm, gains, failures = load()
    started = time.perf_counter()
    table = []
    for cal, conf in GRID:
        rng = np.random.default_rng(np.random.SeedSequence([STREAM, cal, conf]))
        counts, ap, hp, ac, hc = calibration_bootstrap(plants, adopt, harm, cal, rng)
        cc = rng.multinomial(conf, np.full(len(adopt), 1 / len(adopt)), size=B)
        adoption_ok, gain_ok = gain_adoption_checks(cc, adopt, gains, conf)
        for stress_index, stress in enumerate(STRESSES):
            for mechanism_index, mechanism in enumerate(MECHANISMS):
                srng = np.random.default_rng(np.random.SeedSequence([STREAM, cal, conf, stress_index, mechanism_index]))
                hh = added_harms(counts, ap, hp, hc, stress, mechanism, srng)
                risk_ok = risk_checks(ac, hh, cal)
                passed = risk_ok & adoption_ok & gain_ok
                k = int(passed.sum())
                mc_lower = float(beta.ppf(.01 / (len(GRID) * len(MECHANISMS)), k, B - k + 1)) if k else 0.
                row = dict(calibration_initials=cal, confirmation_initials=conf,
                    actual_runs_if_registered=7 * cal + 16 * conf, bootstrap_replicates=B,
                    extra_conditional_harm_stress=stress, shock_dependence=mechanism,
                    joint_passes=k, synthetic_joint_pass_fraction=float(passed.mean()),
                    mc_joint_pass_lower=mc_lower, synthetic_risk_pass_fraction=float(risk_ok.mean()),
                    synthetic_adoption_pass_fraction=float(adoption_ok.mean()),
                    synthetic_gain_pass_fraction=float(gain_ok.mean()))
                table.append(row)
                print(json.dumps(row), flush=True)
    chosen = None
    if not failures:
        for cal, conf in GRID:
            selected_rows = [r for r in table if r['calibration_initials'] == cal and r['confirmation_initials'] == conf
                             and r['extra_conditional_harm_stress'] == .005]
            if len(selected_rows) == 2 and all(r['mc_joint_pass_lower'] > .8 for r in selected_rows):
                chosen = dict(calibration_initials=cal, confirmation_initials=conf,
                              actual_runs_if_registered=7 * cal + 16 * conf)
                break
    write(OUT / 'report.json', dict(completed=True, whole_grid=table, selected_design=chosen,
        pilot_initials=320, pilot_actual_runs=2240, actual_pilot_failures=failures,
        pilot_adoption=adopt.sum(0).tolist(), pilot_harmful_adopted=(adopt & harm).sum(0).tolist(),
        pilot_mean_gain_percent=gains.mean(0).tolist(),
        registration_sha256=sha(OUT / 'registration.json'), proposed_attempt=6, proposed_error=ERROR,
        new_error_allocated=False, new_formal_runs=0, pilot_effects_read_before_sizing_rule=True,
        conditional_planning_only=True, real_power_guaranteed=False,
        elapsed_s=time.perf_counter() - started))
    print(json.dumps(dict(completed=True, selected_design=chosen, new_error_allocated=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('register', 'run'))
    args = parser.parse_args()
    (register if args.action == 'register' else run)()
