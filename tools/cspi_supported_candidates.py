"""Disclosed CSPI compatibility adapter: tuning-only numerical support filtering.

Author selection/test function bodies remain untouched. This is not a new risk
certificate, an endpoint repair, or a finite-sample guarantee. Test data never
enter candidate selection. Any selected-family test degeneracy rejects that
entire family without selecting a replacement or removing a tested hypothesis.
"""
import hashlib
import json
import warnings

import numpy as np

BASELINE = 1e6


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def influence(data, cutoffs, dr):
    columns = ('treatment', 'propensity', 'outcome', 'mu_0_est', 'mu_1_est', 'ite_est')
    arrays = {name: np.asarray(data[name], dtype=float) for name in columns}
    n = len(arrays['outcome'])
    if n < 2 or any(a.shape != (n,) or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('Finite complete independent-unit rows required')
    t, p, y, m0, m1, score = (arrays[name] for name in columns)
    if not np.isin(t, [0, 1]).all() or np.any((p <= 0) | (p >= 1)):
        raise ValueError('Binary assignment with strict overlap required')
    cutoffs = np.asarray(cutoffs, dtype=float)
    if cutoffs.ndim != 1 or not len(cutoffs) or not np.isfinite(cutoffs).all():
        raise ValueError('Nonempty finite cutoff vector required')
    if np.any(score >= BASELINE) or np.any(cutoffs >= BASELINE):
        raise ValueError('Fixed always-baseline threshold domain exceeded')
    if np.isin(score, np.append(cutoffs, BASELINE)).any():
        raise ValueError('Original author PDE endpoint ambiguity; no silent repair')
    effect = t*y/p-(1-t)*y/(1-p)
    if dr:
        effect = m1-m0+t*(y-m1)/p-(1-t)*(y-m0)/(1-p)
    difference = (score[None, :] >= cutoffs[:, None]).astype(float)
    return difference*effect[None, :], difference.sum(axis=1).astype(int)


def statistics(f, data, cutoffs, dr):
    scores, support = influence(data, cutoffs, dr)
    values = scores.mean(axis=1)
    covariance = np.atleast_2d(np.cov(scores)/scores.shape[1])
    original_values, original_covariance = f['PDE'](cutoffs, BASELINE, data, by_quantile=False, dr=dr)
    np.testing.assert_allclose(original_values, values, atol=1e-12, rtol=0)
    np.testing.assert_allclose(np.atleast_2d(original_covariance), covariance, atol=1e-12, rtol=0)
    if not np.isfinite(values).all() or not np.isfinite(covariance).all():
        raise ValueError('Nonfinite influence estimates')
    return values, covariance, support


def choose(f, fit_scores, tune, seed):
    """Only fitting scores and tuning observations are accepted by this API."""
    fit_scores = np.asarray(fit_scores, dtype=float)
    if fit_scores.ndim != 1 or not len(fit_scores) or not np.isfinite(fit_scores).all():
        raise ValueError('Finite training-only scores required')
    cutoffs = np.quantile(fit_scores, [.1, .3, .5, .7, .9])
    if len(set(cutoffs)) != 5:
        raise ValueError('Duplicate training candidates; no jitter')
    diagnostics = []
    keep = np.ones(len(cutoffs), dtype=bool)
    for dr in (False, True):
        values, covariance, support = statistics(f, tune, cutoffs, dr)
        valid = (support > 0) & (np.diag(covariance) > 0)
        keep &= valid
        diagnostics.append(dict(dr=dr, estimates=values.tolist(), variance=np.diag(covariance).tolist(),
                                support=support.tolist(), computable=valid.tolist()))
    retained = cutoffs[keep]
    if len(retained) >= 2:
        np.random.seed(seed)
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            raw = f['cutoff_selection'](retained, BASELINE, tune, gamma=.05, n_sim=10000, test_ratio=1)
        selected = dict(cspi=np.asarray(raw[1][1:2], dtype=float), cspi_mt=np.asarray(raw[3], dtype=float))
        route = 'unchanged_author_selection'
    elif len(retained) == 1:
        # The author selector expects a matrix but PDE returns scalar covariance
        # for a singleton. There is exactly one available policy to test.
        selected = dict(cspi=retained.copy(), cspi_mt=retained.copy())
        route = 'singleton_no_search'
    else:
        selected = dict(cspi=np.array([]), cspi_mt=np.array([]))
        route = 'no_computable_candidate'
    if any(a.ndim != 1 or not np.isin(a, retained).all() or len(set(a)) != len(a) for a in selected.values()):
        raise ValueError('Author selection escaped the frozen retained family')
    result = dict(candidates=cutoffs.tolist(), retained=retained.tolist(), diagnostics=diagnostics,
                  selected={k: v.tolist() for k, v in selected.items()}, route=route, seed=seed,
                  tune_inputs_sha256=fingerprint({k: np.asarray(tune[k]).tolist() for k in tune.columns}))
    return dict(payload=result, sha256=fingerprint(result))


def test_selected(f, selection, test, seed):
    payload = selection['payload']
    if fingerprint(payload) != selection['sha256']:
        raise ValueError('Selection changed after freezing')
    results = []
    for method, thresholds in payload['selected'].items():
        selected = np.asarray(thresholds, dtype=float)
        result = dict(method=method, selected=thresholds, passed=[], deployed_cutoff=BASELINE)
        if not len(selected):
            result.update(status='untestable', reason='No tuning-computable candidate')
            results.append(result)
            continue
        try:
            values, covariance, support = statistics(f, test, selected, True)
            result.update(estimates=values.tolist(), variance=np.diag(covariance).tolist(), support=support.tolist())
            if np.any(support <= 0) or np.any(np.diag(covariance) <= 0):
                raise ValueError('Selected test family has zero support or variance; no replacement search')
            with warnings.catch_warnings():
                warnings.simplefilter('error', RuntimeWarning)
                if method == 'cspi':
                    passed = f['safety_test_single'](selected, BASELINE, test, gamma=.05, dr=True)
                else:
                    np.random.seed(seed)
                    passed = f['safety_test'](selected, BASELINE, test, gamma=.05, nsim=10000, dr=True)
            passed = np.asarray(passed).reshape(-1)
            if passed.shape != selected.shape or not np.isin(passed, [0, 1]).all():
                raise ValueError('Invalid author test response')
            passed = passed.astype(bool)
            # Selecting among a simultaneously certified family does not test
            # additional cutoffs. Use only this test's values for a fixed tie rule.
            if passed.any():
                eligible = np.flatnonzero(passed)
                best = eligible[np.argmax(values[eligible])]
                result['deployed_cutoff'] = float(selected[best])
            result.update(status='tested', passed=passed.tolist())
        except (ValueError, RuntimeWarning) as error:
            result.update(status='untestable', reason=str(error))
        results.append(result)
    return dict(selection_sha256=selection['sha256'], methods=results, new_risk_certificate=False)
