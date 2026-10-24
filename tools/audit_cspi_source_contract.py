"""Probe exact upstream PDE function against threshold-policy identities.

Only the named function AST is executed; upstream simulations/plotting are not
launched. This does not reproduce CSPI cutoff selection or its error rates.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np


BLOB = '84323a8cf90351cf69fbbc59300523ddca54f625'
COMMIT = 'b5af605a38a7866c52745c153e02cb4bf049ea01'


class Fixture:
    def __init__(self, scores):
        self.ite_est = np.asarray(scores, dtype=float)
        n = len(scores)
        self.treatment = np.arange(n) % 2
        self.propensity = np.full(n, .5)
        self.mu_0_est, self.mu_1_est = np.zeros(n), np.ones(n)
        self.outcome = self.treatment.astype(float)
        self.shape = (n, 6)

    def __len__(self):
        return self.shape[0]


def exact_fixture_difference(data, cutoffs, baseline):
    # Both nuisance means equal the true potential outcomes (0 and 1), so the
    # DR treatment-effect score is exactly 1. Only policy disagreement remains.
    scores = np.array([(data.ite_est >= c).astype(float)
                       - (data.ite_est >= baseline).astype(float) for c in cutoffs])
    return scores.mean(axis=1), np.cov(scores) / len(data)


def probe(source):
    raw = source.read_bytes()
    blob = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    if blob != BLOB:
        raise ValueError('Upstream source is not the pinned exact Git blob')
    tree = ast.parse(raw.decode('utf-8'))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'PDE']
    assert len(selected) == 1
    namespace = {'np': np}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), namespace)
    pde = namespace['PDE']

    data = Fixture(np.linspace(-2., 2., 256))
    cutoffs, baseline = np.array([-.7, .6]), .1
    actual, covariance = pde(cutoffs, baseline, data, by_quantile=False, dr=True)
    expected, expected_cov = exact_fixture_difference(data, cutoffs, baseline)
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)
    np.testing.assert_allclose(covariance, expected_cov, atol=1e-14, rtol=0)

    tied = Fixture(np.zeros(256))
    equal, _ = pde(np.array([0.]), 0., tied, by_quantile=False, dr=True)
    equal_expected, _ = exact_fixture_difference(tied, [0.], 0.)
    assert float(equal_expected[0]) == 0. and float(equal[0]) == 1.
    endpoint, _ = pde(np.array([.5]), 0., tied, by_quantile=False, dr=True)
    endpoint_expected, _ = exact_fixture_difference(tied, [.5], 0.)
    np.testing.assert_array_equal(endpoint, endpoint_expected)
    upper_tied = Fixture(np.full(256, .5))
    upper, _ = pde(np.array([.5]), 0., upper_tied, by_quantile=False, dr=True)
    upper_expected, _ = exact_fixture_difference(upper_tied, [.5], 0.)
    assert float(upper[0]) == -1. and float(upper_expected[0]) == 0.

    return dict(upstream_repository='https://github.com/facebookresearch/CSPI-MT',
                upstream_commit=COMMIT, upstream_file=source.name, upstream_git_blob=blob,
                upstream_sha256=hashlib.sha256(raw).hexdigest(),
                extracted_function='PDE', function_ast_modified=False,
                full_script_executed=False, synthetic_fixture=True,
                sumo_runs=0, efficacy_claimed=False,
                no_tie_point_estimate_and_covariance_match=True,
                identical_threshold_true_difference=float(equal_expected[0]),
                identical_threshold_upstream_estimate=float(equal[0]),
                one_boundary_direction_positive_control=True,
                distinct_cutoffs_upper_tie_true_difference=float(upper_expected[0]),
                distinct_cutoffs_upper_tie_upstream_estimate=float(upper[0]),
                interpretation='inclusive disagreement interval fails identical-threshold identity when score has point mass; no claim of failure for continuous-score paper experiments')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
