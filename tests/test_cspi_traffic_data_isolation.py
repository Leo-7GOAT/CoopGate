"""Data-boundary checks for the episode-level author-method integration.

Run with the separate baseline environment; do not install new libraries into
the frozen simulation runtime. These checks make no efficacy claim.
"""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import run_cspi_traffic_development as integration


class DataBoundaries(unittest.TestCase):
    def test_heldout_rewards_cannot_change_fitted_models_or_scores(self):
        rng = np.random.default_rng(902)
        x = rng.normal(size=(160, 10))
        treatment = np.arange(160) % 2
        observed = x[:, 0] + treatment * x[:, 1] + rng.normal(size=160)
        first, params = integration.observations(x, treatment, observed)
        altered = observed.copy()
        altered[64:] += 1000000
        second, params_after = integration.observations(x, treatment, altered)
        self.assertEqual(params, params_after)
        for field in ('mu_0_est', 'mu_1_est', 'ite_est'):
            np.testing.assert_array_equal(first[field], second[field])
        self.assertFalse(np.array_equal(first.outcome, second.outcome))

    def test_evaluation_features_cannot_change_training_or_test_predictions(self):
        rng = np.random.default_rng(903)
        x = rng.normal(size=(160, 10))
        treatment = np.arange(160) % 2
        observed = rng.normal(size=160)
        first, params = integration.observations(x, treatment, observed)
        shifted = x.copy()
        shifted[128:] += 10000
        second, params_after = integration.observations(shifted, treatment, observed)
        self.assertEqual(params, params_after)
        np.testing.assert_array_equal(first.ite_est[:128], second.ite_est[:128])

    def test_duplicate_scores_are_rejected_without_jitter(self):
        data, _ = integration.observations(np.ones((160, 10)), np.arange(160) % 2, np.zeros(160))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            integration.select({}, data, 0)

    def test_fallback_does_not_hide_failed_reference(self):
        baseline = dict(team_cost=25.)
        cases = [dict(baseline=baseline, baseline_failed=True) for _ in range(32)]
        result = integration.evaluate(cases, np.zeros(32, bool))
        self.assertEqual(result['completed'], 0)
        self.assertEqual(result['baseline_failed'], 32)
        self.assertIsNone(result['all_scope_cost_gain_percent'])
        self.assertIsNone(result['empirical_conditional_harm'])


if __name__ == '__main__':
    unittest.main()
