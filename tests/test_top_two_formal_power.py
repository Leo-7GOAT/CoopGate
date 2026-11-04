import numpy as np
from scipy.stats import beta, binom, t
from coopgate.stratified_risk import conditional_risk_certificate
from operations import top_two_formal_power_20261104 as p


def test_multinomial_sufficient_statistics_equal_explicit_resampled_rows():
    rng = np.random.default_rng(8119)
    adopt = rng.random((320, 3)) < .3
    gains = rng.normal(1.3, 4., (320, 3))
    indices = rng.integers(320, size=(12, 1024))
    counts = np.array([np.bincount(row, minlength=320) for row in indices])
    ao, go = p.gain_adoption_checks(counts, adopt, gains, 1024)
    for row, aok, gok in zip(indices, ao, go):
        n = adopt[row].sum(0)
        lower = np.where(n > 0, beta.ppf(p.ERROR, n, 1025 - n), 0.)
        assert aok == np.all((lower >= .2 + 1e-12) & (binom.sf(n - 1, 1024, .2) <= p.ERROR))
        g = gains[row]
        se = g.std(0, ddof=1) / 32
        gl = g.mean(0) - t.isf(p.ERROR, 1023) * se
        assert gok == np.all((gl > .5) & (t.sf((g.mean(0) - .5) / se, 1023) <= p.ERROR))
        np.testing.assert_allclose(counts[0] @ gains / 1024, gains[indices[0]].mean(0), atol=1e-12)


def test_synthetic_risk_decisions_match_unchanged_stratified_certificate():
    # Distinct low/high adoption, zero/nonzero harm, and strong-lag failures.
    examples = [([310, 330, 320, 315], [0, 1, 0, 0]),
                ([310, 330, 320, 315], [5, 6, 5, 5]),
                ([210, 220, 210, 220], [0, 0, 0, 0]),
                ([950, 940, 930, 50], [0, 0, 0, 4])]
    for a, h in examples:
        ac = np.repeat(np.array(a)[:, None, None], 3, axis=2)
        hc = np.repeat(np.array(h)[:, None, None], 3, axis=2)
        actual = p.risk_checks(ac, hc, 4096)[0]
        mix = conditional_risk_certificate(h, a, [1024] * 4, [.25] * 4,
                                          alpha=p.ERROR, risk_limit=.05)
        strong_upper = beta.isf(p.ERROR, h[-1] + 1, a[-1] - h[-1])
        expected = mix['certified'] and strong_upper <= .05 and binom.cdf(h[-1], a[-1], .05) <= p.ERROR
        assert actual == expected


def test_zero_shock_retains_empirical_harms_and_never_counts_nonadopters():
    rng = np.random.default_rng(18)
    plants = np.repeat(p.PLANTS, 80)
    a = rng.random((320, 3)) < .3
    h = rng.random((320, 3)) < .1
    counts, ap, hp, ac, hc = p.calibration_bootstrap(plants, a, h, 4096, rng)
    for mechanism in p.MECHANISMS:
        z = p.added_harms(counts, ap, hp, hc, 0., mechanism, rng)
        np.testing.assert_array_equal(z, hc)
        z = p.added_harms(counts, ap, hp, hc, 1., mechanism, rng)
        np.testing.assert_array_equal(z, ac)
        assert np.all(hc <= ac)


def test_shared_shocks_preserve_model_identity_and_independent_shocks_do_not_force_it():
    rng = np.random.default_rng(82)
    plants = np.repeat(p.PLANTS, 80)
    a = np.repeat((np.arange(320) % 3 == 0)[:, None], 3, axis=1)
    counts, ap, hp, ac, hc = p.calibration_bootstrap(plants, a, np.zeros_like(a), 4096, rng)
    shared = p.added_harms(counts, ap, hp, hc, .005, 'shared', rng)
    np.testing.assert_array_equal(shared[..., 0], shared[..., 1])
    independent = p.added_harms(counts, ap, hp, hc, .005, 'independent', rng)
    assert np.any(independent[..., 0] != independent[..., 1])
