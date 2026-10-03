"""Unit coverage for the PLS-importance filters in ``selection.pseudo_multivariate``.

These three scores had no direct tests, which is how the selectivity-ratio and sMC
duplication went unnoticed: both conditions ran, both produced plausible numbers, and
nothing compared them.
"""

import numpy as np
import pytest

from electropycal.selection import pseudo_multivariate as pm


def _xy(n=120, p=25, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = 2.0 * X[:, 0] - 1.5 * X[:, 3] + rng.normal(scale=0.3, size=n)
    return X, y


@pytest.mark.parametrize("method", ["vip", "sr", "smc"])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_scores_are_finite_and_correctly_shaped(method, k):
    X, y = _xy()
    fn = {"vip": pm.vip_scores, "sr": pm.selectivity_ratio, "smc": pm.smc_scores}[method]
    s = fn(X, y, k)
    assert s.shape == (X.shape[1],)
    assert np.all(np.isfinite(s))
    assert np.all(s >= 0)


@pytest.mark.parametrize("method", ["vip", "sr", "smc"])
def test_select_respects_the_threshold_and_is_never_empty(method):
    X, y = _xy()
    hi = pm.select(method, X, y, k=2, threshold=1.0)
    lo = pm.select(method, X, y, k=2, threshold=0.1)
    assert hi.size >= 1 and lo.size >= 1          # falls back to argmax rather than empty
    assert set(hi) <= set(lo)                     # a higher cutoff keeps a subset
    assert lo.size <= X.shape[1]


def test_zero_variance_columns_are_excluded_not_fatal():
    """A column constant within a fold makes PLS standardization divide by zero."""
    X, y = _xy()
    X[:, 5] = 3.0
    for method in ("vip", "sr", "smc"):
        idx = pm.select(method, X, y, k=2, threshold=0.5)
        assert 5 not in set(idx)


def test_vip_and_selectivity_ratio_rank_features_differently():
    """Control for the xfail below: the harness can detect two filters that do differ.

    If this ever fails, the comparison itself is broken and the xfail means nothing.
    """
    X, y = _xy(n=300, p=40)
    def rank(a):
        return a.argsort().argsort()
    vip = rank(pm.vip_scores(X, y, 2))
    sr = rank(pm.selectivity_ratio(X, y, 2))
    assert not np.array_equal(vip, sr)
    assert np.corrcoef(vip, sr)[0, 1] < 0.999


def test_smc_is_distinct_from_selectivity_ratio():
    """The two filters must not collapse onto each other.

    They did: smc_scores used to regress each feature on the target projection with an
    intercept, which recovers the loading and so recomputes the selectivity ratio. sMC is
    defined to use the normalized regression vector directly.
    """
    X, y = _xy(n=300, p=40)
    sr = pm.selectivity_ratio(X, y, 2)
    smc = pm.smc_scores(X, y, 2)
    assert not np.allclose(sr, smc, rtol=1e-6)

    def rank(a):
        return a.argsort().argsort()
    assert np.corrcoef(rank(sr), rank(smc))[0, 1] < 0.95


def test_smc_is_an_f_statistic_scaling_with_sample_size():
    """sMC is (SS_exp / 1) / (SS_res / (n - 2)), so it grows with n for a real effect.

    Measured on the MAXIMUM, not the median: most features here are noise with a
    near-zero association, and their F stays near zero at any n. Only genuinely
    associated features accumulate evidence as rows are added, which is the behaviour
    that distinguishes an F statistic from the bare variance ratio this used to return.
    """
    small = pm.smc_scores(*_xy(n=60, p=12, seed=1), 2)
    large = pm.smc_scores(*_xy(n=600, p=12, seed=1), 2)
    assert np.all(small >= 0) and np.all(large >= 0)
    assert large.max() > 3 * small.max()


def test_smc_significance_is_monotone_in_the_f_statistic():
    """Selection uses -log10(p), which must preserve the F ordering."""
    X, y = _xy(n=200, p=30)
    f = pm.smc_scores(X, y, 2)
    s = pm.smc_significance(X, y, 2)
    finite = np.isfinite(f) & np.isfinite(s)

    def rank(a):
        return a.argsort().argsort()
    assert np.corrcoef(rank(f[finite]), rank(s[finite]))[0, 1] > 0.999


def test_smc_threshold_means_the_same_significance_at_any_fold_size():
    """A raw F cutoff would drift with n; a -log10(p) cutoff does not.

    At threshold 2.0 every selected feature must satisfy p <= 0.01 on both a small and a
    large fold, which is the property that makes the threshold portable across folds.
    """
    from scipy.stats import f as fdist
    for n in (60, 400):
        X, y = _xy(n=n, p=25, seed=2)
        idx = pm.select("smc", X, y, k=2, threshold=2.0)
        p_vals = fdist.sf(pm.smc_scores(X, y, 2), 1, n - 2)
        assert np.all(p_vals[idx] <= 0.01 + 1e-12)


def test_the_three_filters_are_mutually_distinct():
    """Batch 3 is a designed contrast, so no two of its filters may coincide."""
    X, y = _xy(n=300, p=40)

    def rank(a):
        return a.argsort().argsort()
    sr = rank(pm.selectivity_ratio(X, y, 2))
    vip = rank(pm.vip_scores(X, y, 2))
    smc = rank(pm.smc_scores(X, y, 2))
    for a, b, names in ((sr, vip, "sr/vip"), (sr, smc, "sr/smc"), (vip, smc, "vip/smc")):
        assert np.corrcoef(a, b)[0, 1] < 0.99, f"{names} rank-correlate above 0.99"
