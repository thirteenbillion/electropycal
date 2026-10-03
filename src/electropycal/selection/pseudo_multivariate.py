"""Pseudo-multivariate feature selectors: SR, sMC and VIP-style scores.

VIP and Selectivity Ratio (SR) score each feature from a fitted PLSR model, then a
threshold prunes the featureset.
"""

from __future__ import annotations

import numpy as np

from ..models.variants import LinearPLSR


def vip_scores(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    return LinearPLSR(k=k).fit(X, y).vip()


def selectivity_ratio(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """SR: explained/residual variance of each feature on the target-projection.

    Projects samples onto the model's regression direction, then per feature takes
    ``SS_explained / SS_residual`` of that feature regressed on the projection
    Smoother / less noise-sensitive than VIP.
    """
    X = np.asarray(X, float)
    model = LinearPLSR(k=k).fit(X, y)
    b = model.coef.ravel()
    nb = np.linalg.norm(b)
    if nb == 0:
        return np.zeros(X.shape[1])
    t = (X - model.x_mean) @ (b / nb)                 # target-projected scores
    tt = float(t @ t)
    sr = np.empty(X.shape[1])
    Xc = X - model.x_mean
    for j in range(X.shape[1]):
        loading = (t @ Xc[:, j]) / tt if tt else 0.0
        expl = (loading * t)
        ss_exp = float(expl @ expl)
        ss_res = float(((Xc[:, j] - expl) ** 2).sum())
        sr[j] = ss_exp / ss_res if ss_res > 0 else np.inf
    return sr


def smc_scores(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """sMC: significance multivariate correlation, as an F statistic with (1, n-2) df.

    Like the selectivity ratio this is built on target projection, but the explained part of
    each feature uses the **normalized regression vector** ``b/||b||`` directly rather than
    the per-feature loading. That difference is the method: using ``b`` mixes predictive with
    orthogonal variation, which makes sMC sharper and noisier than SR. Computing the loading
    instead makes the two indices identical, which is what this function used to do.

    Returns ``(SS_explained / 1) / (SS_residual / (n - 2))``, so the value is an F statistic
    and a cutoff is a significance level. Use :func:`smc_significance` when you want a score
    whose threshold means the same thing across folds of different size.
    """
    X = np.asarray(X, float)
    n = X.shape[0]
    model = LinearPLSR(k=k).fit(X, y)
    b = model.coef.ravel()
    nb = np.linalg.norm(b)
    if nb == 0 or n <= 2:
        return np.zeros(X.shape[1])
    bn = b / nb
    Xc = X - model.x_mean
    t = Xc @ bn                                    # target-projected score
    Xhat = np.outer(t, bn)                         # explained part, via b_j not the loading
    ss_exp = (Xhat ** 2).sum(axis=0)
    ss_res = ((Xc - Xhat) ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        F = (ss_exp / 1.0) / (ss_res / (n - 2))
    return np.where(ss_res > 0, F, np.inf)


def smc_significance(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """sMC expressed as ``-log10(p)`` against F(1, n-2). Higher means more important.

    Selection needs a threshold that means the same thing on a 60-row fold and a 2000-row
    one. A raw F does not: its critical value moves with the sample size. This converts to a
    significance scale, so a threshold of 2 is "p <= 0.01" everywhere, which is what
    :func:`select` uses for ``method="smc"``.
    """
    F = smc_scores(X, y, k)
    n = np.asarray(X).shape[0]
    if n <= 2:
        return np.zeros_like(F)
    from scipy.stats import f as _f
    with np.errstate(divide="ignore"):
        p = _f.sf(np.where(np.isfinite(F), F, np.finfo(float).max), 1, n - 2)
        out = -np.log10(np.clip(p, 1e-300, 1.0))
    return np.nan_to_num(out, nan=0.0, posinf=300.0)


def select(method: str, X: np.ndarray, y: np.ndarray, k: int, threshold: float) -> np.ndarray:
    """Return indices whose score >= ``threshold`` for method in {'vip', 'sr', 'smc'}.

    Threshold scales differ by method and are not interchangeable. ``vip`` and ``sr`` are
    scored on their own natural scale, where 1.0 is the conventional cutoff. ``smc`` is
    scored as ``-log10(p)``, so 2.0 means p <= 0.01. See :func:`smc_significance`.

    Zero-variance columns (e.g. a feature that is constant within this train fold) are scored 0 and
    excluded from the internal PLSR fit — a constant column makes sklearn's PLS standardization divide
    by a zero std and emit NaN loadings. ``k`` is capped to the number of live columns so PLS always fits.
    """
    X = np.asarray(X, float)
    live = X.var(axis=0) > 0
    scores = np.zeros(X.shape[1])
    n_live = int(live.sum())
    if n_live:
        kk = max(1, min(int(k), n_live))
        # "smc" scores on the -log10(p) significance scale, not the raw F: an F threshold
        # would mean a different significance level on every fold, since folds differ in n.
        _fn = {"vip": vip_scores, "sr": selectivity_ratio, "smc": smc_significance}[method]
        scores[live] = _fn(X[:, live], y, kk)
    idx = np.where(scores >= threshold)[0]
    return idx if idx.size else np.array([int(np.argmax(scores))])
