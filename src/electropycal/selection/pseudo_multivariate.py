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
    """Significance Multivariate Correlation: direct OLS of each feature on the
    target-projection score (with intercept). Maximizes X–Y correlation — sharper
    than SR but more noise-sensitive."""
    X = np.asarray(X, float)
    model = LinearPLSR(k=k).fit(X, y)
    b = model.coef.ravel()
    nb = np.linalg.norm(b)
    if nb == 0:
        return np.zeros(X.shape[1])
    t = (X - model.x_mean) @ (b / nb)
    A = np.column_stack([t, np.ones_like(t)])            # OLS with intercept
    smc = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        beta, *_ = np.linalg.lstsq(A, X[:, j], rcond=None)
        fit = A @ beta
        ss_exp = float(((fit - X[:, j].mean()) ** 2).sum())
        ss_res = float(((X[:, j] - fit) ** 2).sum())
        smc[j] = ss_exp / ss_res if ss_res > 0 else np.inf
    return smc


def select(method: str, X: np.ndarray, y: np.ndarray, k: int, threshold: float) -> np.ndarray:
    """Return indices whose score ≥ ``threshold`` for method in {'vip', 'sr', 'smc'}.

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
        scores[live] = {"vip": vip_scores, "sr": selectivity_ratio, "smc": smc_scores}[method](
            X[:, live], y, kk)
    idx = np.where(scores >= threshold)[0]
    return idx if idx.size else np.array([int(np.argmax(scores))])
