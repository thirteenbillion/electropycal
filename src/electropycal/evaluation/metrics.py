"""Pooled error metrics for the CV tracks.

Everything is built from three per-fold quantities: ``sse``, ``n_test``, and
``tss``. Pooled metrics are ``sqrt(ΣSSE/ΣN)`` and ``1 − ΣSSE/ΣTSS`` — the RMSE/Q²
over the *concatenation* of held-out residuals (a sample-weighted / micro
average), which correctly handles unequal fold sizes. Macro variants weight each
channel equally instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

#: Ridge penalties searched by :func:`fit_ridge` when no explicit ``alpha`` is given. The range is
#: deliberately wide and reaches far above sklearn's defaults: the recalibration problem is
#: p ≫ n (~145 leakage-safe predictors against a few dozen training rows in the early forward-chained
#: folds), so the useful penalties live around 1e3–1e5, and anything near 1.0 interpolates the
#: training set and extrapolates wildly out of sample.
ALPHA_GRID: tuple[float, ...] = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6)


def fit_ridge(X, y, alpha: float | None = None, alphas: Sequence[float] = ALPHA_GRID):
    """Fit a ridge, choosing ``alpha`` by inner CV when it is not supplied.

    ``alpha=None`` (the default for the E6/E8 estimators) runs :class:`~sklearn.linear_model.RidgeCV`
    over ``alphas`` using leave-one-out generalized CV **on the training fold only**, so the penalty
    is selected without ever seeing held-out data. Passing a float pins the penalty instead, which is
    only useful for reproducing a specific historical run — a fixed small penalty on this problem
    produces Q² ≈ −45 and an RMSEP ~7× the naive mean.

    Falls back to the largest grid value if the inner CV cannot run (e.g. a degenerate fold).
    """
    from sklearn.linear_model import Ridge, RidgeCV

    if alpha is not None:
        return Ridge(alpha=alpha).fit(X, y)
    try:
        return RidgeCV(alphas=np.asarray(alphas, float)).fit(X, y)
    except Exception:
        return Ridge(alpha=max(alphas)).fit(X, y)


@dataclass(frozen=True)
class FoldResult:
    """Held-out error summary for a single CV fold.

    Attributes mirror the per-fold serialization written to ``metrics.json``.
    ``channel`` / ``t_test`` identify the fold so results can be grouped for
    Track 1 (per channel) vs Track 2/3 (pooled).
    """

    channel: int
    t_test: object
    sse: float          # Σ_i (y_i − ŷ_i)^2 over held-out rows
    n_test: int         # number of held-out rows
    tss: float          # Σ_i (y_i − mean_train)^2 over held-out rows

    @classmethod
    def from_predictions(
        cls,
        channel: int,
        t_test: object,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        mean_train: float,
    ) -> "FoldResult":
        """Build a fold result from held-out predictions (linear-scale units).

        ``mean_train`` is the mean response over the fold's *training* rows — the
        honest "predict-the-mean" baseline for Q². Back-transform log-model
        predictions to linear units *before* calling this.
        """
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape")
        resid = y_true - y_pred
        return cls(
            channel=channel,
            t_test=t_test,
            sse=float(resid @ resid),
            n_test=int(y_true.size),
            tss=float(((y_true - mean_train) ** 2).sum()),
        )


def pooled_rmsep(folds: Iterable[FoldResult]) -> float:
    """Sample-weighted (micro) pooled RMSEP = ``sqrt(ΣSSE/ΣN_test)``."""
    folds = list(folds)
    n = sum(f.n_test for f in folds)
    if n == 0:
        return float("nan")
    return float(np.sqrt(sum(f.sse for f in folds) / n))


def pooled_q2(folds: Iterable[FoldResult]) -> float:
    """Micro-pooled predictive Q² = ``1 − ΣSSE/ΣTSS``."""
    folds = list(folds)
    tss = sum(f.tss for f in folds)
    if tss == 0:
        return float("nan")
    return float(1.0 - sum(f.sse for f in folds) / tss)


def macro_rmsep(folds: Iterable[FoldResult]) -> float:
    """Macro-averaged RMSEP: mean of per-channel pooled RMSEPs (each channel
    weighted equally, regardless of sample count)."""
    folds = list(folds)
    by_channel: dict[int, list[FoldResult]] = {}
    for f in folds:
        by_channel.setdefault(f.channel, []).append(f)
    if not by_channel:
        return float("nan")
    return float(np.mean([pooled_rmsep(fs) for fs in by_channel.values()]))


def bootstrap_rmsep_ci(
    folds: Sequence[FoldResult],
    n_boot: int = 2000,
    alpha: float = 0.05,
    random_state: int | None = 0,
) -> tuple[float, float]:
    """Bootstrap CI for the pooled RMSEP by resampling *folds* with replacement.

    Returns ``(lo, hi)`` percentile bounds. Resampling at the fold level (not the
    row level) is deliberate: the fold is the unit of uncertainty here, because rows
    within a fold share a sensor and a timepoint and so are not independent. With few
    folds the interval comes out wide — which is the honest result, not a defect.
    """
    folds = list(folds)
    if len(folds) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(random_state)
    idx = np.arange(len(folds))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        sample = [folds[i] for i in rng.choice(idx, size=len(folds), replace=True)]
        boots[b] = pooled_rmsep(sample)
    lo = float(np.nanpercentile(boots, 100 * alpha / 2))
    hi = float(np.nanpercentile(boots, 100 * (1 - alpha / 2)))
    return (lo, hi)


def fold_spread(folds: Sequence[FoldResult]) -> dict[str, float]:
    """Per-fold RMSEP spread (SD / IQR) as a cheap uncertainty summary."""
    per_fold = np.array(
        [np.sqrt(f.sse / f.n_test) if f.n_test else np.nan for f in folds]
    )
    per_fold = per_fold[~np.isnan(per_fold)]
    if per_fold.size == 0:
        return {"sd": float("nan"), "iqr": float("nan")}
    q1, q3 = np.percentile(per_fold, [25, 75])
    return {"sd": float(per_fold.std(ddof=1)) if per_fold.size > 1 else 0.0,
            "iqr": float(q3 - q1)}


def aggregate(folds: Sequence[FoldResult], with_ci: bool = True) -> dict:
    """Full aggregated summary for a testing condition (→ ``aggregated_metrics.json``)."""
    out: dict = {
        "pooled_rmsep": pooled_rmsep(folds),
        "pooled_q2": pooled_q2(folds),
        "macro_rmsep": macro_rmsep(folds),
        "n_folds": len(folds),
        "n_test_total": int(sum(f.n_test for f in folds)),
        **{f"fold_{k}": v for k, v in fold_spread(folds).items()},
    }
    if with_ci:
        lo, hi = bootstrap_rmsep_ci(folds)
        out["rmsep_ci_lo"], out["rmsep_ci_hi"] = lo, hi
    return out
