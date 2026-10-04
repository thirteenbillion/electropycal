"""Normalization: per-sensor D0 referencing, then leakage-safe scaling.

D0-normalization: additive shift for bounded/phase/log-slope features, division
for magnitude/area-scaling features (schema.d0_normalization_kind). Z-scoring is
sample-dependent and MUST be computed on the training split only, then frozen and
applied to test (leakage safety). Log tables subtract the log-scaled D0.
"""
from __future__ import annotations

import numpy as np

from ..data.schema import d0_normalization_kind, is_log_transformable


def to_log_representation(X: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """Turn a **linear** D0-normalized matrix into the **log-model** representation.

    The linear D0-normalization already gives multiplicative (log-transformable) features as the
    ratio ``x/d0`` and additive features as ``x − d0``. The log model differs *only* in the
    multiplicative features, which become ``log(x/d0)``, i.e. ``log`` of those already-normalized
    ratio columns; additive features are left unchanged. This is identical to
    ``d0_normalize(..., log=True)`` applied to the raw features, without needing them.

    Multiplicative D0-normalized features are positive magnitude/area ratios; any non-positive value
    (shouldn't occur) is floored to a tiny positive before the ``log`` so the matrix stays finite.
    """
    X = np.asarray(X, float).copy()
    for j, name in enumerate(feature_names):
        if is_log_transformable(name):
            X[:, j] = np.log(np.maximum(X[:, j], 1e-12))
    return X


def d0_normalize(X: np.ndarray, feature_names: list[str], d0_row: np.ndarray,
                 log: bool = False) -> np.ndarray:
    """D0-normalize each feature per its type.

    ``d0_row`` is the per-feature baseline (in-vitro D0, or the early in-vivo
    baseline during deployment). Additive/bounded features are shifted
    (``x − d0``, 0.0-centered); magnitude/area-scaling features are divided
    (``x / d0``, 1.0-centered). With ``log=True`` (the log-model table), only the
    log-transformable (multiplicative) features become ``log(x) − log(d0)`` =
    ``log(x/d0)``; bounded features stay additive in raw units.
    """
    X = np.asarray(X, dtype=float).copy()
    d0 = np.asarray(d0_row, dtype=float)
    for j, name in enumerate(feature_names):
        if d0_normalization_kind(name) == "additive":
            X[:, j] = X[:, j] - d0[j]                      # 0.0-centered (both tables)
        elif log and is_log_transformable(name):
            with np.errstate(invalid="ignore", divide="ignore"):
                X[:, j] = np.log(X[:, j]) - np.log(d0[j])  # log(x/d0)
        else:
            X[:, j] = X[:, j] / d0[j] if d0[j] != 0 else np.nan  # 1.0-centered
    return X


def d0_group_key(key) -> str:
    """Stable string id for a ``d0_normalize_frame`` group, e.g. ``("3-2", 5)`` → ``"3-2|5"``.

    Used as the key when per-sensor baselines are recorded to (or replayed from) an
    extraction pin, so the mapping survives a JSON round-trip.
    """
    return "|".join(str(k) for k in (key if isinstance(key, tuple) else (key,)))


def d0_normalize_frame(df, feature_cols: list[str], group=("device", "channel"),
                       time_col: str = "timepoint", log: bool = False,
                       d0_rows: dict | None = None, return_d0_rows: bool = False):
    """D0-normalize a featureset DataFrame **per sensor** against its own earliest
    timepoint (see DESIGN §9 step 2).

    For each ``group`` (default ``(device, channel)``) the baseline ``d0_row`` is the
    mean feature vector at that sensor's minimum ``time_col``: the in-vitro D0 or,
    in vivo, the early in-vivo baseline. Returns a copy with ``feature_cols``
    normalized; non-feature columns are untouched.

    That baseline is derived from *whichever timepoints are present*, so extracting over a
    subset that omits a sensor's true D0 session silently re-baselines it onto a later
    timepoint. ``d0_rows`` (``{``:func:`d0_group_key`\\ ``: vector}``, in ``feature_cols``
    order) supplies recorded baselines instead; a group absent from it falls back to its
    own earliest timepoint. ``return_d0_rows=True`` additionally returns the baselines
    actually used, for recording into an extraction pin.
    """
    out = df.copy()
    names = list(feature_cols)
    used: dict[str, np.ndarray] = {}
    for key, g in df.groupby(list(group)):
        gk = d0_group_key(key)
        pinned = None if d0_rows is None else d0_rows.get(gk)
        d0_row = (np.asarray(pinned, float) if pinned is not None else
                  g.loc[g[time_col] == g[time_col].min(), names].mean().to_numpy(float))
        if d0_row.shape != (len(names),):
            raise ValueError(f"d0_row for {gk!r} has {d0_row.shape} values, expected "
                             f"{len(names)} (one per feature column)")
        used[gk] = d0_row
        out.loc[g.index, names] = d0_normalize(g[names].to_numpy(float), names, d0_row, log=log)
    return (out, used) if return_d0_rows else out


def fit_median_impute(X_train: np.ndarray) -> np.ndarray:
    """Per-column median of ``X_train``, for imputing missing cells.

    **Fit on the training split only, then apply to held-out rows** with
    :func:`apply_median_impute`: the same discipline as :func:`fit_zscore` and the ridge
    alpha selection. Computing the median over train and test together would let held-out
    rows influence the values the model is fitted on.

    A column with no finite training value has no defined median (all-NaN within this fold,
    even if present elsewhere in the corpus); it falls back to ``0.0`` rather than NaN, so a
    fold never fails on a feature that happens to be empty in its training rows.
    """
    import warnings
    X_train = np.asarray(X_train, dtype=float)
    with warnings.catch_warnings():
        # an all-NaN column is expected and handled below, not a condition to warn about
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(np.where(np.isfinite(X_train), X_train, np.nan), axis=0)
    med = np.asarray(med, dtype=float)
    med[~np.isfinite(med)] = 0.0
    return med


def apply_median_impute(X: np.ndarray, col_median: np.ndarray) -> np.ndarray:
    """Fill non-finite cells of ``X`` from ``col_median`` (from :func:`fit_median_impute`).

    Returns a copy; ``X`` is untouched. Non-finite covers NaN and ±inf, so a divide-by-zero
    upstream is imputed rather than propagated into the fit.
    """
    X = np.array(np.asarray(X, dtype=float), copy=True)
    bad = ~np.isfinite(X)
    if bad.any():
        X[bad] = np.take(np.asarray(col_median, dtype=float), np.where(bad)[1])
    return X


def fit_zscore(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) from the training block, to freeze and reuse on test.

    Z-scoring statistics are sample-dependent, so computing them over all data would leak test
    information into training. Fit here on the training split only, then apply with
    :func:`apply_zscore`.
    """
    mean = X_train.mean(0)
    std = X_train.std(0, ddof=0)
    std[std == 0] = 1.0
    return mean, std


def apply_zscore(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def fit_robust(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (median, IQR) for robust scaling (preferred in vivo).

    ``X_robust = (X − median) / IQR`` resists outlier spikes from biological
    events / motion artifacts that standard Z-scoring would over-compress.
    """
    center = np.median(X_train, axis=0)
    q1, q3 = np.percentile(X_train, [25, 75], axis=0)
    scale = q3 - q1
    scale[scale == 0] = 1.0
    return center, scale


def apply_robust(X: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (X - center) / scale
