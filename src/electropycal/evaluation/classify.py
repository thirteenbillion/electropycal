"""Drift-beyond-trust classifier (study experiment E5) — the pragmatic pivot.

Regressing the exact drifted sensitivity from leakage-safe electrode state did not generalize
out-of-sample (E0-E4). A coarser, genuinely-useful question may still be answerable: **can the
electrode state flag when a sensor has drifted past the point of trusting its frozen calibration?**
That is a binary classification (usable / degraded), a much easier target than the exact value, and
it is what a deployment actually needs (know when to stop trusting a channel).

Label: per sensor, take the baseline (earliest-timepoint) sensitivity; a later ``(sensor, timepoint)``
is **degraded** if its sensitivity has moved more than ``frac`` (fractional change) from that baseline
— drift in *either* direction breaks a frozen calibration. Predictors: the same D0-normalized EIS +
background features. Evaluation: forward-chained (train on earlier timepoints), pooled ROC-AUC +
balanced accuracy — deployment-faithful, same as the regression tracks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def degraded_labels(sens_df: pd.DataFrame, value_col: str = "sensitivity",
                    frac: float = 0.5) -> pd.Series:
    """Per ``(device, channel, timepoint)`` boolean: has ``value_col`` moved > ``frac`` (fractional,
    absolute) from that sensor's baseline (earliest-timepoint) value? NaN where no positive baseline."""
    id_cols = [c for c in ("device", "channel") if c in sens_df.columns]
    df = sens_df.copy()
    df = df.sort_values(id_cols + ["timepoint"])
    base = (df.groupby(id_cols)[value_col].transform("first"))
    rel = (df[value_col] - base).abs() / base.abs()
    lab = (rel > frac).astype("float")
    lab[~np.isfinite(base) | (base.abs() < 1e-12)] = np.nan
    return lab


def drift_classifier_cv(sens_df: pd.DataFrame, value_col: str = "sensitivity", frac: float = 0.5,
                        min_train_times: int = 3, C: float = 0.5) -> dict:
    """Forward-chained logistic-regression classifier of the degraded label from electrode state.

    ``sens_df`` is a per-sensor-timepoint frame (e.g. from ``sensitivity_featureset``): numeric
    non-reserved columns are predictors. For each test timepoint (after ``min_train_times``), train an
    L2 logistic regression on all earlier timepoints and score the held-out timepoint. Returns pooled
    ``roc_auc``, ``balanced_accuracy``, ``base_rate`` (degraded prevalence), ``n_folds``, ``n_test``.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    from ..data.schema import RESERVED_COLUMNS
    df = sens_df.copy()
    df["_deg"] = degraded_labels(df, value_col, frac)
    df = df[np.isfinite(df["_deg"])]
    feat_cols = [c for c in df.columns if c not in set(RESERVED_COLUMNS) and c not in ("_deg", "time_index")
                 and pd.api.types.is_numeric_dtype(df[c])]
    tp = df["timepoint"].to_numpy(float)
    times = np.unique(tp)
    X = np.array(df[feat_cols].to_numpy(float), dtype=float, copy=True)
    X[~np.isfinite(X)] = np.nan
    y = df["_deg"].to_numpy(int)

    y_true, y_prob, y_pred = [], [], []
    n_folds = 0
    for ti, t_test in enumerate(times):
        if ti < min_train_times:
            continue
        te = np.where(tp == t_test)[0]
        tr = np.where(tp < t_test)[0]
        if te.size == 0 or tr.size < 8 or np.unique(y[tr]).size < 2:   # need both classes to train
            continue
        mu = np.nanmean(X[tr], 0); sd = np.nanstd(X[tr], 0); sd = np.where(sd > 0, sd, 1.0)
        def z(a):
            b = (a - mu) / sd; b[~np.isfinite(b)] = 0.0; return b
        clf = LogisticRegression(C=C, max_iter=1000, class_weight="balanced")   # L2 (default)
        try:
            clf.fit(z(X[tr]), y[tr])
            p = clf.predict_proba(z(X[te]))[:, 1]
        except Exception:
            continue
        y_true.extend(y[te].tolist()); y_prob.extend(p.tolist()); y_pred.extend((p >= 0.5).astype(int).tolist())
        n_folds += 1

    y_true = np.asarray(y_true)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return {"roc_auc": float("nan"), "balanced_accuracy": float("nan"),
                "base_rate": float(np.mean(y_true)) if y_true.size else float("nan"),
                "n_folds": n_folds, "n_test": int(y_true.size)}
    return {"roc_auc": float(roc_auc_score(y_true, y_prob)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, np.asarray(y_pred))),
            "base_rate": float(np.mean(y_true)), "n_folds": n_folds, "n_test": int(y_true.size)}
