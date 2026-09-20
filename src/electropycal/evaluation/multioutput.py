"""Multi-output (PLS2) dose-curve-vector experiment (study experiment E3).

Predicts the **whole dose-response curve** — ``NormIpeak`` at every concentration, as a *vector* — from
the dose-invariant electrode-state features via multivariate PLS (**PLS2**), under forward-chained CV.
This sidesteps the scalar curve-fit misspecification entirely (no Langmuir/Hill, no exploding ``Kd``):
the target is the measured curve itself. It does **not** change the fold count (still set by the
channel×timepoint splits), so it targets *skill/robustness*, not fold starvation. Classical multivariate
PLS — not deep learning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def dose_curve_matrix(df: pd.DataFrame, value_col: str = "NormIpeak",
                      conc_col: str = "concentration", min_conc: int = 3):
    """Collapse a per-dose featureset to one **curve vector** per ``(device, channel, timepoint)``.

    Returns ``(meta, X, Y, feat_names, concs)`` — ``meta`` (the id columns + ``timepoint``), the mean
    dose-invariant predictor matrix ``X``, the dose-vector target ``Y`` (one column per concentration),
    the feature names, and the sorted concentrations. Groups with fewer than ``min_conc`` measured
    concentrations are dropped; remaining missing curve entries are left as NaN (imputed per-fold).
    """
    from ..data.schema import RESERVED_COLUMNS
    id_cols = [c for c in ("device", "channel", "timepoint") if c in df.columns]
    if "channel" not in id_cols or "timepoint" not in id_cols:
        raise ValueError("dose_curve_matrix needs 'channel' and 'timepoint' columns")
    feat_cols = [c for c in df.columns if c not in set(RESERVED_COLUMNS) and c != "time_index"
                 and pd.api.types.is_numeric_dtype(df[c])]     # numeric predictors only
    concs = sorted(float(c) for c in pd.unique(df[conc_col].dropna()))
    meta_rows, Xr, Yr = [], [], []
    for key, g in df.groupby(id_cols):
        piv = g.groupby(conc_col)[value_col].mean()
        y = piv.reindex(concs).to_numpy(float)
        if np.isfinite(y).sum() < min_conc:
            continue
        meta_rows.append(dict(zip(id_cols, key if isinstance(key, tuple) else (key,))))
        Xr.append([float(g[c].mean()) for c in feat_cols])       # dose-invariant -> mean is exact
        Yr.append(y)
    meta = pd.DataFrame(meta_rows)
    return meta, np.asarray(Xr, float), np.asarray(Yr, float), feat_cols, concs


def _impute(mat: np.ndarray, col_fill: np.ndarray) -> np.ndarray:
    out = mat.copy()
    idx = np.where(~np.isfinite(out))
    out[idx] = np.take(col_fill, idx[1])
    return out


def pls2_forward_chained(df: pd.DataFrame, k_grid=(2, 3), min_train_times: int = 3,
                         value_col: str = "NormIpeak", conc_col: str = "concentration",
                         min_conc: int = 3) -> dict:
    """Forward-chained PLS2 CV over the dose-curve vector. Deployment-faithful (train on timepoints
    ``< t_test``), pooled globally.

    ``k`` (latent variables) is chosen per fold by a nested forward split (inner ``t_val`` = latest
    timepoint ``< t_test``) minimizing inner curve-RMSE. Returns a dict with ``pooled_q2`` (over the
    flattened curve, vs a train-mean-curve baseline), ``per_dose_q2`` (one Q² per concentration),
    ``n_folds``, ``n_test_curves``, and ``concs``.
    """
    from sklearn.cross_decomposition import PLSRegression

    meta, X, Y, feats, concs = dose_curve_matrix(df, value_col, conc_col, min_conc)
    if len(meta) == 0:
        return {"pooled_q2": float("nan"), "n_folds": 0, "n_test_curves": 0, "concs": concs}
    tp = meta["timepoint"].to_numpy(float)
    times = np.unique(tp)

    def _fit_predict(Xtr, Ytr, Xte, k):
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd = np.where(sd > 0, sd, 1.0)
        ycol = np.nanmean(Ytr, axis=0)
        Xtr_i = _impute((Xtr - mu) / sd, np.zeros(Xtr.shape[1]))
        Xte_i = _impute((Xte - mu) / sd, np.zeros(Xte.shape[1]))
        Ytr_i = _impute(Ytr, ycol)
        kk = int(min(k, Xtr_i.shape[1], max(1, Xtr_i.shape[0] - 1)))
        m = PLSRegression(n_components=kk, scale=False).fit(Xtr_i, Ytr_i)
        return m.predict(Xte_i), ycol

    y_true_all, y_pred_all, y_base_all = [], [], []
    n_folds = 0
    for ti, t_test in enumerate(times):
        if ti < min_train_times:
            continue
        te = np.where(tp == t_test)[0]
        tr = np.where(tp < t_test)[0]
        if te.size == 0 or tr.size < 2:
            continue
        # inner k-selection: forward split within the training set
        inner_times = np.unique(tp[tr])
        best_k = k_grid[0]
        vt = inner_times[inner_times < t_test]
        if vt.size:
            t_val = vt.max()
            itr = tr[tp[tr] < t_val]; iva = tr[tp[tr] == t_val]
            if itr.size >= 2 and iva.size:
                best, best_rmse = None, np.inf
                for k in k_grid:
                    try:
                        pred, _ = _fit_predict(X[itr], Y[itr], X[iva], k)
                    except Exception:
                        continue
                    rmse = float(np.sqrt(np.nanmean((_impute(Y[iva], np.nanmean(Y[itr], 0)) - pred) ** 2)))
                    if rmse < best_rmse:
                        best, best_rmse = k, rmse
                if best is not None:
                    best_k = best
        try:
            pred, ycol = _fit_predict(X[tr], Y[tr], X[te], best_k)
        except Exception:
            continue
        yt = _impute(Y[te], ycol)
        y_true_all.append(yt); y_pred_all.append(pred)
        y_base_all.append(np.repeat(ycol[None, :], te.size, axis=0))   # train-mean-curve baseline
        n_folds += 1

    if not y_true_all:
        return {"pooled_q2": float("nan"), "n_folds": 0, "n_test_curves": 0, "concs": concs}
    YT = np.vstack(y_true_all); YP = np.vstack(y_pred_all); YB = np.vstack(y_base_all)

    def _q2(t, p, b):
        sse = float(np.sum((t - p) ** 2)); sst = float(np.sum((t - b) ** 2))
        return 1.0 - sse / sst if sst > 0 else float("nan")

    per_dose = {float(c): _q2(YT[:, j], YP[:, j], YB[:, j]) for j, c in enumerate(concs)}
    return {"pooled_q2": _q2(YT, YP, YB), "per_dose_q2": per_dose,
            "n_folds": n_folds, "n_test_curves": int(YT.shape[0]), "concs": concs}
