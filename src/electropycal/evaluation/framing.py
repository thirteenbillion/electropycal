"""Head-to-head comparison of discovery target framings on the SAME folds.

Compares, on identical forward-chained-over-timepoints splits:
  - ``current``     : state features -> per-dose NormIpeak (the dose-invariant default)
  - ``sensitivity`` : state -> per-(sensor,timepoint) [slope, intercept] of NormIpeak vs log10[DA]
  - ``sensitivity_quadratic`` : state -> [a, b, c] of a+b·log10[DA]+c·log10[DA]² (adds shape/curvature)
  - ``interaction`` : state + log_conc + state x log_conc -> NormIpeak (optional; overfits when P>N)

on two metrics: the common **per-dose NormIpeak RMSEP** (``sensitivity`` reconstructs NormIpeak from
its predicted slope+intercept), and **concentration recovery** (invert each calibration to estimate
log10[DA] from the measured NormIpeak — the actual recalibration use). ``current`` is flat, so it has
no calibration slope and cannot recover concentration.

This is a **light, self-contained diagnostic** (plain sklearn PLS, no feature selection / bootstrap),
meant to help pick a framing — not to reproduce a full discovery run's numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..discovery.config import RESERVED_COLUMNS, effective_min_train_times


def _state_cols(df: pd.DataFrame) -> list[str]:
    """Predictor columns, minus any that are entirely undefined.

    Mirrors ``RunData.from_frame``: a feature can be all-NaN because it is undefined for the
    chosen ``band`` (``ideality_C_band_HF`` / ``n_band_HF`` at band=(2,2000)), and PLS cannot
    fit a NaN column. Without this, ``compare_target_framings`` raises on the project's own
    featureset while passing on synthetic frames that happen to have no empty columns.
    """
    cols = [c for c in df.columns
            if c not in set(RESERVED_COLUMNS) and c != "time_index" and not c.startswith("_")]
    return [c for c in cols if not df[c].isna().all()]


def _zfit(X):
    """Train-only Z-score statistics, NaN-tolerant.

    ``nanmean``/``nanstd`` so a sparse missing cell does not poison the whole column; apply
    with :func:`_zapply`, which sends anything still non-finite to 0.0 — i.e. to the training
    mean. Same idiom as ``evaluation.hierarchical`` / ``stratify`` / ``classify``.
    """
    m, s = np.nanmean(X, 0), np.nanstd(X, 0)
    s = np.where((s == 0) | ~np.isfinite(s), 1.0, s)
    return np.where(np.isfinite(m), m, 0.0), s


def _zapply(X, m, s):
    z = (np.asarray(X, float) - m) / s
    z[~np.isfinite(z)] = 0.0
    return z


def _pls(Xtr, Ytr, Xte, k):
    from sklearn.cross_decomposition import PLSRegression
    k = int(max(1, min(k, Xtr.shape[1], Xtr.shape[0] - 1)))
    return PLSRegression(n_components=k, scale=False).fit(Xtr, Ytr).predict(Xte)


def _invert_quad(a, b, c, y, x_ref):
    """Solve ``a + b·x + c·x² = y`` for x (log-conc); pick the root nearest ``median(x_ref)``.

    ``x_ref`` must be a reference available **at prediction time** — the training dose grid.
    Passing the held-out group's own true log-concentrations (as this did) uses the quantity
    being recovered to choose between the two roots, which flatters
    ``conc_recovery_rmse_log10``. The dose grid a calibration was built over is legitimate
    design information and is what a real deployment has.
    """
    y = np.asarray(y, float)
    if abs(c) < 1e-12:
        return np.where(abs(b) > 1e-12, (y - a) / b, np.nan)
    disc = b * b - 4 * c * (a - y)
    ok = disc >= 0
    sq = np.sqrt(np.where(ok, disc, 0.0))
    r1, r2 = (-b + sq) / (2 * c), (-b - sq) / (2 * c)
    ref = float(np.median(x_ref))
    pick = np.where(np.abs(r1 - ref) <= np.abs(r2 - ref), r1, r2)
    return np.where(ok, pick, np.nan)


def compare_target_framings(df: pd.DataFrame, value_col: str = "NormIpeak",
                            conc_col: str = "concentration", k: int = 3,
                            min_train_times: int = 3, include_interaction: bool = True,
                            include_quadratic: bool = True, min_conc: int = 3,
                            min_conc_quad: int = 4) -> pd.DataFrame:
    """Return a comparison DataFrame (one row per framing).

    Columns: ``framing, normipeak_rmsep, conc_recovery_rmse_log10, conc_recovery_frac,
    n_folds, n_test``. Folds are forward-chained over timepoints (train on earlier, test on
    each later timepoint), with ``min_train_times`` capped to what the data supports.
    """
    df = df.copy()
    df["_sensor"] = (df["device"].astype(str) + ":" + df["channel"].astype(str)
                     if "device" in df.columns else df["channel"].astype(str))
    df["_lc"] = np.log10(df[conc_col].to_numpy(float))
    state = _state_cols(df)
    tps = sorted(df["timepoint"].unique())
    mtt = effective_min_train_times(min_train_times, len(tps))

    cur, sens, inter, quad = [], [], [], []   # (y_true, y_pred) NormIpeak per framing
    rec_s, rec_i, rec_q = [], [], []          # (log_conc_true, log_conc_hat)
    n_folds = 0
    for ti, t in enumerate(tps):
        if ti < mtt:
            continue                       # need >= mtt earlier timepoints
        tr, te = df[df["timepoint"] < t], df[df["timepoint"] == t]
        if len(tr) < 2 * k or te.empty:
            continue
        n_folds += 1
        Xtr, Xte = tr[state].to_numpy(float), te[state].to_numpy(float)
        m, s = _zfit(Xtr)
        Xtr, Xte = _zapply(Xtr, m, s), _zapply(Xte, m, s)
        ytr, yte = tr[value_col].to_numpy(float), te[value_col].to_numpy(float)
        lcte = te["_lc"].to_numpy(float)

        # current
        cur.append((yte, _pls(Xtr, ytr, Xte, k).ravel()))

        # sensitivity: per train group slope/intercept -> predict from state -> reconstruct
        Gx, Gy = [], []
        for _, g in tr.groupby(["_sensor", "timepoint"]):
            if g[conc_col].nunique() < min_conc:
                continue
            sl, ic = np.polyfit(g["_lc"].to_numpy(float), g[value_col].to_numpy(float), 1)
            Gx.append(g[state].iloc[0].to_numpy(float)); Gy.append([sl, ic])
        if len(Gx) >= 2:
            Gx = np.array(Gx); Gy = np.array(Gy)
            mg, sg = _zfit(Gx); Gxz = _zapply(Gx, mg, sg)
            for _, g in te.groupby(["_sensor", "timepoint"]):
                xs = _zapply(g[state].iloc[0].to_numpy(float)[None, :], mg, sg).ravel()
                si = _pls(Gxz, Gy, xs.reshape(1, -1), k).ravel()
                slp, itc = float(si[0]), float(si[1])
                x, y = g["_lc"].to_numpy(float), g[value_col].to_numpy(float)
                sens.append((y, itc + slp * x))
                rec_s.append((x, (y - itc) / slp if abs(slp) > 1e-12 else np.full_like(y, np.nan)))

        # sensitivity_quadratic: per train group [c,b,a] of a+b*lc+c*lc^2 -> predict -> reconstruct + invert
        if include_quadratic:
            Qx, Qy = [], []
            for _, g in tr.groupby(["_sensor", "timepoint"]):
                if g[conc_col].nunique() < min_conc_quad:
                    continue
                cf = np.polyfit(g["_lc"].to_numpy(float), g[value_col].to_numpy(float), 2)  # [c, b, a]
                Qx.append(g[state].iloc[0].to_numpy(float)); Qy.append([cf[0], cf[1], cf[2]])
            if len(Qx) >= 2:
                Qx = np.array(Qx); Qy = np.array(Qy)
                mq, sq = _zfit(Qx); Qxz = _zapply(Qx, mq, sq)
                # the TRAINING dose grid: known design information, available at deploy time.
                # Using the test group's own x here would pick the root with the answer.
                x_ref_tr = tr["_lc"].to_numpy(float)
                for _, g in te.groupby(["_sensor", "timepoint"]):
                    xs = _zapply(g[state].iloc[0].to_numpy(float)[None, :], mq, sq).ravel()
                    ci = _pls(Qxz, Qy, xs.reshape(1, -1), k).ravel()
                    cc_, bb_, aa_ = float(ci[0]), float(ci[1]), float(ci[2])
                    x, y = g["_lc"].to_numpy(float), g[value_col].to_numpy(float)
                    quad.append((y, aa_ + bb_ * x + cc_ * x ** 2))
                    rec_q.append((x, _invert_quad(aa_, bb_, cc_, y, x_ref_tr)))

        # interaction (optional)
        if include_interaction:
            lctr = tr["_lc"].to_numpy(float)
            XA_tr = np.column_stack([Xtr, lctr, Xtr * lctr[:, None]])
            from sklearn.cross_decomposition import PLSRegression
            kk = int(max(1, min(k, XA_tr.shape[1], XA_tr.shape[0] - 1)))
            mdl = PLSRegression(n_components=kk, scale=False).fit(XA_tr, ytr)
            z_lc = np.column_stack([Xte, lcte, Xte * lcte[:, None]])
            inter.append((yte, mdl.predict(z_lc).ravel()))
            b0 = mdl.predict(np.column_stack([Xte, np.zeros(len(te)), Xte * 0.0])).ravel()
            b1 = mdl.predict(np.column_stack([Xte, np.ones(len(te)), Xte * 1.0])).ravel()
            slope = b1 - b0
            rec_i.append((lcte, np.where(np.abs(slope) > 1e-12, (yte - b0) / slope, np.nan)))

    def _rmse(pairs):
        if not pairs:
            return np.nan, 0
        yt = np.concatenate([a for a, _ in pairs]); yp = np.concatenate([b for _, b in pairs])
        return float(np.sqrt(np.mean((yt - yp) ** 2))), int(yt.size)

    def _rec(pairs):
        if not pairs:
            return np.nan, np.nan
        xt = np.concatenate([a for a, _ in pairs]); xh = np.concatenate([b for _, b in pairs])
        ok = np.isfinite(xh)
        if not ok.any():
            return np.nan, 0.0
        return float(np.sqrt(np.mean((xt[ok] - xh[ok]) ** 2))), float(ok.mean())

    out = []
    r_cur, n_cur = _rmse(cur)
    out.append({"framing": "current", "normipeak_rmsep": r_cur,
                "conc_recovery_rmse_log10": np.nan, "conc_recovery_frac": 0.0,
                "n_folds": n_folds, "n_test": n_cur})
    r_s, n_s = _rmse(sens); rr_s, rf_s = _rec(rec_s)
    out.append({"framing": "sensitivity", "normipeak_rmsep": r_s,
                "conc_recovery_rmse_log10": rr_s, "conc_recovery_frac": rf_s,
                "n_folds": n_folds, "n_test": n_s})
    if include_quadratic:
        r_q, n_q = _rmse(quad); rr_q, rf_q = _rec(rec_q)
        out.append({"framing": "sensitivity_quadratic", "normipeak_rmsep": r_q,
                    "conc_recovery_rmse_log10": rr_q, "conc_recovery_frac": rf_q,
                    "n_folds": n_folds, "n_test": n_q})
    if include_interaction:
        r_i, n_i = _rmse(inter); rr_i, rf_i = _rec(rec_i)
        out.append({"framing": "interaction", "normipeak_rmsep": r_i,
                    "conc_recovery_rmse_log10": rr_i, "conc_recovery_frac": rf_i,
                    "n_folds": n_folds, "n_test": n_i})
    return pd.DataFrame(out)
