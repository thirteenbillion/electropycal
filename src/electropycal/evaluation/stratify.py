"""Stratify-by-drift diagnostic: is the signal just not evolved yet?

The pooled Q²≈0 could mean the state→drift map is genuinely absent, **or** that most channels are still on
the noisy early plateau before their asymptotic decay, so within the measured window the target barely
moves and there is nothing to predict beyond the mean. These look identical in a single pooled Q², but
they prescribe opposite actions (stop vs *keep measuring*).

This diagnostic separates them. It runs the same forward-chained population model, then pools Q²
**separately** over held-out sensor-timepoints that have already **drifted** (cumulative
|value−baseline|/|baseline| > ``drift_threshold``) vs those still **flat**, and reports each stratum's
**target standard deviation** (how much there is to predict there). Signature of "not evolved yet":
``still_flat`` has near-zero target std and undefined/≈0 Q², while ``has_drifted`` has real target
variance and a **higher** Q², meaning the signal appears once channels evolve, so more timespan pays off.

``clean_trend_channels`` additionally isolates channels with a strong monotone time trend (a ceiling
diagnostic: if the model works only on clean-trend channels, the signal is real but needs well-behaved,
evolved sensors; scope the deliverable accordingly).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def drift_magnitude(sens_df: pd.DataFrame, value_col: str = "sensitivity") -> np.ndarray:
    """Per ``(device, channel, timepoint)`` cumulative fractional drift from the sensor's baseline
    (earliest-timepoint) value: |value − baseline| / |baseline|. NaN where the baseline is non-positive."""
    id_cols = [c for c in ("device", "channel") if c in sens_df.columns]
    df = sens_df.sort_values(id_cols + ["timepoint"]) if id_cols else sens_df
    base = df.groupby(id_cols)[value_col].transform("first") if id_cols else df[value_col]
    mag = (df[value_col] - base).abs() / base.abs()
    out = np.array(mag.to_numpy(float), dtype=float, copy=True)
    b = np.asarray(base, float)
    out[~np.isfinite(b) | (np.abs(b) < 1e-12)] = np.nan
    return pd.Series(out, index=df.index).reindex(sens_df.index).to_numpy(float)


def clean_trend_channels(sens_df: pd.DataFrame, value_col: str = "sensitivity",
                         min_abs_spearman: float = 0.6, min_times: int = 4) -> set:
    """Set of ``(device, channel)`` keys whose ``value_col`` has a strong monotone trend vs time
    (|Spearman r| ≥ ``min_abs_spearman`` over ≥ ``min_times`` timepoints), the well-behaved subset."""
    from scipy.stats import spearmanr
    id_cols = [c for c in ("device", "channel") if c in sens_df.columns]
    keep = set()
    for key, g in sens_df.groupby(id_cols):
        gg = g.dropna(subset=[value_col])
        if gg["timepoint"].nunique() < min_times:
            continue
        r, _ = spearmanr(gg["timepoint"].to_numpy(float), gg[value_col].to_numpy(float))
        if np.isfinite(r) and abs(r) >= min_abs_spearman:
            keep.add(key if isinstance(key, tuple) else (key,))
    return keep


def stratify_by_drift(sens_df: pd.DataFrame, target: str = "sensitivity", drift_threshold: float = 0.2,
                      min_train_times: int = 3, alpha: float | None = None) -> pd.DataFrame:
    """Forward-chained population ridge, with pooled Q²/RMSEP reported per drift stratum.

    Returns a DataFrame with one row per stratum (``all``, ``has_drifted``, ``still_flat``): ``n`` test
    points, ``q2`` (vs the forward train-mean), ``rmsep``, and ``target_std`` (the dynamic range of the
    target in that stratum). Read ``has_drifted`` Q² ≫ ``still_flat`` Q² with ``still_flat`` target_std≈0
    as "the signal only appears once channels evolve; keep measuring".

    ``alpha`` is the ridge penalty. **Default ``None`` selects it per fold by inner cross-validation**
    over :data:`ALPHA_GRID`, which is what you want here: this problem has ~145 predictors and only a
    few dozen training rows early in the forward chain, so a fixed small penalty overfits catastrophically
    (a fixed ``alpha=1.0`` returns Q² ≈ −45 with RMSEP ~7× the naive mean: a property of the penalty,
    not of the data). Pass a float only to reproduce a specific historical run.
    """
    from ..data.schema import RESERVED_COLUMNS
    from .metrics import fit_ridge
    df = sens_df.copy()
    df = df[np.isfinite(df[target])].reset_index(drop=True)
    if "timepoint" not in df.columns or df.empty:
        return pd.DataFrame(columns=["stratum", "n", "q2", "rmsep", "target_std"])
    id_cols = [c for c in ("device", "channel") if c in df.columns]
    base_tp = df.groupby(id_cols)["timepoint"].transform("min") if id_cols else df["timepoint"] * 0
    age = df["timepoint"].to_numpy(float) - np.asarray(base_tp, float)
    mag = drift_magnitude(df, target)
    feat_cols = [c for c in df.columns if c not in set(RESERVED_COLUMNS)
                 and c != "time_index" and pd.api.types.is_numeric_dtype(df[c])]
    X = np.array(df[feat_cols].to_numpy(float), dtype=float, copy=True); X[~np.isfinite(X)] = np.nan
    X = np.column_stack([X, age])                                     # age as a known covariate
    y = df[target].to_numpy(float)
    tp = df["timepoint"].to_numpy(float)
    times = np.unique(tp)

    rec_y, rec_p, rec_n, rec_m = [], [], [], []
    for ti, t_test in enumerate(times):
        if ti < min_train_times:
            continue
        te = np.where(tp == t_test)[0]; tr = np.where(tp < t_test)[0]
        if te.size == 0 or tr.size < 8:
            continue
        mu = np.nanmean(X[tr], 0); sd = np.nanstd(X[tr], 0); sd = np.where(sd > 0, sd, 1.0)
        def z(a):
            b = (a - mu) / sd; b[~np.isfinite(b)] = 0.0; return b
        try:
            m = fit_ridge(z(X[tr]), y[tr], alpha); p = m.predict(z(X[te]))
        except Exception:
            continue
        rec_y.append(y[te]); rec_p.append(p); rec_n.append(np.full(te.size, float(np.mean(y[tr]))))
        rec_m.append(mag[te])

    if not rec_y:
        return pd.DataFrame(columns=["stratum", "n", "q2", "rmsep", "target_std"])
    Y = np.concatenate(rec_y); P = np.concatenate(rec_p); N = np.concatenate(rec_n); M = np.concatenate(rec_m)

    def _row(name, mask):
        yy, pp, nn = Y[mask], P[mask], N[mask]
        if yy.size == 0:
            return dict(stratum=name, n=0, q2=float("nan"), rmsep=float("nan"), target_std=float("nan"))
        sse = float(np.sum((yy - pp) ** 2)); sst = float(np.sum((yy - nn) ** 2))
        return dict(stratum=name, n=int(yy.size), q2=(1 - sse / sst) if sst > 0 else float("nan"),
                    rmsep=float(np.sqrt(np.mean((yy - pp) ** 2))), target_std=float(np.std(yy)))

    drifted = np.isfinite(M) & (M > drift_threshold)
    flat = np.isfinite(M) & (M <= drift_threshold)
    return pd.DataFrame([_row("all", np.ones(Y.size, bool)),
                         _row("has_drifted", drifted), _row("still_flat", flat)])
