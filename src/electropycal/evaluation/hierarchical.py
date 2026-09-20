"""Hierarchical / partial-pooling recalibration model (study experiment E6).

The pooled ``global`` track fits one state→drift map for every channel (infinite shrinkage); the
``channel`` track fits a separate map per channel (zero shrinkage, and starved). **Partial pooling** is
the principled interpolation between them: a population fixed-effect map plus a per-group **random
effect** whose strength is set by how much that group's own history supports departing from the
population. Empirical-Bayes shrinkage does this without hand-picking "good" channels — a noisy group is
pulled hard toward the population, a well-behaved group keeps its own trajectory.

Forward-chained and deployment-faithful: a test group's random effect is estimated from **its own
earlier timepoints only** (the outer loop trains on ``timepoint < t_test``), so the population prior
carries a channel early and it specializes as its history accrues. Random **intercept** + random
**age-slope** (``time_since_baseline``), since E1 shows per-channel slopes differ.

``group_level``:
  - ``channel``            — random effect per ``(device, channel)`` sensor.
  - ``device``             — random effect shared by all channels on a device.
  - ``channel_in_device``  — nested: device-level effect + channel-within-device effect (the
                             physically-correct "auto" grouping; two variance components).

Returns pooled ``q2`` (partial pooling) alongside ``q2_fixed`` (population map only, no random effects)
so the *lift from pooling* is explicit, plus ``rmsep`` / ``naive_rmsep`` / ``n_folds`` / ``n_test``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _group_keys(df: pd.DataFrame, level: str) -> list[np.ndarray]:
    """Return the list of grouping-key arrays for ``level`` (outer→inner for the nested level)."""
    dev = df["device"].astype(str).to_numpy() if "device" in df.columns else np.array(["d"] * len(df))
    ch = df["channel"].astype(str).to_numpy() if "channel" in df.columns else np.array(["c"] * len(df))
    if level == "device":
        return [dev]
    if level == "channel":
        return [np.char.add(np.char.add(dev, "|"), ch)]
    if level == "channel_in_device":
        return [dev, np.char.add(np.char.add(dev, "|"), ch)]
    raise ValueError(f"unknown group_level {level!r}")


def _shrunk_effects(resid: np.ndarray, age: np.ndarray, groups: np.ndarray,
                    k_intercept: float, k_slope: float) -> dict:
    """Empirical-Bayes random intercept + age-slope per group, shrunk toward 0 by group support.

    intercept_g = mean(resid_g) · n/(n+k_intercept); slope_g fit by LS of resid on centered age within
    the group, shrunk by n/(n+k_slope). Groups with <2 distinct ages get slope 0."""
    eff = {}
    for g in np.unique(groups):
        m = groups == g
        n = int(m.sum())
        r = resid[m]
        inter = float(np.mean(r)) * (n / (n + k_intercept)) if n > 0 else 0.0
        slope = 0.0
        a = age[m]
        if np.unique(a).size >= 2:
            ac = a - a.mean()
            denom = float(np.sum(ac * ac))
            if denom > 0:
                raw = float(np.sum(ac * (r - r.mean())) / denom)
                slope = raw * (n / (n + k_slope))
        eff[g] = (inter, slope, float(a.mean()) if a.size else 0.0)
    return eff


def hierarchical_cv(sens_df: pd.DataFrame, target: str = "sensitivity", group_level: str = "channel",
                    min_train_times: int = 3, alpha: float | None = None,
                    k_intercept: float = 2.0, k_slope: float = 4.0) -> dict:
    """Forward-chained, globally-pooled partial-pooling CV. See module docstring.

    Population map = ridge on the leakage-safe numeric predictors + ``time_since_baseline``,
    standardized per fold from the training set. Random effects added on top per ``group_level`` from the
    training residuals (forward-chained). Metrics pooled globally vs the forward train-mean baseline.

    ``alpha`` is the ridge penalty. **Default ``None`` selects it per fold by inner cross-validation**
    over :data:`~electropycal.evaluation.metrics.ALPHA_GRID`. This matters more here than anywhere else
    in the library: with ~145 predictors and only a few dozen early training rows, a fixed ``alpha=1.0``
    drives the population map to Q² ≈ −45 (RMSEP ~7× the naive mean), and because the random effects are
    estimated from *that* map's residuals, the pooling comparison becomes meaningless too. Pass a float
    only to reproduce a specific historical run.

    .. note::
       Read ``q2`` against ``q2_fixed`` **and** against a feature-free control (see
       :func:`~electropycal.evaluation.baselines.channel_persistence_cv`). On the real data the control
       beats both, i.e. the apparent lift from pooling is the target's per-channel autocorrelation
       rather than anything the electrode state contributes.
    """
    from ..data.schema import RESERVED_COLUMNS
    from .metrics import fit_ridge
    df = sens_df.copy()
    df = df[np.isfinite(df[target])]
    if "timepoint" not in df.columns or df.empty:
        return {"q2": float("nan"), "q2_fixed": float("nan"), "rmsep": float("nan"),
                "naive_rmsep": float("nan"), "n_folds": 0, "n_test": 0, "group_level": group_level}
    id_cols = [c for c in ("device", "channel") if c in df.columns]
    base_tp = df.groupby(id_cols)["timepoint"].transform("min") if id_cols else df["timepoint"] * 0
    df = df.assign(_age=df["timepoint"].to_numpy(float) - np.asarray(base_tp, float))
    feat_cols = [c for c in df.columns if c not in set(RESERVED_COLUMNS)
                 and c not in ("_age", "time_index") and pd.api.types.is_numeric_dtype(df[c])]
    feat_cols = feat_cols + ["_age"]                                   # age is a known-at-deployment covariate
    X = np.array(df[feat_cols].to_numpy(float), dtype=float, copy=True)
    X[~np.isfinite(X)] = np.nan
    y = df[target].to_numpy(float)
    age = df["_age"].to_numpy(float)
    tp = df["timepoint"].to_numpy(float)
    gkeys = _group_keys(df, group_level)
    times = np.unique(tp)

    yt, yp_h, yp_f, yn = [], [], [], []
    n_folds = 0
    for ti, t_test in enumerate(times):
        if ti < min_train_times:
            continue
        te = np.where(tp == t_test)[0]
        tr = np.where(tp < t_test)[0]
        if te.size == 0 or tr.size < 8:
            continue
        mu = np.nanmean(X[tr], 0); sd = np.nanstd(X[tr], 0); sd = np.where(sd > 0, sd, 1.0)

        def z(a):
            b = (a - mu) / sd; b[~np.isfinite(b)] = 0.0; return b
        try:
            fx = fit_ridge(z(X[tr]), y[tr], alpha)
            f_tr = fx.predict(z(X[tr]))
            f_te = fx.predict(z(X[te]))
        except Exception:
            continue
        resid = y[tr] - f_tr
        pred_h = f_te.copy()
        # random effects, outer→inner (nested adds device then channel-in-device on the running residual)
        run_resid = resid.copy()
        for keys in gkeys:
            eff = _shrunk_effects(run_resid, age[tr], keys[tr], k_intercept, k_slope)
            # subtract this level's fitted effect from the residual before the next (nested) level
            for g, (inter, slope, amean) in eff.items():
                m = keys[tr] == g
                run_resid[m] = run_resid[m] - (inter + slope * (age[tr][m] - amean))
            for j in range(te.size):
                g = keys[te][j]
                if g in eff:
                    inter, slope, amean = eff[g]
                    pred_h[j] += inter + slope * (age[te][j] - amean)
        mean_tr = float(np.mean(y[tr]))
        yt.extend(y[te].tolist()); yp_h.extend(pred_h.tolist()); yp_f.extend(f_te.tolist())
        yn.extend([mean_tr] * te.size)
        n_folds += 1

    yt = np.asarray(yt); yph = np.asarray(yp_h); ypf = np.asarray(yp_f); yn = np.asarray(yn)
    if yt.size == 0:
        return {"q2": float("nan"), "q2_fixed": float("nan"), "rmsep": float("nan"),
                "naive_rmsep": float("nan"), "n_folds": 0, "n_test": 0, "group_level": group_level}

    def _q2(p):
        sse = float(np.sum((yt - p) ** 2)); sst = float(np.sum((yt - yn) ** 2))
        return (1.0 - sse / sst) if sst > 0 else float("nan")
    return {"q2": _q2(yph), "q2_fixed": _q2(ypf),
            "rmsep": float(np.sqrt(np.mean((yt - yph) ** 2))),
            "naive_rmsep": float(np.sqrt(np.mean((yt - yn) ** 2))),
            "n_folds": n_folds, "n_test": int(yt.size), "group_level": group_level}
