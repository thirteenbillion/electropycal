"""Trivial baselines that bound what the electrode-state model must beat.

Two reference points make a Q² legible:

- **naïve mean**: predict every held-out sensor-timepoint's target with the *training-set mean*. This is
  the SST denominator of Q² itself; its RMSEP is the "do-nothing" error. A small model-RMSEP is only
  impressive **relative to this**: if the target barely moves, the naïve RMSEP is already tiny and a low
  model-RMSEP means nothing (Q²≈0).
- **time-only**: predict the target from the sensor's **age alone** (`time_since_baseline`, days since its
  first timepoint), forward-chained, no EIS/FSCV features. Comparing the full electrode-state model to this
  answers one question directly: *does the impedance probe add anything over simply knowing how old the
  channel is?* If full ≈ time-only, EIS is not carrying orthogonal drift information.

- **channel persistence**: predict a channel's next value (or degraded/not) from **its own earlier
  timepoints**, with *no electrode-state features at all*. This is the sharpest of the three and the one
  that settled the project's central question: on the real data it **beats** every state-based model
  (regression q² 0.34 vs the hierarchical model's best 0.22; classification AUC 0.70/0.76/0.81 vs the
  drift classifier's 0.70/0.70/0.80). Any state-based result that merely matches it is measuring the
  target's per-channel autocorrelation, not the electrode state. Always report it alongside.

All use the same forward-chained, globally-pooled protocol as the discovery tracks, so the Q²/RMSEP
numbers are directly comparable to a run's ``pooled_q2`` / ``pooled_rmsep``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _time_since_baseline(sens_df: pd.DataFrame) -> np.ndarray:
    id_cols = [c for c in ("device", "channel") if c in sens_df.columns]
    base = sens_df.groupby(id_cols)["timepoint"].transform("min")
    return (sens_df["timepoint"].to_numpy(float) - base.to_numpy(float))


def time_only_baseline_cv(sens_df: pd.DataFrame, target: str = "sensitivity",
                          min_train_times: int = 3, degree: int = 1) -> dict:
    """Forward-chained, globally-pooled CV of a **time-only** model (age → target) and the **naïve mean**.

    ``sens_df`` is a per-sensor-timepoint frame (e.g. ``sensitivity_featureset``). For each test timepoint
    (after ``min_train_times``), fit a degree-``degree`` polynomial of ``time_since_baseline`` on all
    earlier timepoints (pooled across sensors) and predict the held-out timepoint; in parallel score the
    train-mean prediction. Returns pooled ``time_q2`` and ``naive`` /  ``time`` RMSEPs vs the same
    train-mean baseline, plus ``n_folds`` / ``n_test``. ``time_q2`` > 0 ⇒ a population age-trend predicts
    the target; compare a full-feature run's ``pooled_q2`` to this to isolate what EIS adds over age.
    """
    df = sens_df.copy()
    df = df[np.isfinite(df[target])]
    if "timepoint" not in df.columns or df.empty:
        return {"time_q2": float("nan"), "naive_rmsep": float("nan"), "time_rmsep": float("nan"),
                "n_folds": 0, "n_test": 0}
    age = _time_since_baseline(df)
    y = df[target].to_numpy(float)
    tp = df["timepoint"].to_numpy(float)
    times = np.unique(tp)

    y_true, y_time, y_naive = [], [], []
    n_folds = 0
    for ti, t_test in enumerate(times):
        if ti < min_train_times:
            continue
        te = np.where(tp == t_test)[0]
        tr = np.where(tp < t_test)[0]
        if te.size == 0 or tr.size < max(2, degree + 1):
            continue
        mean_tr = float(np.mean(y[tr]))
        try:
            coef = np.polyfit(age[tr], y[tr], deg=min(degree, tr.size - 1))
            pred = np.polyval(coef, age[te])
        except Exception:
            continue
        y_true.extend(y[te].tolist())
        y_time.extend(np.asarray(pred, float).tolist())
        y_naive.extend([mean_tr] * te.size)
        n_folds += 1

    yt = np.asarray(y_true); yp = np.asarray(y_time); yn = np.asarray(y_naive)
    if yt.size == 0:
        return {"time_q2": float("nan"), "naive_rmsep": float("nan"), "time_rmsep": float("nan"),
                "n_folds": 0, "n_test": 0}
    sse = float(np.sum((yt - yp) ** 2)); sst = float(np.sum((yt - yn) ** 2))
    return {"time_q2": (1.0 - sse / sst) if sst > 0 else float("nan"),
            "naive_rmsep": float(np.sqrt(np.mean((yt - yn) ** 2))),
            "time_rmsep": float(np.sqrt(np.mean((yt - yp) ** 2))),
            "target_std": float(np.std(yt)), "n_folds": n_folds, "n_test": int(yt.size)}


def _forward_folds(tp: np.ndarray, min_train_times: int):
    """Forward-chained folds (train = every timepoint strictly before the test one)."""
    for ti, t in enumerate(np.unique(tp)):
        if ti < min_train_times:
            continue
        te = np.where(tp == t)[0]
        tr = np.where(tp < t)[0]
        if te.size == 0 or tr.size < 8:
            continue
        yield tr, te


def channel_persistence_cv(sens_df: pd.DataFrame, target: str = "sensitivity",
                           min_train_times: int = 3, fracs: tuple[float, ...] = (0.3, 0.5, 0.7),
                           k_intercept: float = 2.0, k_slope: float = 4.0) -> pd.DataFrame:
    """Feature-free controls: how far does a channel's **own past** get you, with no electrode state?

    Three controls, all forward-chained on the same folds the state-based evaluators use:

    - ``chan_mean``: predict the shrunk mean of that channel's own earlier values (regression, ``q2``).
    - ``chan_trend``: additionally extrapolate its own age-slope (regression, ``q2``). Compare to
      ``chan_mean`` to test whether *dynamics* add anything: if it is worse, momentum carries no signal
      and a state-space/Kalman model is not warranted.
    - ``persistence_frac{f}``: score a channel by the fraction of its own earlier timepoints already
      labelled degraded at threshold ``f`` (classification, ``roc_auc``); the control for
      :func:`~electropycal.evaluation.classify.drift_classifier_cv`.

    Returns one row per control with ``control``, ``kind``, ``metric``, ``value``, ``rmsep``, ``n_test``.

    These are **the bar**, not a curiosity. A drift monitor that only matches ``persistence`` is not
    reading the electrode; it is reading "this channel was already bad", which in vivo you cannot
    measure anyway (it needs dopamine standards). Shrinkage (``k_intercept`` / ``k_slope``) matches
    :func:`~electropycal.evaluation.hierarchical.hierarchical_cv` so the comparison is like-for-like.
    """
    from sklearn.metrics import roc_auc_score

    from .classify import degraded_labels

    df = sens_df[np.isfinite(sens_df[target])].reset_index(drop=True)
    ids = [c for c in ("device", "channel") if c in df.columns]
    if not ids or "timepoint" not in df.columns or df.empty:
        return pd.DataFrame(columns=["control", "kind", "metric", "value", "rmsep", "n_test"])
    gid = df[ids].astype(str).agg("|".join, axis=1).to_numpy()
    tp = df["timepoint"].to_numpy(float)
    y = df[target].to_numpy(float)
    age = tp - df.groupby(ids)["timepoint"].transform("min").to_numpy(float)

    rows = []
    for model in ("chan_mean", "chan_trend"):
        yt, yp, yn = [], [], []
        for tr, te in _forward_folds(tp, min_train_times):
            mtr = float(np.mean(y[tr]))
            pred = np.full(te.size, mtr)
            for j, i in enumerate(te):
                m = gid[tr] == gid[i]
                n = int(m.sum())
                if n == 0:
                    continue
                r = y[tr][m] - mtr
                pred[j] = mtr + float(np.mean(r)) * (n / (n + k_intercept))
                if model == "chan_trend":
                    a = age[tr][m]
                    am = float(a.mean())
                    if np.unique(a).size >= 2:
                        ac = a - am
                        denom = float(np.sum(ac * ac))
                        if denom > 0:
                            slope = float(np.sum(ac * (r - r.mean())) / denom) * (n / (n + k_slope))
                            pred[j] += slope * (age[i] - am)
            yt.extend(y[te]); yp.extend(pred); yn.extend([mtr] * te.size)
        yt, yp, yn = np.asarray(yt), np.asarray(yp), np.asarray(yn)
        if yt.size == 0:
            continue
        sst = float(np.sum((yt - yn) ** 2))
        rows.append({"control": model, "kind": "regression", "metric": "q2",
                     "value": (1.0 - float(np.sum((yt - yp) ** 2)) / sst) if sst > 0 else float("nan"),
                     "rmsep": float(np.sqrt(np.mean((yt - yp) ** 2))), "n_test": int(yt.size)})

    for frac in fracs:
        lab = degraded_labels(df, target, frac=frac).to_numpy(float)
        yt, yp = [], []
        for tr, te in _forward_folds(tp, min_train_times):
            for i in te:
                if not np.isfinite(lab[i]):
                    continue
                m = (gid[tr] == gid[i]) & np.isfinite(lab[tr])
                yp.append(float(np.mean(lab[tr][m])) if m.sum() else float(np.nanmean(lab[tr])))
                yt.append(lab[i])
        yt, yp = np.asarray(yt), np.asarray(yp)
        auc = float(roc_auc_score(yt, yp)) if yt.size and len(np.unique(yt)) > 1 else float("nan")
        rows.append({"control": f"persistence_frac{frac}", "kind": "classification",
                     "metric": "roc_auc", "value": auc, "rmsep": float("nan"), "n_test": int(yt.size)})
    return pd.DataFrame(rows)
