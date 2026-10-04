"""Discovery results review: plots over a finished run directory (folds discovery_results_review).

A discovery run writes a directory tree (``report/condition_ranking.parquet``, ``summary.parquet``,
``report/feature_ranking.parquet``, ``conditions/<name>/predictions.parquet``). These functions read
that tree and draw the review figures, so the notebook is thin calls and the same plots are reproducible
from a script. Each takes the run directory (a path); the calibration review also takes an optional
condition name (default: the best-ranked).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from .. import viz


def _provenance(run_dir: Path) -> tuple[str, str]:
    """(provenance slug, data-stage label) for a run directory, for :func:`viz.emit`.

    Read from the run's own ``sweep_manifest.json`` rather than inferred from the path, so a
    directory that was moved or renamed still reports what actually produced it. A run with
    no manifest is the self-provisioned example case, which is exactly the one that must not
    be mistaken for measurements, so it is labelled loudly instead of left blank.
    """
    import json
    name = run_dir.name or "run"
    try:
        m = json.loads((run_dir / "sweep_manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return (f"unverified-{name}",
                ("run directory carries no manifest: SOURCE UNVERIFIED, may be a synthetic "
                 "example rather than measurements"))
    env = m.get("environment", {})
    released = env.get("electropycal_released")
    ver = env.get("electropycal", "?")
    if released is True:
        return (f"{name}-rel{ver}", f"discovery run {name}, library {ver} (released)")
    commit = (env.get("electropycal_commit") or "?")[:7]
    return (f"{name}-dev{commit}",
            (f"discovery run {name}, library built from checkout {commit}: NOT attributable "
             f"to a released version"))


def _tp_label(tp) -> str:
    return f"D{int(tp)}"


def plot_condition_ranking(run_dir, show: bool = True) -> pd.DataFrame:
    """Horizontal bar of pooled RMSEP (95% CI) per condition (best at bottom). Returns the ranking frame."""
    viz.ensure_style()
    import matplotlib.pyplot as plt
    run_dir = Path(run_dir)
    _prov, _stage = _provenance(run_dir)
    cr = pd.read_parquet(run_dir / "report" / "condition_ranking.parquet").dropna(subset=["pooled_rmsep"])
    fig, ax = plt.subplots(figsize=(9, 3.6))
    err = [cr.pooled_rmsep - cr.rmsep_ci_lo, cr.rmsep_ci_hi - cr.pooled_rmsep]
    ax.barh(cr.condition[::-1], cr.pooled_rmsep[::-1], xerr=[e[::-1] for e in err], capsize=3, color="#4C72B0")
    ax.set_xlabel("pooled RMSEP (95% CI)"); ax.set_title("Condition ranking (best at bottom)")
    plt.tight_layout()
    if show:
        viz.emit("review_condition_ranking", provenance=_prov, stage=_stage)
    return cr


def plot_fold_spread(run_dir, show: bool = True):
    """Box plot of per-fold RMSEP per condition (ordered by median)."""
    viz.ensure_style()
    import matplotlib.pyplot as plt
    run_dir = Path(run_dir)
    _prov, _stage = _provenance(run_dir)
    sm = pd.read_parquet(run_dir / "summary.parquet")
    order = list(sm.groupby("condition")["rmsep"].median().sort_values().index)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.boxplot([sm[sm.condition == c]["rmsep"].dropna() for c in order])
    ax.set_xticks(range(1, len(order) + 1)); ax.set_xticklabels(order, rotation=25, ha="right")
    ax.set_ylabel("per-fold RMSEP"); ax.set_title("Fold-level spread per condition")
    plt.tight_layout()
    if show:
        viz.emit("review_fold_spread", provenance=_prov, stage=_stage)


def plot_feature_ranking(run_dir, top: int = 15, show: bool = True) -> pd.DataFrame:
    """Horizontal bar of the top-``top`` features by cross-condition mean selection frequency."""
    viz.ensure_style()
    import matplotlib.pyplot as plt
    run_dir = Path(run_dir)
    _prov, _stage = _provenance(run_dir)
    fr = pd.read_parquet(run_dir / "report" / "feature_ranking.parquet").head(top)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(fr.feature[::-1], fr.mean_selection_frequency[::-1], color="#55A868")
    ax.set_xlabel("mean selection frequency across conditions"); ax.set_title("Top features")
    plt.tight_layout()
    if show:
        viz.emit("review_feature_ranking", provenance=_prov, stage=_stage)
    return fr


def plot_calibration_review(run_dir, condition: str | None = None, show: bool = True) -> pd.DataFrame:
    """Calibration (true vs predicted by timepoint), parity, and a 6-panel residual diagnostic for one
    condition (default: best-ranked). Prints RMSEP slices + a heteroscedasticity indicator; returns the
    held-out predictions frame."""
    viz.ensure_style()
    import matplotlib.pyplot as plt
    run_dir = Path(run_dir)
    _prov, _stage = _provenance(run_dir)
    cr = pd.read_parquet(run_dir / "report" / "condition_ranking.parquet").dropna(subset=["pooled_rmsep"])
    cond = condition or cr.sort_values("pooled_rmsep").condition.iloc[0]
    P = pd.read_parquet(run_dir / "conditions" / cond / "predictions.parquet")
    print(f"calibration review: {cond}  |  {len(P)} held-out predictions, {P.device.nunique()} device(s), "
          f"{P.channel.nunique()} channel(s), {P.t_test.nunique()} timepoint(s)")

    def rmse(x):
        return float(np.sqrt(np.mean(np.asarray(x, float) ** 2)))
    tps = sorted(P.t_test.unique()); cmap = plt.get_cmap("viridis")
    tcol = {t: cmap(i / max(len(tps) - 1, 1)) for i, t in enumerate(tps)}

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for t in tps:
        g = P[P.t_test == t]
        tru = g.groupby("concentration").y_true.agg(["mean", "std"]); prd = g.groupby("concentration").y_pred.mean()
        ax.errorbar(tru.index, tru["mean"], yerr=tru["std"].fillna(0.0), fmt="o-", color=tcol[t],
                    capsize=2, lw=1.3, label=f"{_tp_label(t)} true")
        ax.plot(prd.index, prd.values, "--", color=tcol[t], lw=1.3, label=f"{_tp_label(t)} pred")
    ax.set_xscale("log"); ax.set_xlabel("concentration (nM)"); ax.set_ylabel("NormIpeak")
    ax.set_title(f"{cond}\ncalibration: true (mean +/- s.d. across sensors) vs predicted, by timepoint")
    ax.legend(fontsize=6, ncol=2); plt.tight_layout()
    if show:
        viz.emit("review_calibration_review_1", provenance=_prov, stage=_stage)

    fig, ax = plt.subplots(figsize=(4.4, 4.1))
    sc = ax.scatter(P.y_true, P.y_pred, c=np.log10(P.concentration), cmap="plasma", s=14, alpha=0.75)
    lim = [float(min(P.y_true.min(), P.y_pred.min())), float(max(P.y_true.max(), P.y_pred.max()))]
    ax.plot(lim, lim, "k--", lw=0.8); ax.set_xlabel("true NormIpeak"); ax.set_ylabel("predicted NormIpeak")
    ax.set_title("predicted vs true (parity)"); fig.colorbar(sc, ax=ax, label="log10 conc"); plt.tight_layout()
    if show:
        viz.emit("review_calibration_review_2", provenance=_prov, stage=_stage)

    r = P.residual.to_numpy()
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.6))
    axs[0, 0].scatter(P.concentration, r, s=10, alpha=0.6); axs[0, 0].set_xscale("log")
    axs[0, 0].axhline(0, color="k", lw=0.7); axs[0, 0].set_xlabel("concentration (nM)")
    axs[0, 0].set_ylabel("residual (pred - true)"); axs[0, 0].set_title("vs concentration")
    axs[0, 1].scatter(P.t_test, r, s=10, alpha=0.6); axs[0, 1].axhline(0, color="k", lw=0.7)
    axs[0, 1].set_xlabel("timepoint (t_test, days)"); axs[0, 1].set_title("vs time  (drift of model relevance)")
    axs[0, 2].scatter(P.y_true, r, s=10, alpha=0.6); axs[0, 2].axhline(0, color="k", lw=0.7)
    axs[0, 2].set_xlabel("true NormIpeak"); axs[0, 2].set_title("vs true  (nonlinearity)")
    axs[1, 0].scatter(np.abs(P.y_pred), np.abs(r), s=10, alpha=0.6)
    axs[1, 0].set_xlabel("|predicted|"); axs[1, 0].set_ylabel("|residual|"); axs[1, 0].set_title("heteroscedasticity")
    chs = sorted(P.channel.unique())
    axs[1, 1].boxplot([P[P.channel == c].residual for c in chs]); axs[1, 1].axhline(0, color="k", lw=0.7)
    axs[1, 1].set_xticklabels(chs, fontsize=6); axs[1, 1].set_xlabel("channel"); axs[1, 1].set_title("by channel")
    devs = sorted(P.device.unique())
    axs[1, 2].boxplot([P[P.device == d].residual for d in devs]); axs[1, 2].axhline(0, color="k", lw=0.7)
    axs[1, 2].set_xticklabels(devs, fontsize=7); axs[1, 2].set_xlabel("device"); axs[1, 2].set_title("by device")
    fig.suptitle(f"{cond}: residual diagnostics", y=1.01); plt.tight_layout()
    if show:
        viz.emit("review_calibration_review_3", provenance=_prov, stage=_stage)

    het = float(np.corrcoef(np.abs(P.y_pred), np.abs(r))[0, 1]) if len(P) > 2 else float("nan")
    print("RMSEP by concentration:", {float(k): round(v, 3) for k, v in P.groupby("concentration").residual.apply(rmse).items()})
    print("RMSEP by timepoint:    ", {_tp_label(k): round(v, 3) for k, v in P.groupby("t_test").residual.apply(rmse).items()})
    print(f"heteroscedasticity: corr(|predicted|, |residual|) = {het:+.2f}  "
          f"(near 0 = homoscedastic; strongly positive = error grows with signal)")
    return P


def plot_target_framing_comparison(featureset: pd.DataFrame, include_interaction: bool = True,
                                   show: bool = True) -> pd.DataFrame:
    """Compare target framings (NormIpeak vs sensitivity vs …) on a featureset: per-dose RMSEP and
    concentration-recovery error. Returns the comparison frame."""
    viz.ensure_style()
    import matplotlib.pyplot as plt
    from ..evaluation.framing import compare_target_framings
    cmp = compare_target_framings(featureset, include_interaction=include_interaction)
    best = cmp.loc[cmp.normipeak_rmsep.idxmin(), "framing"]
    print(f"lowest per-dose NormIpeak RMSEP: '{best}'.  'current' cannot recover concentration "
          f"(flat prediction); compare sensitivity vs interaction recovery (log10[DA], lower=better).")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].bar(cmp.framing, cmp.normipeak_rmsep, color="#4C78A8")
    ax[0].set_ylabel("RMSEP"); ax[0].set_title("per-dose NormIpeak RMSEP (lower = better)")
    rec = cmp.set_index("framing")["conc_recovery_rmse_log10"]
    ax[1].bar(rec.index, rec.fillna(0).values, color="#E45756")
    ax[1].set_ylabel("RMSE (log10[DA])"); ax[1].set_title("concentration recovery (lower = better)")
    for i, (n, v) in enumerate(rec.items()):
        ax[1].text(i, (0 if not np.isfinite(v) else v), ("n/a" if not np.isfinite(v) else f"{v:.2f}"),
                   ha="center", va="bottom", fontsize=8)
    for a in ax:
        a.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    if show:
        viz.emit("review_target_framing_comparison")
    return cmp
