"""Model deployment: freeze a trained model and recalibrate in vivo (see DESIGN §9).

Two entry points:

- :func:`freeze_model`: produce a portable ``frozen_model/`` bundle from a model
  the *user chooses* (architecture, ``k``, featureset) trained on 100% of the
  in-vitro data, with frozen scalers computed once. This is deployment Step 0; the
  choice of model is deliberately left to the caller.
- :func:`recalibrate`: apply a frozen bundle to new in-vivo features (already
  D0-normalized against an early in-vivo baseline): robust-scale with the frozen
  statistics, predict (back-transform if log), and flag domain shift via CORAL.
  No refitting occurs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..data.io import load_npz, read_json, save_npz, write_json
from ..features.normalize import (apply_median_impute, apply_robust, apply_zscore,
                                  fit_median_impute, fit_robust, fit_zscore)
from ..models.base import predict_from_bundle
from ..models.variants import build
from . import domain


def freeze_model(X: np.ndarray, y: np.ndarray, feature_names: list[str], architecture: str,
                 k: int, out_dir: str | Path, feature_index: np.ndarray | None = None,
                 sample_weight: np.ndarray | None = None, scaler: str = "robust",
                 d0_kinds: dict[str, str] | None = None,
                 provenance: dict | None = None) -> Path:
    """Train the chosen model on 100% of (D0-normalized) in-vitro data and freeze it.

    Writes ``frozen_model/``: ``model_arrays.npz`` (coef/x_mean/y_mean), frozen
    ``scaler_center.npy`` / ``scaler_scale.npy``, ``reference_features.npy`` (scaled
    in-vitro features, for CORAL), ``feature_names.json``, ``d0_normalization.json``,
    and ``manifest.json``.

    ``provenance`` is merged into ``manifest.json`` under a ``"provenance"`` key: in
    particular the seed set of the discovery condition this model came from. A frozen bundle
    otherwise records the architecture and feature index but not which seed selected them, so a
    model chosen under a multi-seed sweep could not be traced back to it.
    """
    out_dir = Path(out_dir)
    X = np.asarray(X, float)
    idx = np.arange(X.shape[1]) if feature_index is None else np.asarray(feature_index, int)
    Xsub = X[:, idx]

    # freeze_model trains on 100% of the data -- that IS its training set, with no held-out
    # rows, so fitting the imputation median over all of it is correct rather than a leak. It is
    # SAVED because deployment must apply this same statistic to in-vivo rows; recomputing a
    # median from in-vivo data would scale the frozen model against a different reference.
    impute_median = fit_median_impute(Xsub)
    Xsub = apply_median_impute(Xsub, impute_median)
    center, scale = (fit_robust(Xsub) if scaler == "robust" else fit_zscore(Xsub))
    Xs = apply_robust(Xsub, center, scale) if scaler == "robust" else apply_zscore(Xsub, center, scale)

    arch = build(architecture, k).fit(Xs, y, sample_weight=sample_weight)
    arrays = {name: np.asarray(v) for name, v in arch.to_arrays().items()}
    arrays["feature_index"] = idx.astype(int)           # save all arch arrays (any architecture)

    save_npz(out_dir / "model_arrays.npz", **arrays)
    save_npz(out_dir / "scaler.npz", center=center, scale=scale)
    np.save(out_dir / "impute_median.npy", impute_median)
    np.save(out_dir / "scaler_center.npy", center)
    np.save(out_dir / "scaler_scale.npy", scale)
    np.save(out_dir / "reference_features.npy", Xs)
    write_json(out_dir / "feature_names.json", {"feature_names": [feature_names[i] for i in idx]})
    # full (pre-selection) feature order, so in-vivo columns can be aligned at deploy
    write_json(out_dir / "full_feature_names.json", {"feature_names": list(feature_names)})
    write_json(out_dir / "d0_normalization.json", d0_kinds or {})
    write_json(out_dir / "manifest.json", {
        **arch.manifest(), "scaler": scaler, "feature_index": idx.astype(int).tolist(),
        "n_features": int(idx.size), "provenance": dict(provenance or {})})
    return out_dir


def freeze_top(run_dir: str | Path, data, condition: str | None = None,
               stability_min: float = 0.5, out_dir: str | Path = "outputs/frozen_model",
               scaler: str = "robust") -> Path:
    """Refit a discovery-selected condition on 100% of in-vitro data and freeze it.

    Turnkey Step 0: from a completed discovery run, pick the top-ranked condition (or
    ``condition`` by name), reconstruct its architecture, the **modal ``k``** across
    folds, and a **consensus feature subset** (features selected in ≥ ``stability_min``
    of folds; all features if the condition has no selector), then train on all of
    ``data`` and write a ``frozen_model/`` bundle. ``data`` is a ``RunData`` or any
    featureset ``RunData.from_frame`` accepts. No per-fold files are copied; the
    deployable model is retrained here on the full dataset.
    """
    import statistics

    import pandas as pd

    from ..discovery.config import RunData
    from ..discovery.folds import fold_records
    run_dir = Path(run_dir)
    ranking = pd.read_parquet(run_dir / "report" / "condition_ranking.parquet")
    ranking = ranking.dropna(subset=["pooled_rmsep"]).sort_values("pooled_rmsep")
    if ranking.empty:
        raise ValueError("no ranked conditions with a finite RMSEP in this run")
    condition = condition or ranking.iloc[0]["condition"]
    row = ranking[ranking["condition"] == condition]
    if row.empty:
        raise ValueError(f"condition {condition!r} not in {run_dir}/report/condition_ranking.parquet")
    architecture = row.iloc[0]["architecture"]
    cond_dir = run_dir / "conditions" / condition
    cfg = read_json(cond_dir / "condition_config.json")

    ks = [r["hyperparams"]["k"] for r in fold_records(cond_dir).values() if "hyperparams" in r]
    k = int(statistics.mode(ks)) if ks else int(cfg["k_grid"][0])

    names = None
    if cfg.get("selector"):
        fs = pd.read_parquet(cond_dir / "feature_stability.parquet")
        names = fs.loc[fs["selection_frequency"] >= stability_min, "feature"].tolist()
        if not names:                                    # fallback: top features by frequency
            names = fs.sort_values("selection_frequency", ascending=False).head(max(k * 3, 5))["feature"].tolist()

    if not isinstance(data, RunData):
        data = RunData.from_frame(data)
    feature_index = None
    if names is not None:
        idx = sorted(data.feature_names.index(n) for n in names if n in data.feature_names)
        feature_index = np.asarray(idx, int)
    sample_weight = (1.0 / np.maximum(data.concentration, 1.0)
                     if architecture == "weighted_plsr" else None)
    print(f"freezing '{condition}': {architecture}, k={k}, "
          f"{len(feature_index) if feature_index is not None else len(data.feature_names)} features")
    # condition_config.json records the seed set the condition actually ran over (which differs
    # from the requested set when the selector is deterministic), so the bundle can state it.
    prov = {"condition": str(condition), "run_dir": str(run_dir),
            "seeds": list(cfg.get("seeds") or []),
            "fold_seed": cfg.get("fold_seed"),
            "stochastic_selector": cfg.get("stochastic_selector")}
    return freeze_model(data.X, data.y, data.feature_names, architecture, k, out_dir,
                        feature_index=feature_index, sample_weight=sample_weight, scaler=scaler,
                        provenance=prov)


@dataclass
class FrozenModel:
    arrays: dict
    manifest: dict
    center: np.ndarray
    scale: np.ndarray
    feature_index: np.ndarray
    reference: np.ndarray | None
    full_feature_names: list[str] | None = None
    #: per-feature imputation median frozen at training time, in ``feature_index`` order.
    #: ``None`` for bundles frozen before it was recorded (they simply do not impute).
    impute_median: np.ndarray | None = None


def load_frozen_model(bundle_dir: str | Path) -> FrozenModel:
    """Load a ``frozen_model/`` bundle (no pickle)."""
    bundle_dir = Path(bundle_dir)
    arrays = load_npz(bundle_dir / "model_arrays.npz")
    manifest = read_json(bundle_dir / "manifest.json")
    scaler = load_npz(bundle_dir / "scaler.npz")
    ref_path = bundle_dir / "reference_features.npy"
    reference = np.load(ref_path) if ref_path.exists() else None
    full_path = bundle_dir / "full_feature_names.json"
    full = read_json(full_path)["feature_names"] if full_path.exists() else None
    imp_path = bundle_dir / "impute_median.npy"
    imp = np.load(imp_path) if imp_path.exists() else None
    return FrozenModel(arrays=arrays, manifest=manifest, center=scaler["center"],
                       scale=scaler["scale"], feature_index=arrays["feature_index"].astype(int),
                       reference=reference, full_feature_names=full, impute_median=imp)


def recalibrate(model: FrozenModel | str | Path, X_invivo: np.ndarray,
                with_domain: bool = True) -> dict:
    """Recalibrate new in-vivo features (already D0-normalized) with a frozen model.

    Returns ``{"norm_ipeak": ndarray}`` plus, if ``with_domain``, a CORAL
    ``domain_distance`` vs the in-vitro reference (a confidence/extrapolation flag).
    ``X_invivo`` columns must align to the full feature order; the model's own
    ``feature_index`` selects its subset.
    """
    if not isinstance(model, FrozenModel):
        model = load_frozen_model(model)
    X = np.asarray(X_invivo, float)[:, model.feature_index]
    if model.impute_median is not None:
        X = apply_median_impute(X, model.impute_median)   # the TRAINING median, not an in-vivo one
    Xs = apply_robust(X, model.center, model.scale)
    pred = predict_from_bundle(model.arrays, model.manifest, Xs)
    out = {"norm_ipeak": pred}
    if with_domain and model.reference is not None:
        out["domain_distance"] = domain.coral_distance(model.reference, Xs)["distance"]
    return out


def recalibrate_invivo(model: FrozenModel | str | Path, invivo_root: str | Path,
                       flag_distance: float | None = None, out: str | Path | None = None):
    """End-to-end in-vivo recalibration from a raw directory (see DESIGN §9).

    Extracts the in-vivo featureset (:func:`features.extract.extract_invivo`),
    D0-normalizes it **per sensor against the early in-vivo baseline**, aligns columns
    to the frozen model's full feature order, then recalibrates each session and
    reports the CORAL domain distance. Returns a tidy DataFrame with one row per
    ``(timepoint)``: ``n``, ``mean_norm_ipeak``, ``domain_distance``, and (if
    ``flag_distance`` given) a ``confidence`` flag. If ``out`` is given the frame is
    also written there (parent dirs created). Assumes the in-vivo featureset shares
    the model's feature schema (same frequency grid).
    """
    import pandas as pd

    from ..features.extract import extract_invivo
    from ..features.normalize import d0_normalize_frame
    if not isinstance(model, FrozenModel):
        model = load_frozen_model(model)
    if model.full_feature_names is None:
        raise ValueError("frozen bundle lacks full_feature_names.json; re-freeze the model")

    df = extract_invivo(invivo_root)
    if df.empty:
        return pd.DataFrame(columns=["timepoint", "n", "mean_norm_ipeak", "domain_distance"])
    cols = list(model.full_feature_names)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"in-vivo featureset missing {len(missing)} model features "
                         f"(e.g. {missing[:3]}); check the frequency grid matches training")
    df = d0_normalize_frame(df, cols)                     # per-sensor, vs early in-vivo baseline

    rows = []
    for tp, g in df.groupby("timepoint"):
        r = recalibrate(model, g[cols].to_numpy(float))
        row = {"timepoint": float(tp), "n": int(len(g)),
               "mean_norm_ipeak": float(np.nanmean(r["norm_ipeak"]))}
        if "domain_distance" in r:
            row["domain_distance"] = float(r["domain_distance"])
            if flag_distance is not None:
                row["confidence"] = ("ok" if r["domain_distance"] < flag_distance
                                     else "EXTRAPOLATING")
        rows.append(row)
    res = pd.DataFrame(rows).sort_values("timepoint").reset_index(drop=True)
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        res.to_parquet(out, index=False)
    return res
