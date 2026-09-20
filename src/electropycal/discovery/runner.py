"""Nested-CV execution of one testing condition.

Outer forward-chained folds evaluate; an inner forward-chained split selects
hyperparameters (selector threshold × ``k``) using outer-training data only; the
chosen configuration is refit on the full outer-training set and scored on the
untouched outer-test fold. Each fold serializes a portable asset bundle
(``model_arrays.npz`` + ``hyperparams.json`` + ``metrics.json``) into a per-fold
directory layout; per-condition aggregation writes pooled metrics + feature
stability.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _mean_aggregates(aggs: list[dict]) -> dict:
    """Average per-seed track aggregates for a multi-seed run. Numeric scalar keys common to all
    seeds are mean-reduced (ignoring NaN); non-numeric or first-only keys (e.g. ``track``,
    ``n_folds``, nested per-channel tables) are taken from seed[0] unchanged. Keeps each seed's
    aggregate statistically clean (no duplicate-row pooling) and reports the seed-mean metric."""
    import numpy as _np
    base = dict(aggs[0])
    common = set(base).intersection(*(set(a) for a in aggs[1:]))
    for key in common:
        vals = [a[key] for a in aggs]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            if key in ("n_folds", "n_test"):                 # counts: identical across seeds, keep as-is
                continue
            finite = [float(v) for v in vals if _np.isfinite(v)]
            base[key] = float(_np.mean(finite)) if finite else float("nan")
    return base

from ..data.io import write_json, write_parquet
from ..evaluation.cv import inner_split, outer_folds
from ..evaluation.metrics import FoldResult
from ..evaluation.tracks import aggregate_by_track
from ..features.normalize import (apply_median_impute, apply_zscore,
                                  fit_median_impute, fit_zscore)
from ..models.base import save_model_bundle
from ..models.variants import build
from ..selection import cars, icc, univariate as uni
from ..selection import pseudo_multivariate as pm
from .config import Condition, Profile, RunData, effective_min_train_times


def _weights(condition: Condition, data: RunData, idx: np.ndarray):
    """Per-training-row sample weights for the given condition, or None (unweighted)."""
    if condition.weighted_by == "concentration":
        return 1.0 / np.maximum(data.concentration[idx], 1.0)
    if condition.weighted_by == "repeatability_snr":
        if data.repeatability_snr is None:
            return None                                   # featureset carried no SNR -> unweighted
        w = np.asarray(data.repeatability_snr[idx], float)
        fin = w[np.isfinite(w)]
        cap = float(fin.max()) if fin.size else 1.0       # perfectly-reproducible (inf) -> cap, not drop
        w = np.clip(np.where(np.isfinite(w), w, cap), 0.0, None)
        return w if w.sum() > 0 else None
    return None


def _select(condition: Condition, threshold: float, X: np.ndarray, y: np.ndarray,
            meta: dict, cars_kw: dict, seed: int) -> np.ndarray:
    """Feature indices selected on ``X`` (already Z-scored) per the condition."""
    sel = condition.selector
    if sel is None:
        return np.arange(X.shape[1])
    if sel in ("vip", "sr", "smc"):
        return pm.select(sel, X, y, k=meta["k"], threshold=threshold)
    if sel == "mi":
        return uni.mi_select(X, y, threshold, random_state=seed)
    if sel == "icc":
        return icc.icc_prefilter(X, meta["channel"], meta["timepoint"],
                                 meta["concentration"], threshold)
    if sel == "cars":
        return cars.cars_select(X, y, random_state=seed, **cars_kw)
    if sel == "icc+cars":
        pre = icc.icc_prefilter(X, meta["channel"], meta["timepoint"],
                                meta["concentration"], threshold)
        sub = cars.cars_select(X[:, pre], y, random_state=seed, **cars_kw)
        return pre[np.asarray(sub, dtype=int)]
    raise KeyError(f"unknown selector {sel!r}")


def _fit_predict(condition: Condition, k: int, Xtr, ytr, wtr, Xte):
    arch = build(condition.architecture, k)
    if condition.transform == "log":
        # log model: fit in log(target) space, exp predictions back to NormIpeak so
        # metrics stay on the NormIpeak scale (features are already log(x/d0) via run_condition).
        arch.fit(Xtr, np.log(np.maximum(np.asarray(ytr, float), 1e-12)), sample_weight=wtr)
        # clip the log-space prediction before exp so a badly-extrapolating fold can't overflow to
        # inf and poison the pooled metric (exp(50) ~ 5e21 is already far past any physical NormIpeak)
        return arch, np.exp(np.clip(arch.predict(Xte).ravel(), -50.0, 50.0))
    arch.fit(Xtr, ytr, sample_weight=wtr)
    return arch, arch.predict(Xte).ravel()


def _inner_best(condition: Condition, fold, data: RunData, profile: Profile, seed: int):
    """Inner-loop hyperparameter selection (returns best {threshold, k} or None)."""
    split = inner_split(data.channel, data.timepoint, fold, track=condition.inner_track())
    if split is None:
        return None
    itr, ival = split
    # Impute, then Z-score, both fitted on the INNER-training rows only and applied to the
    # inner-validation rows. RunData.from_frame leaves missing cells in place precisely so the
    # statistic can be fitted here rather than over the whole featureset.
    med = fit_median_impute(data.X[itr])
    Xi_tr, Xi_val = apply_median_impute(data.X[itr], med), apply_median_impute(data.X[ival], med)
    mean, std = fit_zscore(Xi_tr)
    Xitr, Xival = apply_zscore(Xi_tr, mean, std), apply_zscore(Xi_val, mean, std)
    thresholds = condition.threshold_grid if condition.has_threshold() else (0.0,)
    sel_k = max(condition.k_grid)
    best = None
    for thr in thresholds:
        meta = {"k": sel_k, "channel": data.channel[itr], "timepoint": data.timepoint[itr],
                "concentration": data.concentration[itr]}
        try:
            subset = _select(condition, thr, Xitr, data.y[itr], meta, profile.cars, seed)
        except Exception:                                    # degenerate fold (e.g. rank-0 features) → skip
            continue
        if subset.size == 0:
            continue
        for k in condition.k_grid:
            kk = min(k, min(len(itr) - 1, len(subset)))
            if kk < 1 or len(itr) < 2 * kk:
                continue
            try:
                _, pred = _fit_predict(condition, kk, Xitr[:, subset], data.y[itr],
                                       _weights(condition, data, itr), Xival[:, subset])
            except Exception:
                continue
            fr = FoldResult.from_predictions(fold.channel, fold.t_test, data.y[ival], pred,
                                             float(data.y[itr].mean()))
            rmsep = np.sqrt(fr.sse / fr.n_test) if fr.n_test else np.inf
            if best is None or rmsep < best["rmsep"]:
                best = {"threshold": thr, "k": kk, "rmsep": float(rmsep)}
    return best


def _run_fold(condition: Condition, fold, data: RunData, profile: Profile, seed: int):
    """Evaluate one outer fold (no file writes) → payload dict, or None if unusable.

    Pure/return-only so it can run in a joblib worker; the parent writes files.
    """
    best = _inner_best(condition, fold, data, profile, seed)
    if best is None:
        return None
    thr, k = best["threshold"], best["k"]
    tr, te = fold.train_idx, fold.test_idx
    if len(tr) < 2 * k:
        return None
    # Same discipline on the outer fold: the imputation median and the Z-score statistics both
    # come from the outer-training rows and are applied unchanged to the untouched test fold.
    med = fit_median_impute(data.X[tr])
    X_tr_i, X_te_i = apply_median_impute(data.X[tr], med), apply_median_impute(data.X[te], med)
    mean, std = fit_zscore(X_tr_i)
    Xtr, Xte = apply_zscore(X_tr_i, mean, std), apply_zscore(X_te_i, mean, std)
    meta = {"k": k, "channel": data.channel[tr], "timepoint": data.timepoint[tr],
            "concentration": data.concentration[tr]}
    try:
        subset = _select(condition, thr, Xtr, data.y[tr], meta, profile.cars, seed)
        kk = min(k, min(len(tr) - 1, len(subset)))
        arch, pred = _fit_predict(condition, kk, Xtr[:, subset], data.y[tr],
                                  _weights(condition, data, tr), Xte[:, subset])
    except Exception:                                        # degenerate outer fold → skip (rare; inner CV passed)
        return None
    fr = FoldResult.from_predictions(fold.channel, fold.t_test, data.y[te], pred,
                                     float(data.y[tr].mean()))
    arrays = arch.to_arrays()
    arrays.update(zscore_mean=mean, zscore_std=std, feature_index=subset.astype(int))
    rmsep = float(np.sqrt(fr.sse / fr.n_test))
    return {
        "fold_name": f"ch{fold.channel}_t{int(fold.t_test)}",
        "fr": fr, "subset": subset, "arrays": arrays,
        "manifest": {**arch.manifest(), "threshold": thr, "n_selected": int(subset.size)},
        "hyperparams": {"threshold": thr, "k": kk, "feature_index": subset.astype(int).tolist()},
        "metrics": {"sse": fr.sse, "n_test": fr.n_test, "tss": fr.tss, "rmsep": rmsep,
                    "k": kk, "threshold": thr, "n_selected": int(subset.size)},
        "row": {"condition": condition.name, "channel": fold.channel, "t_test": fold.t_test,
                "rmsep": rmsep, "n_test": fr.n_test, "k": kk, "threshold": thr,
                "n_selected": int(subset.size)},
        "preds": {"sensor": int(fold.channel), "t_test": float(fold.t_test),
                  "concentration": np.asarray(data.concentration[te], float),
                  "y_true": np.asarray(data.y[te], float), "y_pred": np.asarray(pred, float)},
    }


def run_condition(condition: Condition, data: RunData, out_dir: str | Path,
                  profile: Profile, seed: int = 0,
                  progress: bool = False, cap_min_train_times: bool = True) -> tuple[list[dict], dict]:
    """Run one condition end-to-end; write its outputs; return (summary rows, agg).

    ``progress=True`` emits timestamped ``[HH:MM:SS]`` lines: a START line (fold count,
    track, grids), one line per outer fold with a running elapsed/ETA estimate (serial
    ``n_jobs=1`` only — parallel folds finish out of order, so just START/DONE are logged),
    and a DONE line with the pooled RMSEP and wall time. Use it to track a long full-profile
    run and estimate remaining time.

    ``cap_min_train_times`` (**default True**) caps ``profile.min_train_times`` to what the
    data's timepoint count can support (:func:`~electropycal.discovery.config.effective_min_train_times`),
    warning when it fires — so a short series does not silently yield 0 folds / nan. Set
    ``False`` to run with the profile's value verbatim.
    """
    out_dir = Path(out_dir)
    cond_dir = out_dir / "conditions" / condition.name
    if condition.transform == "log":
        # full log model on ANY architecture: log(x/d0) for multiplicative features
        # (additive unchanged), and log-target fitting handled in _fit_predict.
        if condition.architecture == "log_plsr":
            raise ValueError("transform='log' is the full log model; pair it with a linear-output "
                             "architecture (e.g. 'linear_plsr'), not 'log_plsr' which only logs the "
                             "target and would double-transform.")
        if np.any(np.asarray(data.y, float) <= 0):
            # the log model fits log(target); a signed target (e.g. a calibration-curve intercept or
            # curvature) would be silently clamped to 1e-12 and corrupted. NormIpeak and the sensitivity
            # SLOPE are positive magnitudes (log-appropriate); intercept/curvature are signed → linear only.
            raise ValueError("transform='log' requires a strictly positive target, but the target has "
                             "non-positive values (e.g. a signed calibration-curve intercept/curvature). "
                             "Use a linear condition (transform='linear') for signed targets.")
        import dataclasses

        from ..features.normalize import to_log_representation
        data = dataclasses.replace(data, X=to_log_representation(data.X, list(data.feature_names)))
    mtt = profile.min_train_times
    if cap_min_train_times:
        n_tp = int(np.unique(np.asarray(data.timepoint)).size)
        mtt = effective_min_train_times(profile.min_train_times, n_tp)
        if mtt != profile.min_train_times:
            import warnings
            warnings.warn(
                f"min_train_times={profile.min_train_times} needs >= {profile.min_train_times + 1} "
                f"distinct timepoints but the data has {n_tp}; capping to {mtt} so forward-chained "
                f"CV yields folds (otherwise 0 folds -> nan). Pass cap_min_train_times=False to opt out.",
                stacklevel=2)
            if progress:
                _log(f"min_train_times capped {profile.min_train_times} -> {mtt} ({n_tp} timepoints)")
    folds = list(outer_folds(data.channel, data.timepoint, mode=condition.cv_mode(),
                             min_train_times=mtt, seed=seed))
    t0 = time.time()
    # Multi-seed only matters for the STOCHASTIC selectors (CARS / MI use random_state); every other
    # architecture+selector is deterministic, so repeating it under new seeds is pure waste — collapse
    # to one seed there. When >1 seed is set for a stochastic selector, repeat the whole nested-CV per
    # seed and average the aggregates + selection frequencies (seed[0]'s per-fold bundles are the ones
    # written to disk). Default profile.seeds=(0,) -> single seed -> byte-identical to before.
    _stochastic = condition.selector in ("cars", "icc+cars", "mi")
    seeds = tuple(dict.fromkeys(profile.seeds)) if (_stochastic and len(set(profile.seeds)) > 1) \
        else (seed,)
    if progress:
        _log(f"START {condition.name}  ({condition.architecture}, track={condition.track}) — "
             f"{len(folds)} folds, k_grid={condition.k_grid}, "
             f"seeds={seeds}{' (selector deterministic -> 1 seed)' if not _stochastic and len(set(profile.seeds)) > 1 else ''}, "
             f"n_boot={profile.n_boot}, n_jobs={profile.n_jobs}")

    parallel = profile.n_jobs and profile.n_jobs != 1 and len(folds) > 1

    def _fold_payloads(sd: int) -> list:
        if parallel:
            # loky process backend; folds are independent. map_sessions caps each worker to one
            # BLAS/OpenMP thread: without that cap every one of n_jobs workers opens its own
            # multi-threaded BLAS pool and they contend for the same cores, so the "speedup" can
            # be a slowdown -- the benchmark had to pin OMP_NUM_THREADS and friends in the
            # workflow to get interpretable timings. The library now defends itself, as the raw
            # walks already did.
            from joblib import delayed

            from .._parallel import map_sessions
            return map_sessions(profile.n_jobs, (
                delayed(_run_fold)(condition, f, data, profile, sd) for f in folds))
        out = []
        for i, f in enumerate(folds, 1):
            out.append(_run_fold(condition, f, data, profile, sd))
            if progress and len(seeds) == 1:      # per-fold ETA only when serial + single-seed
                el = time.time() - t0
                eta = el / i * (len(folds) - i)
                _log(f"  {condition.name}: fold {i}/{len(folds)}  ch{f.channel} t{int(f.t_test)}"
                     f"  ({el:.0f}s elapsed, ETA {eta:.0f}s)")
        return out

    payloads = None                               # seed[0]'s payloads -> the ones written to disk
    seed_aggs: list[dict] = []
    seed_freqs: list[np.ndarray] = []
    for _si, _sd in enumerate(seeds):
        _pl = _fold_payloads(_sd)
        if _si == 0:
            payloads = _pl
        _frs = [p["fr"] for p in _pl if p is not None]
        _subs = [p["subset"] for p in _pl if p is not None]
        seed_aggs.append(aggregate_by_track(_frs, condition.track, n_boot=profile.n_boot))
        seed_freqs.append(cars.selection_stability(_subs, data.X.shape[1]) if _subs
                          else np.zeros(data.X.shape[1]))
        if progress and len(seeds) > 1:
            _log(f"  {condition.name}: seed {_sd} ({_si + 1}/{len(seeds)}) "
                 f"pooled_rmsep={seed_aggs[-1].get('pooled_rmsep', float('nan')):.4f} "
                 f"({time.time() - t0:.0f}s elapsed)")

    folds_results: list[FoldResult] = []          # seed[0]'s folds (for the DONE-log count)
    rows: list[dict] = []
    for p in payloads:
        if p is None:
            continue
        folds_results.append(p["fr"])
        rows.append(p["row"])
        fold_dir = cond_dir / "folds" / p["fold_name"]
        save_model_bundle(fold_dir, p["arrays"], p["manifest"])
        write_json(fold_dir / "hyperparams.json", p["hyperparams"])
        write_json(fold_dir / "metrics.json", p["metrics"])

    # per-test-row predictions (for calibration / residual review), split back to device + channel
    pred_frames = []
    for p in payloads:
        if p is None:
            continue
        pr = p["preds"]
        pred_frames.append(pd.DataFrame({
            "sensor": pr["sensor"], "t_test": pr["t_test"], "concentration": pr["concentration"],
            "y_true": pr["y_true"], "y_pred": pr["y_pred"]}))
    if pred_frames:
        preds_df = pd.concat(pred_frames, ignore_index=True)
        if data.sensor_labels is not None:
            lab = preds_df["sensor"].map(lambda s: str(data.sensor_labels[s]))
            preds_df["device"] = lab.str.split(":").str[0]
            preds_df["channel"] = lab.str.split(":").str[-1]
        else:
            preds_df["device"] = ""
            preds_df["channel"] = preds_df["sensor"].astype(str)
        preds_df["residual"] = preds_df["y_pred"] - preds_df["y_true"]
        write_parquet(cond_dir / "predictions.parquet", preds_df)

    # single seed -> the one aggregate verbatim (unchanged behavior); multi-seed -> average the
    # per-seed aggregates (scalar metrics) and selection frequencies (over folds x seeds).
    if len(seeds) == 1:
        agg = seed_aggs[0]
    else:
        agg = _mean_aggregates(seed_aggs)
    # The effective set, always: it differs from profile.seeds whenever the selector is
    # deterministic (collapsed to one seed), so recording only the request would misstate what ran.
    agg["n_seeds"] = len(seeds)
    agg["seeds"] = list(seeds)
    freq = np.mean(seed_freqs, axis=0)
    if progress:
        _log(f"DONE  {condition.name} — pooled_rmsep={agg.get('pooled_rmsep', float('nan')):.4f} "
             f"over {len(folds_results)} usable folds"
             f"{f' x {len(seeds)} seeds' if len(seeds) > 1 else ''} in {time.time() - t0:.0f}s")
    write_json(cond_dir / "aggregated_metrics.json", agg)
    write_parquet(cond_dir / "feature_stability.parquet",
                  pd.DataFrame({"feature": data.feature_names, "selection_frequency": freq}))
    write_json(cond_dir / "condition_config.json", {
        "name": condition.name, "architecture": condition.architecture, "track": condition.track,
        "selector": condition.selector, "k_grid": list(condition.k_grid),
        "threshold_grid": list(condition.threshold_grid),
        "seeds": list(seeds), "stochastic_selector": bool(_stochastic),
        "fold_seed": int(seed)})
    return rows, agg
