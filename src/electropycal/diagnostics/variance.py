"""Variance diagnostics: where does a feature's variance actually live?

Two offline, leakage-free data characterizations run once on all valid rows:

- ``variance_hierarchy``: the decomposition of each feature's variance
  into between-channel / within-channel-temporal / within-channel-concentration
  components (structural vs dynamic signal).
- ``measurement_reliability``: the diagnostic that *bounds* Assumption
  5: per-feature reliability ``R = σ²_signal / σ²_obs`` from replicate spread,
  the fraction of modeled variance that is real signal rather than measurement
  noise. ``R → 1`` ⇒ the error-free assumption is defensible for that feature.
"""

from __future__ import annotations

import numpy as np


def variance_hierarchy(
    values: np.ndarray,
    channel: np.ndarray,
    timepoint: np.ndarray,
    concentration: np.ndarray,
) -> dict[str, float]:
    """Sequential (nested-ANOVA) variance split for one feature across observations.

    Returns fractions summing to 1: ``between_channel`` (n_C), ``within_temporal``
    (n_cT, after removing n_C), ``within_concentration`` (n_cD, residual). Mirrors
    a sequential ``SS = Σ (v_i − v_mean)²`` decomposition: each level's sum of squares is
    taken after the previous level's group means have been removed, so the fractions are
    order-dependent by construction and must be read in the stated order.
    """
    values = np.asarray(values, dtype=float)
    grand = values.mean()
    total = float(((values - grand) ** 2).sum())
    if total == 0:
        return {"between_channel": 0.0, "within_temporal": 0.0, "within_concentration": 0.0}

    ch_mean = _group_means(values, channel)
    ss_channel = float(((ch_mean - grand) ** 2).sum())

    ct_key = _combine(channel, timepoint)
    ct_mean = _group_means(values, ct_key)
    ss_ct = float(((ct_mean - ch_mean) ** 2).sum())  # temporal, after removing channel

    ss_conc = total - ss_channel - ss_ct  # residual (between concentrations)
    return {
        "between_channel": ss_channel / total,
        "within_temporal": ss_ct / total,
        "within_concentration": max(0.0, ss_conc) / total,
    }


def measurement_reliability(
    replicate_values: np.ndarray,
    observation_id: np.ndarray,
    n_rep: float | None = None,
) -> dict[str, float]:
    """Per-feature measurement reliability from replicate spread.

    Parameters
    ----------
    replicate_values : (M,) feature value computed *per replicate* (on the modeled
        scale, post D0/log), for all replicates of all observations.
    observation_id : (M,) integer id grouping replicates into observations.
    n_rep : replicates per observation used for the mean (defaults to the mean
        group size); enters the ``σ²_meas/n_rep`` sampling term.

    Returns ``sigma2_meas`` (pooled within-observation variance), ``sigma2_obs``
    (between-observation variance of the replicate means), ``sigma2_signal``
    (noise-corrected), and ``reliability`` R ∈ [0, 1].
    """
    replicate_values = np.asarray(replicate_values, dtype=float)
    observation_id = np.asarray(observation_id)
    obs_ids = np.unique(observation_id)

    # Pooled within-observation (measurement) variance: Σ Σ (x − x̄_obs)² / Σ (n_obs − 1)
    ss_within = 0.0
    df_within = 0
    means = np.empty(obs_ids.size)
    sizes = np.empty(obs_ids.size)
    for i, oid in enumerate(obs_ids):
        vals = replicate_values[observation_id == oid]
        m = vals.mean()
        means[i] = m
        sizes[i] = vals.size
        ss_within += float(((vals - m) ** 2).sum())
        df_within += vals.size - 1
    sigma2_meas = ss_within / df_within if df_within > 0 else 0.0

    # Between-observation variance of the replicate means.
    sigma2_obs = float(means.var(ddof=1)) if means.size > 1 else 0.0

    if n_rep is None:
        n_rep = float(sizes.mean()) if sizes.size else 1.0
    sigma2_signal = max(0.0, sigma2_obs - sigma2_meas / n_rep)
    reliability = sigma2_signal / sigma2_obs if sigma2_obs > 0 else 0.0
    return {
        "sigma2_meas": sigma2_meas,
        "sigma2_obs": sigma2_obs,
        "sigma2_signal": sigma2_signal,
        "reliability": reliability,
    }


def drift_reliability(
    values: np.ndarray,
    sensor: np.ndarray,
    timepoint: np.ndarray,
    concentration: np.ndarray,
    sigma2_meas: float,
    n_rep: float,
) -> dict[str, float]:
    """Is a feature's within-sensor **temporal (drift)** variance above its measurement-noise floor?

    Combines the Phase-0 ``within_temporal`` component (the drift *signal*, from
    :func:`variance_hierarchy`) with the Phase-0.5 replicate variance ``sigma2_meas`` (the *noise*,
    from :func:`measurement_reliability`). Unlike the whole-response reliability (whose ``σ²_obs`` is
    dominated by the dose-response), this targets the drift slice the recalibration model actually
    learns from. For EIS features (dose-invariant) the within-temporal component *is* the drift, so
    this is well-defined; for the response it mixes the dose-averaged level shift (interpret with care).

    Returns ``sigma2_drift`` (absolute within-temporal variance), ``noise_floor`` = ``sigma2_meas /
    n_rep`` (noise in a replicate-averaged observation), ``drift_snr`` = ``sigma2_drift / noise_floor``,
    and ``R_drift`` = ``max(0, sigma2_drift − noise_floor) / sigma2_drift ∈ [0, 1]``. ``R_drift → 1``
    (and a non-trivial ``sigma2_drift``) ⇒ real, resolvable drift; ``drift_snr ≲ 1`` ⇒ the drift is at
    or below the noise floor and the model has little to learn from that feature.
    """
    values = np.asarray(values, dtype=float)
    total = float(values.var())                              # ddof=0 ⇒ SS/N, matches the hierarchy
    if not np.isfinite(total) or total == 0:                 # constant feature ⇒ reliability undefined
        return {"sigma2_drift": 0.0, "sigma2_meas": float(sigma2_meas), "noise_floor": float("nan"),
                "drift_snr": float("nan"), "R_drift": float("nan")}
    h = variance_hierarchy(values, sensor, timepoint, concentration)
    sigma2_drift = h["within_temporal"] * total
    noise = sigma2_meas / n_rep if n_rep else float("nan")
    if noise and noise > 0:
        snr = sigma2_drift / noise
    else:
        snr = float("inf") if sigma2_drift > 0 else float("nan")
    R = max(0.0, sigma2_drift - noise) / sigma2_drift if sigma2_drift > 0 else 0.0
    return {"sigma2_drift": sigma2_drift, "sigma2_meas": float(sigma2_meas),
            "noise_floor": float(noise), "drift_snr": float(snr), "R_drift": float(R)}


def drift_alignment(
    feature: np.ndarray,
    response: np.ndarray,
    sensor: np.ndarray,
    timepoint: np.ndarray,
    concentration: np.ndarray,
    method: str = "pearson",
) -> dict[str, float]:
    """Does a feature's drift **track the response's drift**? The decisive pre-discovery screen.

    A feature can drift cleanly yet be irrelevant to the response. Within each ``(sensor,
    concentration)`` cell the observations form a short time series across timepoints; the Pearson
    ``r`` of ``feature`` vs ``response`` there is how the feature co-moves with the response **as that
    sensor drifts at a fixed dose**. Per-cell correlations are pooled across cells by Fisher-``z``
    averaging. Only cells with ≥3 timepoints and nonzero variance in both series contribute.

    ``method='pearson'`` (default) correlates the values directly (linear co-drift); ``'spearman'``
    correlates their ranks (monotonic co-drift, robust to nonlinear-but-monotonic co-movement).

    Returns ``alignment_r`` (pooled correlation, back-transformed), ``abs_alignment`` = ``|r|``, and
    ``n_cells``. A near-zero ``abs_alignment`` across *all* features means no predictor's drift
    explains the response's drift; recalibration is unlikely to work regardless of architecture.
    """
    if method not in ("pearson", "spearman"):
        raise ValueError(f"method must be 'pearson' or 'spearman', got {method!r}")
    feature = np.asarray(feature, dtype=float)
    response = np.asarray(response, dtype=float)
    timepoint = np.asarray(timepoint)
    key = _combine(np.asarray(sensor).astype(str), np.asarray(concentration).astype(str))
    zs: list[float] = []
    for k in np.unique(key):
        m = key == k
        if int(m.sum()) < 3:
            continue
        # collapse replicate rows at the same timepoint to the timepoint mean, then correlate over time
        t = timepoint[m]
        xf = _group_means(feature[m], t)
        yr = _group_means(response[m], t)
        uniq = np.unique(t)
        if uniq.size < 3:
            continue
        xs = np.array([xf[t == u][0] for u in uniq])
        ys = np.array([yr[t == u][0] for u in uniq])
        if np.std(xs) == 0 or np.std(ys) == 0:
            continue
        if method == "spearman":
            from scipy.stats import rankdata
            xs, ys = rankdata(xs), rankdata(ys)
        r = float(np.corrcoef(xs, ys)[0, 1])
        r = max(min(r, 0.999), -0.999)
        zs.append(float(np.arctanh(r)))
    if not zs:
        return {"alignment_r": float("nan"), "abs_alignment": float("nan"), "n_cells": 0}
    r = float(np.tanh(np.mean(zs)))
    return {"alignment_r": r, "abs_alignment": abs(r), "n_cells": len(zs)}


def _group_means(values: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Vector of per-observation group means (same length as ``values``)."""
    keys = np.asarray(keys)
    out = np.empty_like(values, dtype=float)
    for k in np.unique(keys):
        mask = keys == k
        out[mask] = values[mask].mean()
    return out


def _combine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Combine two label arrays into a single composite key array."""
    a = np.asarray(a).astype(str)
    b = np.asarray(b).astype(str)
    return np.char.add(np.char.add(a, "|"), b)
