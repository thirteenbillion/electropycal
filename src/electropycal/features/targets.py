"""Target-framing transforms for discovery.

The default featureset has one row per ``(device, channel, timepoint, concentration)`` with
``NormIpeak`` as the target. Because the predictor features (EIS + the FSCV descriptors) are
**dose-invariant** within a sensor-timepoint, that framing predicts a ~flat NormIpeak across dose
and cannot express the sensor's *sensitivity*: the dose-response slope, which is the quantity that
actually drifts and needs recalibrating.

:func:`sensitivity_featureset` reframes to **one row per (device, channel, timepoint)** whose target
is the dose-response **slope** (``sensitivity``), and ``sensitivity_intercept``, obtained by an OLS
fit of ``NormIpeak`` vs ``log10(concentration)`` over that sensor-timepoint's own doses. Feed the
result to ``RunData.from_frame(df, target="sensitivity")`` to run the same discovery queue on the
sensitivity target. Heterogeneous per-device concentrations are handled naturally (the slope is fit
from whatever doses each sensor has), so no common concentration grid is required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.schema import RESERVED_COLUMNS


def _sigma_from_weights(w: "np.ndarray | None", n: int) -> "np.ndarray | None":
    """Turn per-point SNR weights into ``curve_fit`` sigmas (sigma ∝ 1/SNR → high-SNR points weigh
    more). Returns None (unweighted) when weights are missing/degenerate."""
    if w is None:
        return None
    w = np.asarray(w, float)
    if w.shape[0] != n or not np.any(np.isfinite(w) & (w > 0)):
        return None
    w = np.where(np.isfinite(w) & (w > 0), w, np.nanmin(w[np.isfinite(w) & (w > 0)]))
    return 1.0 / np.maximum(w, 1e-9)


def _fit_langmuir(conc, y, w=None) -> tuple[float, float]:
    """Fit the saturation curve ``y = Imax·C/(Kd + C)`` → ``(Imax, Kd)`` (>0), or ``(nan, nan)``.

    Langmuir is the physically-grounded DA calibration curve (adsorption-controlled; Roberts & Sombers
    2013): it **saturates** and stays **monotonic** (invertible, needed to recover dose from a measured
    signal). ``Imax`` ∝ active-site density (drifts with fouling); ``Kd`` is half-saturation. Optional
    per-point SNR weights ``w`` down-weight noisy (low-dose) points. Needs ≥3 distinct positive concs.
    """
    conc = np.asarray(conc, float); y = np.asarray(y, float)
    if np.unique(conc).size < 3 or np.max(y) <= 0:
        return float("nan"), float("nan")
    try:
        from scipy.optimize import curve_fit
        ymax = float(np.max(y))
        popt, _ = curve_fit(lambda c, imax, kd: imax * c / (kd + c), conc, y,
                            p0=(ymax * 1.2, float(np.median(conc))),
                            bounds=([1e-9, 1e-9], [ymax * 100.0, np.max(conc) * 100.0]),
                            sigma=_sigma_from_weights(w, y.size), maxfev=10000)
        return float(popt[0]), float(popt[1])
    except Exception:
        return float("nan"), float("nan")


def _fit_hill(conc, y, w=None) -> tuple[float, float, float]:
    """Fit the Hill saturation curve ``y = Imax·Cⁿ/(Kdⁿ + Cⁿ)`` → ``(Imax, Kd, n)``, or NaNs.

    Hill generalizes Langmuir with a **cooperativity/steepness** exponent ``n`` (n=1 is Langmuir); it is
    monotonic (invertible) for ``Imax, Kd, n > 0``. Three parameters need ≥4 distinct concentrations to
    be identifiable. Optional per-point SNR weights ``w`` down-weight noisy points.
    """
    conc = np.asarray(conc, float); y = np.asarray(y, float)
    if np.unique(conc).size < 4 or np.max(y) <= 0:
        return float("nan"), float("nan"), float("nan")
    try:
        from scipy.optimize import curve_fit
        ymax = float(np.max(y))
        popt, _ = curve_fit(lambda c, imax, kd, n: imax * c ** n / (kd ** n + c ** n), conc, y,
                            p0=(ymax * 1.2, float(np.median(conc)), 1.0),
                            bounds=([1e-9, 1e-9, 0.1], [ymax * 100.0, np.max(conc) * 100.0, 6.0]),
                            sigma=_sigma_from_weights(w, y.size), maxfev=20000)
        return float(popt[0]), float(popt[1]), float(popt[2])
    except Exception:
        return float("nan"), float("nan"), float("nan")


def _fit_power(conc, y, w=None) -> tuple[float, float]:
    """Fit the power law (Freundlich adsorption) ``y = a·Cᵝ`` → ``(a, beta)``, or ``(nan, nan)``.

    Physically the **Freundlich isotherm**: adsorption on a *heterogeneous* surface with a distribution
    of binding-site energies gives coverage θ ∝ C^(1/n), so the faradaic current follows a power law.
    Unlike Langmuir/Hill it does **not** force saturation: ``beta > 1`` is supra-linear (accelerating),
    ``beta = 1`` linear (Henry's law), ``beta < 1`` sub-linear, so it faithfully fits the straight- or
    upward-curving DA dose curves seen on real (rough / fouled) neurostring interfaces, and stays
    monotone-invertible for ``a, beta > 0``. Fit as a log-log line (``beta`` = slope, ``a`` = exp
    intercept) over the positive points; SNR weights down-weight noisy low-dose points. Needs ≥3
    distinct positive concentrations with positive signal.
    """
    conc = np.asarray(conc, float); y = np.asarray(y, float)
    m = np.isfinite(conc) & np.isfinite(y) & (conc > 0) & (y > 0)
    if np.unique(conc[m]).size < 3:
        return float("nan"), float("nan")
    lx = np.log(conc[m]); ly = np.log(y[m])
    pw = None
    if w is not None:
        wv = np.asarray(w, float)[m]
        pw = np.where(np.isfinite(wv) & (wv > 0), wv, 0.0)
        if pw.sum() <= 0:
            pw = None
    try:
        beta, intc = np.polyfit(lx, ly, 1, w=pw)
        return float(np.exp(intc)), float(beta)
    except Exception:
        return float("nan"), float("nan")


def sensitivity_featureset(df: pd.DataFrame, value_col: str = "NormIpeak",
                           conc_col: str = "concentration", min_conc: int = 3,
                           saturation: bool = True, hill: bool = True, power: bool = True,
                           weight_col: str = "repeatability_snr") -> pd.DataFrame:
    """Collapse a per-dose featureset to one dose-response row per sensor-timepoint.

    For each ``(device, channel, timepoint)`` group with at least ``min_conc`` distinct positive
    concentrations, fit ``value_col ~ log10(conc)`` and emit ``sensitivity`` (slope),
    ``sensitivity_intercept``, ``dose_response_r`` and ``n_conc`` alongside the group's
    (dose-invariant) predictor features. Groups with too few concentrations to fit a line are
    dropped, the same ≥3-dose requirement the monotonicity gate already applies.

    ``sensitivity_curvature`` is the quadratic coefficient of a **separate** ``value_col ~ a + b·log10
    (conc) + c·log10(conc)²`` fit (emitted only when ≥4 distinct concentrations make it identifiable;
    NaN otherwise). It captures dose-response *shape* drift (saturation/curving at high dose) that the
    linear slope cannot, and is a candidate target via ``RunData.from_frame(df,
    target="sensitivity_curvature")``. The linear ``sensitivity`` / ``sensitivity_intercept`` are
    unchanged (still from the degree-1 fit), so the default sensitivity framing is unaffected.

    With ``saturation=True`` (default) it also fits the adsorption-controlled **Langmuir** curve
    ``NormIpeak = Imax·C/(Kd + C)`` (``sat_imax``, ``sat_kd``, ``sat_logkd``) and, with ``hill=True``,
    the **Hill** curve ``Imax·Cⁿ/(Kdⁿ + Cⁿ)`` (``hill_imax``, ``hill_kd``, ``hill_n``), *monotone,
    invertible* calibration models that fit real (saturating) DA dose-responses better than a line or a
    (non-monotone) quadratic. ``sat_imax``/``hill_imax`` ∝ active-site density (drifts with fouling);
    they are positive-magnitude candidate targets via ``RunData.from_frame(df, target="sat_imax")``.

    **SNR-weighted fits.** When a ``weight_col`` (default ``repeatability_snr``) is present, every
    per-sensor-timepoint curve fit (linear slope/intercept, quadratic, Langmuir, Hill) is weighted by
    each dose's SNR, so noisy low-dose points are down-weighted: a cleaner target than the unweighted
    fit. This is *separate* from ``weighted_plsr`` (which weights samples in the discovery model, not
    the dose-response fit that produces the target).

    Returns a DataFrame ready for ``RunData.from_frame(df, target="sensitivity")``.
    """
    import pandas as pd
    id_cols = [c for c in ("device", "channel", "timepoint") if c in df.columns]
    if "channel" not in id_cols or "timepoint" not in id_cols:
        raise ValueError("sensitivity_featureset needs 'channel' and 'timepoint' columns")
    # NUMERIC predictor columns only: stray label columns (e.g. a 'devicetype'/'sensor' string added
    # downstream) are not features and would break the per-group mean.
    feat_cols = [c for c in df.columns if c not in set(RESERVED_COLUMNS) and c != "time_index"
                 and pd.api.types.is_numeric_dtype(df[c])]
    has_w = weight_col in df.columns

    rows: list[dict] = []
    for key, g in df.groupby(id_cols):
        cc = g[conc_col].to_numpy(float)
        y = g[value_col].to_numpy(float)
        keep = np.isfinite(cc) & np.isfinite(y) & (cc > 0)
        cc, y = cc[keep], y[keep]
        w = g[weight_col].to_numpy(float)[keep] if has_w else None       # per-dose SNR weights
        wv = _sigma_from_weights(w, y.size)                              # None when unusable
        pw = (1.0 / wv ** 2) if wv is not None else None                # polyfit weights ∝ SNR²
        if np.unique(cc).size < min_conc:
            continue                                       # can't fit a dose-response line
        x = np.log10(cc)
        slope, intercept = np.polyfit(x, y, 1, w=pw)
        r = float(np.corrcoef(x, y)[0, 1]) if np.std(y) > 0 else float("nan")
        # quadratic shape term (separate fit); needs >=4 distinct doses to be identifiable
        if np.unique(cc).size >= 4:
            curv = float(np.polyfit(x, y, 2, w=pw)[0])
        else:
            curv = float("nan")
        row = dict(zip(id_cols, key if isinstance(key, tuple) else (key,)))
        for c in feat_cols:                                # dose-invariant -> mean is exact
            row[c] = float(g[c].mean())
        row.update(sensitivity=float(slope), sensitivity_intercept=float(intercept),
                   sensitivity_curvature=curv, dose_response_r=r, n_conc=int(np.unique(cc).size))
        if saturation:
            # monotone, invertible saturation targets (physically-grounded; SNR-weighted).
            imax, kd = _fit_langmuir(cc, y, w)
            row.update(sat_imax=imax, sat_kd=kd, sat_logkd=float(np.log10(kd)) if kd > 0 else float("nan"))
        if hill:
            himax, hkd, hn = _fit_hill(cc, y, w)
            row.update(hill_imax=himax, hill_kd=hkd, hill_n=hn)
        if power:
            # Freundlich power law y = a·C^beta: non-saturating (fits straight / upward curves),
            # monotone-invertible. beta>1 supra-linear, =1 linear, <1 sub-linear.
            pa, pb = _fit_power(cc, y, w)
            row.update(power_a=pa, power_beta=pb)
        rows.append(row)
    return pd.DataFrame(rows)
