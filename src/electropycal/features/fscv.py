"""FSCV feature extraction: fast-scan cyclic voltammetry → dopamine response features.

Background subtraction → voltage window → reference level → peak → NormIpeak, plus
predictor features ``mean_Vpeak`` and ``mean_Ibg``. Cycles are current arrays
sampled on a shared ``voltages`` axis; pass per-replicate cycles to average (and
to feed the reliability diagnostic).

``NormIpeak = peak_height(V_ox) / I_bgd(V_ox)`` where ``peak_height`` is the
voltage-axis-perpendicular (vertical) distance from the peak to the reference line
at ``V_ox``, and ``I_bgd(V_ox)`` is the background current at the oxidation voltage.
Normalizing by the background current makes the response comparable across electrodes
whose absolute currents differ by orders of magnitude.
"""

from __future__ import annotations

import numpy as np

DA_WINDOW = (0.6, 0.8)            # nominal dopamine → dopamine-o-quinone oxidation, shaded in plots
#: default window the peak search (V_ox / I_peak) runs over. Widened from the nominal 0.6-0.8 V
#: DA region because real DA peaks are broad and often crest above 0.8 V; the upper bound stops
#: below the ~1.0+ V anodic-limit / switching surge (non-faradaic) so the peak-finder tracks the
#: DA peak, not the switching edge. Calibrate to your electrodes.
PEAK_WINDOW = (0.4, 1.0)
#: window the RMS noise floor is measured over; it MUST stay below the faradaic onset (i.e. below
#: the peak-search window) or it counts real DA signal as "noise". Narrowed to (0.0, 0.3): on real
#: data the background-subtraction residual is smallest/flattest here, so this is the cleanest
#: non-faradaic stretch. (A wider (-0.1, 0.6) window overlaps the DA region and so measures signal
#: as noise; even (-0.1, 0.4) still catches the residual rising near 0.4 V.) Note the RMS floor is a *residual* estimate, not random noise; for a
#: reproducibility-based SNR see :func:`repeatability_snr`. Calibrate to your electrodes' onset.
NONFARADAIC_WINDOW = (0.0, 0.3)

#: a located V_ox within this many volts of either ``PEAK_WINDOW`` bound is treated as "edge-pinned"
#: (~3 samples at a 10 mV step): the argmax ran into the window boundary, so the true peak likely
#: lies outside the window and the height/NormIpeak is a window artifact rather than a resolved peak.
PEAK_EDGE_TOL = 0.03


def peak_at_edge(v_ox: float, v_window: tuple[float, float] = PEAK_WINDOW,
                 tol: float = PEAK_EDGE_TOL) -> bool:
    """True if ``v_ox`` sits within ``tol`` of either end of the peak-search window.

    Flags a peak the finder could not resolve inside the window (it pinned to a boundary),
    as is common at low DA concentrations, whose broad peaks shift toward / below the lower edge.
    A non-finite ``v_ox`` (no peak found) returns ``False``.
    """
    if v_ox is None or not np.isfinite(v_ox):
        return False
    lo, hi = v_window
    return bool(v_ox <= lo + tol or v_ox >= hi - tol)


#: fraction of the peak height still present at a window bound above which the faradaic lobe is judged
#: **clipped** (truncated by the sweep/window edge rather than returning to baseline). At 0.5, the current
#: at the edge is still ≥ half the peak: the lobe demonstrably extends past the boundary.
PEAK_CLIP_FRAC = 0.5


def peak_edge_clipped(bg_sub: np.ndarray, voltages: np.ndarray, height: float,
                      v_window: tuple[float, float] = PEAK_WINDOW,
                      clip_frac: float = PEAK_CLIP_FRAC) -> dict:
    """Detect a faradaic lobe **truncated by a window/sweep edge** (distinct from ``peak_at_edge``).

    ``peak_at_edge`` flags where the *located V_ox* pinned to a bound. This instead asks whether the
    *lobe itself* is cut off: even with V_ox interior, a broad DA peak (common at **low dose**) can still
    be rising at the upper sweep limit (~1.0 V) so its tail (and its **area**) is lost. Rule: the
    background-subtracted current at a window bound is ``≥ clip_frac × height`` (the lobe has not returned
    toward baseline there). Returns ``{clipped_high, clipped_low, clipped}`` (bools). All-False when there
    is no finite peak. This is a hard limit of the max voltage sweep span, not something more filtering can
    fix, but it can be **flagged** so clipped ``peak_area`` values are excluded rather than trusted.
    """
    out = {"clipped_high": False, "clipped_low": False, "clipped": False}
    if not np.isfinite(height) or height <= 0:
        return out
    v = np.asarray(voltages, float); y = np.asarray(bg_sub, float)
    lo, hi = v_window
    idxs = np.where((v >= lo) & (v <= hi))[0]
    if idxs.size < 2:
        return out
    yw = y[idxs]
    thr = clip_frac * height
    out["clipped_low"] = bool(np.isfinite(yw[0]) and yw[0] >= thr)
    out["clipped_high"] = bool(np.isfinite(yw[-1]) and yw[-1] >= thr)
    out["clipped"] = out["clipped_low"] or out["clipped_high"]
    return out


def anodic_sweep(voltages: np.ndarray, *arrays: np.ndarray):
    """Restrict a full FSCV cycle to its anodic (rising-voltage) sweep.

    Real FSCV is a triangular sweep (up then down); DA oxidation occurs on the
    rising half, so feature extraction uses that segment (full-cycle plotting uses
    the raw arrays). Monotonic-increasing inputs are returned unchanged.
    """
    v = np.asarray(voltages, float)
    imin, imax = int(np.argmin(v)), int(np.argmax(v))
    sl = slice(min(imin, imax), max(imin, imax) + 1)
    vv = v[sl]
    out = [np.asarray(a, float)[sl] for a in arrays]
    if vv.size and vv[0] > vv[-1]:
        vv = vv[::-1]
        out = [a[::-1] for a in out]
    return (vv, *out)


BASELINE_WINDOW = (-0.1, 0.2)     # non-Faradaic anchor for optional de-trending


def _detrend(bg_sub: np.ndarray, voltages: np.ndarray,
             baseline_window: tuple[float, float] = BASELINE_WINDOW) -> np.ndarray:
    """Subtract a linear baseline fit to the non-Faradaic region.

    Removes a residual slope left by imperfect 0nM-vs-dosed background matching.
    NOTE: ``baseline_window`` must sit **below** the faradaic onset: on channels whose
    DA response is a broad, low-onset ramp, a baseline fit through the onset
    subtracts the signal itself (see the FSCV peak-definition note in ``docs/DESIGN.md``).
    Off by default.
    """
    lo, hi = baseline_window
    m = (voltages >= lo) & (voltages <= hi)
    if m.sum() < 2:
        return bg_sub
    return bg_sub - np.polyval(np.polyfit(voltages[m], bg_sub[m], 1), voltages)


def _savgol(y: np.ndarray, window: int, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay smooth of ``y`` (the same filter EIS ``local_n`` uses on ``log|Z|``).

    ``window`` is in samples; it is forced odd and clamped to the array length. Returns
    ``y`` unchanged when ``window`` is falsy/too small or the array is too short (or if
    SciPy is unavailable). Off by default in the feature pipeline; pass a window to enable.
    """
    y = np.asarray(y, float)
    n = y.size
    if not window or window < 3 or n < 3:
        return y
    w = int(min(window, n if n % 2 else n - 1))
    if w % 2 == 0:
        w -= 1
    if w < 3:
        return y
    p = min(int(polyorder), w - 1)
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(y, window_length=w, polyorder=p)
    except Exception:  # pragma: no cover - scipy optional at runtime
        return y


#: default Savitzky-Golay smoothing of the raw voltammogram, which removes hardware noise without
#: shifting peak position (the reason for a low polynomial order on a short window). Applied to the (replicate-averaged) signal and background cycles at
#: the start of feature extraction, before background subtraction. Calibrate to your sampling.
FSCV_SMOOTH_WINDOW = 11          # samples (odd); 0 disables
FSCV_SMOOTH_POLY = 2


def smooth_current(current: np.ndarray, window: int = FSCV_SMOOTH_WINDOW,
                   poly: int = FSCV_SMOOTH_POLY) -> np.ndarray:
    """Savitzky-Golay smooth a raw FSCV current trace to remove hardware noise.

    Applied to the raw voltammogram (in acquisition/time order) before any background
    subtraction, so signal and background are de-noised identically. ``window=0`` (or too
    short an array) returns the trace unchanged. Thin wrapper over :func:`_savgol`.
    """
    return _savgol(current, window, poly)


def _locate_peak(bg_sub: np.ndarray, voltages: np.ndarray,
                 v_window: tuple[float, float], method: str = "chord",
                 require_interior: bool = False) -> tuple[int, float, float]:
    """Return (peak_index, V_ox, peak_height) within the voltage window.

    - ``method="chord"``: height is the vertical distance from the
      peak to the straight chord between the window endpoints. It isolates a sharp,
      localized peak on a curved baseline, but under-counts a broad hump that is
      ~linear across the window.
    - ``method="direct"``: height is the background-subtracted current at its
      in-window maximum (deviation from zero, no chord), the standard FSCV
      oxidation-current readout; robust to broad peaks.

    ``require_interior`` (default **off**): when the in-window maximum lands on a
    **window boundary sample**, i.e. the score is still rising at the edge, so there
    is no resolved interior local maximum (the peak is "pinned", or the switching-edge
    residual above ~1.0 V dominates), return ``(-1, nan, nan)`` ("no peak found")
    instead of the boundary point. Off by default, so the pipeline is unchanged unless
    you opt in; on, it turns edge-pinning into an explicit no-peak outcome rather than a
    boundary reading (see the ``peak_at_edge`` QC flag, which reports the same condition).
    """
    lo, hi = v_window
    idxs = np.where((voltages >= lo) & (voltages <= hi))[0]
    if idxs.size < 2:
        return -1, float("nan"), float("nan")
    vw, yw = voltages[idxs], bg_sub[idxs]
    if method == "direct":
        score = yw
    elif method == "chord":
        score = yw - np.interp(vw, [vw[0], vw[-1]], [yw[0], yw[-1]])
    else:
        raise ValueError(f"unknown peak method {method!r} (use 'chord' or 'direct')")
    pk = int(np.argmax(score))
    if require_interior and (pk == 0 or pk == score.size - 1):
        return -1, float("nan"), float("nan")   # argmax pinned to a window bound: no interior peak
    return int(idxs[pk]), float(vw[pk]), float(score[pk])


# backward-compatible alias (chord method)
def _peak(bg_sub, voltages, v_window):
    return _locate_peak(bg_sub, voltages, v_window, method="chord")


def _peak_readout(signal: np.ndarray, background: np.ndarray, voltages: np.ndarray,
                  v_window: tuple[float, float] = PEAK_WINDOW, method: str = "direct",
                  detrend: bool = False, baseline_window: tuple[float, float] = BASELINE_WINDOW,
                  smooth_window: int = 0, smooth_poly: int = 2,
                  require_interior: bool = False) -> dict:
    """Locate the anodic oxidation peak **once** and return every quantity read at it.

    All of ``NormIpeak``, ``V_ox`` and ``I_bgd`` come from the *same* located peak, using
    the *same* peak-detection ``method`` and the *same* pre-processing (``detrend`` /
    ``smooth_window``). This is what keeps the FSCV features mutually consistent (and
    consistent with ``NormIpeak``) rather than each re-locating the peak its own way. Keys:
    ``idx, v_ox, height, i_bgd, norm_ipeak`` (all NaN when no peak is found).
    """
    v, sig, bg = anodic_sweep(voltages, signal, background)
    bg_sub = sig - bg
    if detrend:
        bg_sub = _detrend(bg_sub, v, baseline_window)
    if smooth_window:
        bg_sub = _savgol(bg_sub, smooth_window, smooth_poly)
    idx, v_ox, height = _locate_peak(bg_sub, v, v_window, method=method,
                                     require_interior=require_interior)
    if idx < 0:
        return {"idx": -1, "v_ox": float("nan"), "height": float("nan"),
                "i_bgd": float("nan"), "norm_ipeak": float("nan")}
    i_bgd = float(bg[idx])
    return {"idx": int(idx), "v_ox": float(v_ox), "height": float(height), "i_bgd": i_bgd,
            "norm_ipeak": (height / i_bgd if i_bgd != 0 else float("nan"))}


def _fwhm(v: np.ndarray, y: np.ndarray, idx: int, height: float) -> float:
    """Full width at half maximum (in volts) of the peak at ``idx`` in ``y`` over ascending ``v``."""
    half = height / 2.0
    if not np.isfinite(half) or half <= 0:
        return float("nan")
    n = len(v)
    li = idx
    while li > 0 and y[li] > half:
        li -= 1
    ri = idx
    while ri < n - 1 and y[ri] > half:
        ri += 1
    if y[li] > half or y[ri] > half:            # half-max crossing not enclosed by the data
        return float("nan")
    vl = np.interp(half, [y[li], y[li + 1]], [v[li], v[li + 1]]) if y[li + 1] != y[li] else v[li]
    vr = np.interp(half, [y[ri], y[ri - 1]], [v[ri], v[ri - 1]]) if y[ri - 1] != y[ri] else v[ri]
    return float(abs(vr - vl))


def peak_shape(signal: np.ndarray, background: np.ndarray, voltages: np.ndarray,
               v_window: tuple[float, float] = PEAK_WINDOW, method: str = "direct",
               detrend: bool = False, baseline_window: tuple[float, float] = BASELINE_WINDOW,
               smooth_window: int = 0, smooth_poly: int = 2) -> dict:
    """Faradaic anodic-peak **shape** descriptors: deformation-mode characterization.

    Returns the background-subtracted peak ``height``, the peak ``area`` (charge ∝ ``∫(i−i_bg) dV``
    over the DA window, integrating only the positive faradaic lobe), and the ``fwhm`` (peak width in
    volts at half height). Located the *same way* as :func:`norm_ipeak` (same peak, same
    pre-processing). These capture how the DA response *deforms* over time (peak broadening / area
    loss = kinetics / fouling). **They are derived from the DA signal, so they are RESPONSE
    characterizations / candidate targets, not predictors** of a NormIpeak-derived target (that would
    leak the numerator). All NaN when no peak is found.
    """
    v, sig, bg = anodic_sweep(voltages, signal, background)
    bg_sub = sig - bg
    if detrend:
        bg_sub = _detrend(bg_sub, v, baseline_window)
    if smooth_window:
        bg_sub = _savgol(bg_sub, smooth_window, smooth_poly)
    idx, v_ox, height = _locate_peak(bg_sub, v, v_window, method=method)
    if idx < 0 or not np.isfinite(height):
        return {"height": float("nan"), "area": float("nan"), "fwhm": float("nan"),
                "area_clipped": False}
    lo, hi = v_window
    m = (v >= lo) & (v <= hi)
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    area = float(_trapz(np.clip(bg_sub[m], 0.0, None), v[m])) if int(m.sum()) >= 2 else float("nan")
    # flag a lobe truncated by the window/sweep edge: its area is then an UNDER-estimate (see
    # peak_edge_clipped). Consumers of `area` should exclude clipped values rather than trust them.
    clip = peak_edge_clipped(bg_sub, v, height, v_window)
    return {"height": float(height), "area": area, "fwhm": _fwhm(v, bg_sub, idx, height),
            "area_clipped": bool(clip["clipped"])}


def background_features(voltage: np.ndarray, current: np.ndarray,
                        window: tuple[float, float] = PEAK_WINDOW) -> dict:
    """DA-independent descriptors of the **0 nM background CV**: interface/charging state.

    Computed on the background cycle only (no DA), so they are **leakage-free predictors** and, like
    the EIS features, **dose-invariant** (one value per channel-timepoint). They read the non-faradaic
    charging behavior from the voltammetry side, complementing EIS (which measures the same interface
    across frequency); the most EIS-complementary of them is ``bg_switch`` (potential-window / fouling).

    Returns (all NaN if the CV is too short / not a loop):
    - ``bg_charge``: anodic charging charge, ``∫ i_bg dV`` over ``window`` (∝ C_dl × window),
    - ``bg_cap``: capacitive hysteresis, the mean ``|i_anodic − i_cathodic| / 2`` over ``window`` (a
      double-layer-capacitance proxy, the CV forward/back separation),
    - ``bg_switch``: background current at the anodic switching limit (solvent/electrode window edge).
    """
    v = np.asarray(voltage, dtype=float); i = np.asarray(current, dtype=float)
    out = {"bg_charge": float("nan"), "bg_cap": float("nan"), "bg_switch": float("nan")}
    if v.size < 4:
        return out
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    sw = int(np.argmax(v))                                   # switching potential (top of anodic sweep)
    out["bg_switch"] = float(i[sw])
    lo, hi = window
    va, ia = v[:sw + 1], i[:sw + 1]                          # anodic (rising) half
    ma = (va >= lo) & (va <= hi)
    if int(ma.sum()) >= 2:
        out["bg_charge"] = float(_trapz(ia[ma], va[ma]))
        vc, ic = v[sw:][::-1], i[sw:][::-1]                  # cathodic half, reversed to ascending V
        if vc.size >= 2 and np.all(np.diff(vc) >= 0):
            out["bg_cap"] = float(np.mean(np.abs(ia[ma] - np.interp(va[ma], vc, ic))) / 2.0)
    return out


def norm_ipeak(signal: np.ndarray, background: np.ndarray, voltages: np.ndarray,
               v_window: tuple[float, float] = PEAK_WINDOW, method: str = "direct",
               detrend: bool = False,
               baseline_window: tuple[float, float] = BASELINE_WINDOW,
               smooth_window: int = 0, smooth_poly: int = 2,
               require_interior: bool = False) -> float:
    """NormIpeak for one cycle pair (background-subtract → window → peak → normalize).

    ``method`` selects the peak-height definition (``"direct"`` = current at V_ox,
    the default, robust to the broad DA peaks seen in real data; ``"chord"`` = the
    the chord-baseline height, for sharp peaks; see :func:`_locate_peak`).
    ``detrend`` optionally removes a residual baseline slope first (opt-in; can eat
    broad-onset signal). ``smooth_window`` (samples, odd; **0 = off**) applies a
    Savitzky-Golay smooth (``smooth_poly``) to the background-subtracted cycle before
    locating the peak (the same filter family EIS ``local_n`` uses); off by default so
    the feature pipeline is unchanged unless you opt in here (and in ``extract_dataset``).
    ``mean_vpeak`` / ``mean_ibg`` accept the same ``method``/``detrend`` so V_ox and I_bgd
    are read at this very peak; pass them the same values you pass here.
    """
    return _peak_readout(signal, background, voltages, v_window, method, detrend,
                         baseline_window, smooth_window, smooth_poly, require_interior)["norm_ipeak"]


def mean_vpeak(signal: np.ndarray, background: np.ndarray, voltages: np.ndarray,
               v_window: tuple[float, float] = PEAK_WINDOW, method: str = "direct",
               detrend: bool = False, baseline_window: tuple[float, float] = BASELINE_WINDOW,
               smooth_window: int = 0, smooth_poly: int = 2) -> float:
    """Oxidation peak voltage V_ox, located the SAME way as ``norm_ipeak``.

    Pass the same ``method`` / ``detrend`` / ``smooth_window`` used for ``NormIpeak`` so
    this V_ox is the voltage at which that NormIpeak's height was measured.
    """
    return _peak_readout(signal, background, voltages, v_window, method, detrend,
                         baseline_window, smooth_window, smooth_poly)["v_ox"]


def mean_ibg(background: np.ndarray, signal: np.ndarray, voltages: np.ndarray,
             v_window: tuple[float, float] = PEAK_WINDOW, method: str = "direct",
             detrend: bool = False, baseline_window: tuple[float, float] = BASELINE_WINDOW,
             smooth_window: int = 0, smooth_poly: int = 2) -> float:
    """Background current at V_ox: a proxy for interfacial capacitance.

    Located the SAME way as ``norm_ipeak`` (pass the same ``method``/``detrend``/
    ``smooth_window``), so this equals that NormIpeak's denominator ``I_bgd``.
    """
    return _peak_readout(signal, background, voltages, v_window, method, detrend,
                         baseline_window, smooth_window, smooth_poly)["i_bgd"]


def noise_floor(signal: np.ndarray, background: np.ndarray, voltages: np.ndarray,
                window: tuple[float, float] = NONFARADAIC_WINDOW, detrend: bool = False,
                baseline_window: tuple[float, float] = BASELINE_WINDOW,
                smooth_window: int = 0, smooth_poly: int = 2) -> float:
    """RMS of the background-subtracted, background-normalized cycle over a
    non-Faradaic window. ``NormIpeak < 3·noise_floor`` is invalid.

    ``detrend`` matches :func:`norm_ipeak`: when the peak is measured on a de-trended
    cycle, the noise floor should be too, or the SNR is inconsistent. ``smooth_window``
    likewise matches :func:`norm_ipeak`: smooth both or neither so the SNR is consistent.
    """
    v, sig, bg = anodic_sweep(voltages, signal, background)
    bg_sub = sig - bg
    if detrend:
        bg_sub = _detrend(bg_sub, v, baseline_window)
    if smooth_window:
        bg_sub = _savgol(bg_sub, smooth_window, smooth_poly)
    bg_sub_norm = bg_sub / np.where(bg == 0, np.nan, bg)
    lo, hi = window
    mask = (v >= lo) & (v <= hi)
    vals = bg_sub_norm[mask]
    vals = vals[np.isfinite(vals)]
    return float(np.sqrt(np.mean(vals**2))) if vals.size else float("nan")


def repeatability_snr(normipeak: float, rep_normipeaks) -> float:
    """Reproducibility SNR: ``NormIpeak`` divided by the cycle-to-cycle scatter
    ``std(NormIpeak across replicate cycles)``.

    Unlike ``NormIpeak / noise_floor``, whose denominator is dominated by the systematic
    background-subtraction residual and so stays ~1x even for obvious peaks, this measures whether
    the peak is *reproducible* across the replicate cycles, which is what makes a dose-response
    "clearly visible." On real in-vitro data it separates kept vs rejected channels ~6x vs ~0.7x
    where the RMS-floor SNR cannot. Needs ≥2 finite replicate NormIpeaks; else ``NaN``. Perfectly
    identical replicates (zero scatter) return ``inf``.
    """
    a = np.asarray([x for x in rep_normipeaks if np.isfinite(x)], float)
    if a.size < 2 or not np.isfinite(normipeak):
        return float("nan")
    s = float(np.std(a))
    return float(normipeak / s) if s > 0 else float("inf")


def dose_response_corr(concentrations, values, method: str = "pearson") -> float:
    """Dose-response correlation of ``values`` (NormIpeak) vs ``log10(concentration)``.

    ``method='pearson'`` (default) measures **linear-in-log-dose** association (the
    monotonicity gate). ``method='spearman'`` measures **rank / monotonic** association, which does
    **not** assume a functional form, so a genuinely monotonic-but-saturating dose-response (common
    for DA at high concentration, where Pearson is depressed by the curvature) is not penalized.

    Needs ≥3 distinct positive concentrations and nonzero spread in ``values``; else ``NaN``.
    """
    c = np.asarray(concentrations, dtype=float)
    y = np.asarray(values, dtype=float)
    m = np.isfinite(c) & np.isfinite(y) & (c > 0)
    c, y = c[m], y[m]
    if np.unique(c).size < 3 or np.std(y) == 0:
        return float("nan")
    x = np.log10(c)
    if method == "spearman":
        from scipy.stats import spearmanr
        r = spearmanr(x, y).correlation
        return float(r) if r is not None and np.isfinite(r) else float("nan")
    if method != "pearson":
        raise ValueError(f"method must be 'pearson' or 'spearman', got {method!r}")
    return float(np.corrcoef(x, y)[0, 1])



def average_cycles(cycles: np.ndarray) -> np.ndarray:
    """Mean over replicate cycles, shape (n_cycles, n_samples) -> (n_samples,)."""
    return np.asarray(cycles, dtype=float).mean(axis=0)
