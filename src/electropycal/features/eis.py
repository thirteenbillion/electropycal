"""EIS feature extraction: impedance spectra to electrode-state features.

**Sign convention.** This module expects ``Z = Z' + j·Im(Z)`` with ``Im(Z) < 0`` for a capacitive
interface. A PSTrace export labels its column ``Z'' / Ohm`` but stores ``-Im(Z)`` there (positive
at low frequency for a capacitive interface), so :mod:`electropycal.data.pstrace` negates it on
ingest. Verify that sign on a known-capacitive channel of your own exports before trusting derived
capacitances, since it is a property of the instrument software rather than of the physics.
Capacitances carry an explicit sign correction so ``C_s, C_p, tau > 0``. The ``local_n`` derivative
and its band averages smooth ``log|Z|`` before differencing, because differencing raw ``log|Z|``
amplifies point noise into spurious slope structure.
"""

from __future__ import annotations

import numpy as np

from ..data.schema import BANDS_HZ

# np.trapz was renamed to np.trapezoid in NumPy 2.0.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


def eis_features(freqs: np.ndarray, z_real: np.ndarray, z_imag: np.ndarray,
                 smooth: bool = True) -> dict[str, np.ndarray]:
    """The 7 frequency-dependent EIS feature arrays.

    ``z_imag`` is passed as stored (negative for capacitive). Returns arrays keyed
    ``R_s, R_p, C_s, C_p, ideality_C, tau, local_n``, aligned to ``freqs``.
    """
    f = np.asarray(freqs, dtype=float)
    zr = np.asarray(z_real, dtype=float)
    zi = np.asarray(z_imag, dtype=float)          # negative (capacitive)
    w = 2 * np.pi * f
    z_mag2 = zr**2 + zi**2
    z_mag = np.sqrt(z_mag2)

    r_s = zr                                       # Z'
    r_p = z_mag2 / zr                              # |Z|^2 / Z' = 1/Re(Y)
    c_s = -1.0 / (w * zi)                          # sign: -1/(w Z'') > 0
    c_p = -zi / (w * z_mag2)                       # sign: -Z''/(w|Z|^2) = Im(Y)/w > 0
    ideality_c = (zi / z_mag) ** 2                 # sin^2(phase); sign-independent
    tau = -zi / (w * zr)                           # sign: R_p C_p = -Z''/(w Z') > 0
    local_n = _local_n(f, z_mag, smooth=smooth)    # -d log|Z| / d log f

    return {"R_s": r_s, "R_p": r_p, "C_s": c_s, "C_p": c_p,
            "ideality_C": ideality_c, "tau": tau, "local_n": local_n}


def eis_global_features(freqs: np.ndarray, z_real: np.ndarray, z_imag: np.ndarray,
                        smooth: bool = True, full_spectrum=None) -> dict[str, float]:
    """The whole-spectrum / band-averaged EIS features, plus two
    inductive-degradation predictors: ``min_neg_phase`` (severity, degrees) and
    ``inductive_onset_hz`` (frequency location of onset, Hz).

    The band-averaged features use the in-band ``freqs``/``z_real``/``z_imag`` passed in. The two
    inductive-degradation predictors are instead computed over ``full_spectrum`` (an optional
    ``(freq, z_real, z_imag)`` covering the **entire measured range**), so they are **independent of
    the analysis band**. This matters because a channel-timepoint only reaches feature extraction by
    passing EIS.1 (capacitive at every in-band point), which would otherwise pin the in-band onset at
    the band ceiling for every row (zero variance) and keep ``min_neg_phase`` blind to any inductive
    behaviour that lives just above the band. ``full_spectrum`` defaults to the in-band arrays."""
    f = np.asarray(freqs, dtype=float)
    feats = eis_features(f, z_real, z_imag, smooth=smooth)
    order = np.argsort(f)
    logf = np.log10(f[order])
    f_sorted = f[order]

    out: dict[str, float] = {}
    for name in ("R_s", "R_p", "C_s", "C_p"):
        out[f"{name}_integral"] = float(_trapz(feats[name][order], logf))
    out["f_ideality_crossover"] = _crossover(f_sorted, feats["ideality_C"][order], 0.5)
    for band, (lo, hi) in BANDS_HZ.items():
        m = (f >= lo) & (f < hi)
        out[f"ideality_C_band_{band}"] = float(feats["ideality_C"][m].mean()) if m.any() else np.nan
        out[f"n_band_{band}"] = float(feats["local_n"][m].mean()) if m.any() else np.nan
    imin, imax = int(np.argmin(f)), int(np.argmax(f))
    out["tau_ratio"] = float(feats["tau"][imin] / feats["tau"][imax])
    # --- band-INDEPENDENT inductive-degradation predictors (over the full measured spectrum) ---
    if full_spectrum is None:
        ff, fzr, fzi = f_sorted, np.asarray(z_real, float)[order], np.asarray(z_imag, float)[order]
    else:
        ff = np.asarray(full_spectrum[0], float)
        fo = np.argsort(ff)
        ff = ff[fo]
        fzr = np.asarray(full_spectrum[1], float)[fo]
        fzi = np.asarray(full_spectrum[2], float)[fo]
    # min_neg_phase (degrees): the closest the phase gets to 0 across the measured range: the
    # least-capacitive / most-inductive point. -phase > 0 while capacitive; goes negative if the
    # interface is inductive anywhere in range. Severity of inductive degradation. DA-independent.
    neg_phase = -np.degrees(np.arctan2(fzi, fzr))
    out["min_neg_phase"] = float(np.min(neg_phase))
    # inductive_onset_hz (Hz): the lowest frequency at which the interface reads inductive
    # (Im(Z) >= 0): the frequency *location* of onset, censored at the top of the measured range
    # when it never goes inductive. Lower = onset intrudes to lower frequency = more degraded.
    # DA-independent. (Full-range, so it is not pinned to the band ceiling by the EIS.1 gate.)
    ind = np.flatnonzero(fzi >= 0)
    out["inductive_onset_hz"] = float(ff[ind[0]]) if ind.size else float(ff[-1])
    return out


def _local_n(f: np.ndarray, z_mag: np.ndarray, smooth: bool) -> np.ndarray:
    """``-d log|Z| / d log f`` with optional smoothing of ``log|Z|`` first."""
    order = np.argsort(f)
    logf = np.log10(f[order])
    logz = np.log10(z_mag[order])
    if smooth and logz.size >= 5:
        try:
            from scipy.signal import savgol_filter
            win = logz.size if logz.size % 2 else logz.size - 1
            win = min(win, 7)
            if win >= 5:
                logz = savgol_filter(logz, window_length=win, polyorder=2)
        except Exception:  # pragma: no cover - scipy optional at runtime
            pass
    slope = np.gradient(logz, logf)
    out = np.empty_like(slope)
    out[order] = -slope
    return out


def _crossover(f_sorted: np.ndarray, y_sorted: np.ndarray, level: float) -> float:
    """First frequency where ``y`` crosses ``level`` (linear interp in log f).

    If ``y`` never crosses ``level`` in-band, the crossover is out of range: return
    the band edge on the side it would occur (``f_max`` if ``y`` stays above the
    level, ``f_min`` if below) so the feature stays finite rather than NaN."""
    d = y_sorted - level
    idx = np.where(np.diff(np.sign(d)) != 0)[0]
    if idx.size == 0:
        return float(f_sorted[-1] if np.nanmean(d) > 0 else f_sorted[0])
    i = int(idx[0])
    logf = np.log10(f_sorted)
    denom = d[i + 1] - d[i]
    t = (-d[i] / denom) if denom != 0 else 0.0
    return float(10 ** (logf[i] + t * (logf[i + 1] - logf[i])))
