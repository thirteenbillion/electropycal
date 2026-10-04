"""Measurement quality checks: per-spectrum and per-voltammogram QC gates.

EIS checks A-C validate that a spectrum is a passive, capacitive electrode
interface (Quadrant-I Nyquist). FSCV check D (noise floor) lives in
``features.fscv``. Only observations passing all checks feed feature extraction.
"""

from __future__ import annotations

import numpy as np


def eis_quality(freq: np.ndarray, z_real: np.ndarray, z_imag: np.ndarray,
                band: tuple[float, float] = (10.0, 100_000.0),
                mono_tol: float = 0.05) -> dict:
    """Return per-check booleans + overall ``valid`` for one EIS spectrum.

    Checks are applied within the analysis ``band`` (default 10 Hz to 100 kHz, the range the
    features are computed over); real electrodes are inductive above ~100 kHz from
    instrument/lead effects, which is not an electrode fault. ``z_imag`` is the
    math ``Im(Z)`` (capacitive < 0), as returned by ``data.pstrace``.

    - **A. Capacitive interface**: ``Im(Z) < 0`` at **every** in-band frequency
      (a working capacitive electrode stays capacitive across the whole band); a
      non-capacitive point anywhere in-band (low-f *or* a mid-band inductive
      excursion) indicates shunting / disruption → invalid. Checking all in-band
      points (not just the lowest) makes A self-contained: it no longer relies on
      the ``band`` upper bound having been placed below every inductive onset.
    - **B. Monotonic |Z|**: ``|Z|`` non-increasing with frequency within the band
      (``|Z| ~ 1/f``); a rising ``|Z|`` indicates DC drift → invalid.
    - **C. Environment**: passive medium has ``Z' > 0``; ``Z' < 0`` indicates
      amplifier saturation → invalid.

    Thresholds (``mono_tol``, band) should be calibrated against known good/broken
    channels; the defaults are a starting point.
    """
    freq = np.asarray(freq, float)
    zr = np.asarray(z_real, float)
    zi = np.asarray(z_imag, float)
    m = (freq >= band[0]) & (freq <= band[1])
    if m.sum() < 2:
        return {"A_capacitive": False, "B_monotonic": False, "C_environment": False,
                "valid": False, "n_in_band": int(m.sum())}
    order = np.argsort(freq[m])
    zr, zi = zr[m][order], zi[m][order]
    zmag = np.sqrt(zr**2 + zi**2)

    a_capacitive = bool(np.all(zi < 0))                       # capacitive across the whole band
    dz = np.diff(zmag)
    frac_rise = dz / np.maximum(zmag[:-1], 1e-12)             # step-to-step fractional |Z| change
    z_rise_max = float(np.max(frac_rise)) if frac_rise.size else 0.0
    b_monotonic = bool(z_rise_max <= mono_tol)                # EIS.2 statistic: max fractional rise
    c_environment = bool(np.all(zr > 0))
    return {"A_capacitive": a_capacitive, "B_monotonic": b_monotonic,
            "C_environment": c_environment, "n_in_band": int(m.sum()),
            "z_rise_max": z_rise_max,
            "valid": a_capacitive and b_monotonic and c_environment}


def inductive_onset(freq: np.ndarray, z_imag: np.ndarray) -> float:
    """Lowest frequency where ``Im(Z) >= 0``: the capacitive→inductive crossover.

    ``Im(Z) < 0`` is capacitive; at/above this onset the interface reads inductive
    and fails EIS.1 (check A). Returns ``nan`` when the spectrum stays capacitive
    across the whole measured range. Used to set the analysis ``band`` from data:
    keep the upper bound below the minimum onset so every in-band point is
    capacitive (see :func:`electropycal.data.inventory.recommended_band`).
    """
    freq = np.asarray(freq, float)
    zi = np.asarray(z_imag, float)
    order = np.argsort(freq)
    f, zi = freq[order], zi[order]
    ind = np.flatnonzero(zi >= 0)
    return float(f[ind[0]]) if ind.size else float("nan")
