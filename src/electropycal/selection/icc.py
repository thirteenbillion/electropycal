"""Variance-structure (ICC) pre-filter: keep features that are reliable within a sensor.

Keeps features whose variance is dominated by the *within-channel, between-time*
axis, i.e. features that actually track the sensitivity evolution we want to
recalibrate, rather than static between-channel or dose-response variance. Uses
the variance decomposition per feature. It is leakage-free, and ICC varies smoothly with the
threshold, so it does not need to be CV-optimized inside the modeling loop.
"""

from __future__ import annotations

import numpy as np

from ..diagnostics.variance import variance_hierarchy


def temporal_fraction(X: np.ndarray, channel: np.ndarray, timepoint: np.ndarray,
                      concentration: np.ndarray) -> np.ndarray:
    """Per-feature fraction of variance on the within-channel temporal axis."""
    X = np.asarray(X, float)
    return np.array([
        variance_hierarchy(X[:, j], channel, timepoint, concentration)["within_temporal"]
        for j in range(X.shape[1])
    ])


def icc_prefilter(X: np.ndarray, channel: np.ndarray, timepoint: np.ndarray,
                  concentration: np.ndarray, threshold: float) -> np.ndarray:
    """Indices of features with within-temporal variance fraction ≥ ``threshold``."""
    frac = temporal_fraction(X, channel, timepoint, concentration)
    idx = np.where(frac >= threshold)[0]
    return idx if idx.size else np.array([int(np.argmax(frac))])
