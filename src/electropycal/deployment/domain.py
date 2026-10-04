"""Domain-shift quantification via CORAL: how far is in vivo from in vitro?

CORAL aligns the first and second moments (mean, covariance) of two feature
distributions in the *original* feature basis (no subspace projection), so it
stays interpretable. In deployment we use it two ways: as a **distance** to flag
how far an in-vivo timepoint has drifted from the in-vitro training distribution
(Assumption-4 strain), and, optionally, as a **transform** to align in-vivo
features back onto the in-vitro scale before recalibration. Chaining distances
across timepoints traces the drift path.
"""

from __future__ import annotations

import numpy as np


def _cov(x: np.ndarray, eps: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    c = np.cov(x, rowvar=False)
    c = np.atleast_2d(c)
    return c + eps * np.eye(c.shape[0])


def coral_distance(
    source: np.ndarray,
    target: np.ndarray,
    include_mean: bool = True,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Domain distance between two feature matrices (rows = samples).

    Returns the covariance Frobenius distance ``cov_frob = ||Cov_s − Cov_t||_F``,
    the mean shift ``mean_l2 = ||μ_s − μ_t||₂``, and a combined ``distance``. Use
    the in-vitro training features as ``source`` and an in-vivo timepoint's
    features as ``target``.
    """
    cs, ct = _cov(source, eps), _cov(target, eps)
    cov_frob = float(np.linalg.norm(cs - ct, ord="fro"))
    mean_l2 = float(np.linalg.norm(np.mean(source, 0) - np.mean(target, 0)))
    return {
        "cov_frob": cov_frob,
        "mean_l2": mean_l2 if include_mean else 0.0,
        "distance": cov_frob + (mean_l2 if include_mean else 0.0),
    }


def coral_transform(
    source: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Align ``source`` onto ``target``'s second-order statistics (classic CORAL).

    Whitens the source covariance then re-colors with the target covariance:
    ``A = Cov_s^{-1/2} · Cov_t^{1/2}``, applied to mean-centered source, then the
    target mean is added back. Returns the transformed source matrix.
    """
    source = np.asarray(source, dtype=float)
    cs, ct = _cov(source, eps), _cov(target, eps)
    a = _matrix_power(cs, -0.5) @ _matrix_power(ct, 0.5)
    mu_s, mu_t = source.mean(0), np.asarray(target, dtype=float).mean(0)
    return (source - mu_s) @ a + mu_t


def drift_path(
    source: np.ndarray,
    targets_by_time: dict[object, np.ndarray],
    eps: float = 1e-6,
) -> list[dict]:
    """CORAL distance from the in-vitro ``source`` to each in-vivo timepoint.

    ``targets_by_time`` maps timepoint → feature matrix. Returns a time-ordered
    list of ``{timepoint, cov_frob, mean_l2, distance}``: the drift trajectory to
    log during deployment (and a candidate process model should a state-space/Kalman
    variant ever be built).
    """
    out = []
    for tp in sorted(targets_by_time):
        d = coral_distance(source, targets_by_time[tp], eps=eps)
        out.append({"timepoint": tp, **d})
    return out


def _matrix_power(m: np.ndarray, power: float) -> np.ndarray:
    """Symmetric-matrix power via eigendecomposition (m assumed SPD)."""
    vals, vecs = np.linalg.eigh(m)
    vals = np.clip(vals, 1e-12, None)
    return (vecs * (vals ** power)) @ vecs.T
