"""Uniform model-asset bundle (DESIGN §8).

Every architecture reduces to a linear map in feature space plus an optional
output transform, so a fitted model serializes as arrays + a manifest and can be
reconstructed/applied with no pickle and no scikit-learn at load time:

    y = (X - x_mean) @ coef + y_mean          # then output_transform (identity | exp)

The extra W*/P/C arrays are kept for VIP/interpretation but are not needed for
prediction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..data import io


def save_model_bundle(directory: str | Path, arrays: dict[str, np.ndarray],
                      manifest: dict) -> Path:
    """Write ``model_arrays.npz`` + ``manifest.json`` to ``directory``."""
    directory = Path(directory)
    io.save_npz(directory / "model_arrays.npz", **arrays)
    io.write_json(directory / "manifest.json", manifest)
    return directory


def load_model_bundle(directory: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    directory = Path(directory)
    arrays = io.load_npz(directory / "model_arrays.npz")
    manifest = io.read_json(directory / "manifest.json")
    return arrays, manifest


def poly_powers(n_components: int, degree: int) -> np.ndarray:
    """Exponent rows for a total-degree polynomial design (incl. bias row of 0s)."""
    from itertools import product
    rows = [e for e in product(range(degree + 1), repeat=n_components) if sum(e) <= degree]
    return np.array(sorted(rows, key=sum), dtype=float)


def poly_design(T: np.ndarray, powers: np.ndarray) -> np.ndarray:
    """Design matrix ``Phi[:, j] = prod_c T[:, c] ** powers[j, c]``."""
    T = np.asarray(T, float)
    return np.stack([np.prod(T ** powers[j], axis=1) for j in range(powers.shape[0])], axis=1)


def predict_from_bundle(arrays: dict[str, np.ndarray], manifest: dict,
                        X: np.ndarray) -> np.ndarray:
    """Reconstruct predictions from a bundle.

    Linear family: ``y = (X − x_mean) @ coef + y_mean`` (+ exp for log models).
    Nonlinear PLSR: project to scores then apply the stored polynomial inner map.
    """
    X = np.asarray(X, dtype=float)
    if manifest.get("architecture") == "nonlinear_plsr":
        T = (X - arrays["x_mean"]) @ arrays["rotations"]
        return (poly_design(T, arrays["powers"]) @ arrays["beta"]).ravel()
    y = (X - arrays["x_mean"]) @ arrays["coef"] + arrays["y_mean"]
    y = np.asarray(y).ravel()
    if manifest.get("output_transform") == "exp":
        y = np.exp(y)
    return y
