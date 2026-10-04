"""Linear PLSR core: fit, predict, VIP scores, and pickle-free serialization.

Thin wrapper over scikit-learn's ``PLSRegression`` engine with ``scale=False``
(scaling is done explicitly in the CV loop, so it stays leakage-safe), plus VIP scores and a
pickle-free asset bundle. Because the fitted linear model *is* its coefficient
matrix, intercept, and (upstream) normalization scalars, the bundle serializes as
``.npz`` + a JSON manifest, portable and version-robust (see DESIGN §8).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression


@dataclass
class PLSRModel:
    """Linear PLSR with ``k`` latent variables.

    Fit on already-D0-normalized, Z-scored features (``scale=False``). ``Y`` may
    be multi-column (one column per response, ``D`` of them); here typically a single
    ``NormIpeak`` column.
    """

    k: int
    _pls: PLSRegression | None = field(default=None, repr=False)
    _live: np.ndarray | None = field(default=None, repr=False)      # nonzero-variance column mask
    _p_full: int = field(default=0, repr=False)                     # original feature count

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "PLSRModel":
        X = np.ascontiguousarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        n, self._p_full = X.shape
        # Robustness: sklearn's PLS (scale=False) emits NaN loadings on a zero-variance column (its
        # y-side normalization divides by a zero score), which happens on band-empty features and on
        # CARS/CV row-resamples that make a column constant. Drop those for the fit (they carry no
        # signal → coef 0). Cap k at the numerical RANK too: collinear columns (adjacent-frequency EIS
        # features are near-duplicates) give an effective rank below p, and PLS asks for more components
        # than the data supports → NaN. matrix_rank uses an SVD tolerance, so it guards near-collinearity.
        self._live = X.std(0, ddof=0) > 0                           # nonzero-variance columns
        Xl = X[:, self._live] if self._live.any() else X[:, :0]
        rank = int(np.linalg.matrix_rank(Xl - Xl.mean(0))) if Xl.size else 0
        k = min(self.k, n - 1, Xl.shape[1], rank)                  # k ≤ min(n-1, p_live, rank)
        if k < 1 or np.ptp(Y, axis=0).max() == 0.0:                # no usable X variance, or constant target
            raise ValueError("cannot fit PLSR: degenerate fold (no feature variance / rank 0 / constant "
                             "target). Callers skip such folds; it does not arise on non-degenerate data.")
        self._pls = PLSRegression(n_components=k, scale=False)
        self._pls.fit(Xl, Y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on the modeled scale. Callers back-transform log-models to
        linear ``NormIpeak`` units before computing error metrics."""
        self._check_fit()
        X = np.ascontiguousarray(X, dtype=float)
        return self._pls.predict(X[:, self._live])

    def vip(self) -> np.ndarray:
        """Variable Importance in Projection, one score per feature.

        Multi-response form: ``VIP_j = sqrt( P · Σ_a w̃_ja² · SSY_a / Σ_a SSY_a )``,
        with per-component unit-normalized weights ``w̃`` and ``SSY_a`` the Y
        sum-of-squares explained by component ``a``.
        """
        self._check_fit()
        t = self._pls.x_scores_            # (n, A)
        w = self._pls.x_weights_           # (p_live, A)
        q = self._pls.y_loadings_          # (D, A)
        p_feat, a = w.shape
        ssy = np.array([(t[:, i] ** 2).sum() * (q[:, i] ** 2).sum() for i in range(a)])
        total = ssy.sum()
        out = np.zeros(self._p_full)                   # full-width; dropped columns score 0
        if total == 0:
            return out
        w_norm = w / np.linalg.norm(w, axis=0, keepdims=True)
        out[self._live] = np.sqrt(p_feat * ((w_norm ** 2) @ ssy) / total)
        return out

    # --- serialization: npz + manifest, no pickle -------------------------------

    def to_arrays(self) -> dict[str, np.ndarray]:
        """Arrays sufficient to reconstruct predictions in any language:
        ``Y = (X − x_mean) @ coef + y_mean``. Also keeps W*/P/C for interpretation.

        ``coef`` is computed as ``B = W* Cᵀ`` (shape P×D) rather than
        from ``pls.coef_``, whose orientation changed across scikit-learn versions.
        """
        self._check_fit()
        p, live = self._p_full, self._live
        pls = self._pls
        coef_l = np.asarray(pls.x_rotations_) @ np.asarray(pls.y_loadings_).T   # (P_live, D)
        a = np.asarray(pls.x_weights_).shape[1]

        def _pad(arr_live):                            # expand rows from the live set to full P (zeros elsewhere)
            arr_live = np.asarray(arr_live)
            full = np.zeros((p,) + arr_live.shape[1:]); full[live] = arr_live; return full
        return {
            "coef": _pad(coef_l),
            "x_mean": _pad(np.asarray(pls._x_mean).ravel()),
            "y_mean": np.asarray(pls._y_mean).ravel(),
            "x_weights": _pad(np.asarray(pls.x_weights_)),         # W* (rotations)
            "x_loadings": _pad(np.asarray(pls.x_loadings_)),       # P
            "y_loadings": np.asarray(pls.y_loadings_),             # C (D×A, unaffected by column drop)
        }

    def save_bundle(self, directory: str | Path, manifest_extra: dict | None = None) -> Path:
        """Write ``model_arrays.npz`` + ``manifest.json`` to ``directory``."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / "model_arrays.npz", **self.to_arrays())
        manifest = {
            "architecture": "linear_plsr",
            "k": int(self._pls.n_components),
            "predict": "Y = (X - x_mean) @ coef + y_mean",
            **_library_versions(),
            **(manifest_extra or {}),
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return directory

    def _check_fit(self) -> None:
        if self._pls is None:
            raise RuntimeError("PLSRModel is not fitted; call .fit(X, Y) first")


def _library_versions() -> dict[str, str]:
    import sklearn
    import scipy
    return {
        "versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        }
    }
