"""PLSR architecture registry: the linear core plus its weighted/orthogonal/log variants.

Each architecture reduces to a linear map + output transform (``models.base``), so
all serialize/predict uniformly. Implemented: linear, log-transformed, weighted.
Orthogonal / nonlinear / kernel / multi-block are registered as clear extension
points (raise on use) — the discovery machinery is complete and adding one is a
plug-in, not a rewrite.

Fit on already-D0-normalized, Z-scored features (``scale=False`` inside PLSR).
"""

from __future__ import annotations

import numpy as np

from .plsr import PLSRModel


class BaseArch:
    name = "base"
    output_transform = "identity"

    def __init__(self, k: int):
        self.k = k
        self.plsr: PLSRModel | None = None
        self.coef = self.x_mean = self.y_mean = None

    def _finalize(self, arrays: dict[str, np.ndarray]) -> "BaseArch":
        self.coef = arrays["coef"]
        self.x_mean = arrays["x_mean"]
        self.y_mean = arrays["y_mean"]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        y = (np.asarray(X, float) - self.x_mean) @ self.coef + self.y_mean
        y = np.asarray(y).ravel()
        return np.exp(y) if self.output_transform == "exp" else y

    def vip(self) -> np.ndarray:
        return self.plsr.vip()

    def to_arrays(self) -> dict[str, np.ndarray]:
        a = self.plsr.to_arrays()
        a["coef"], a["x_mean"], a["y_mean"] = self.coef, self.x_mean, np.atleast_1d(self.y_mean)
        return a

    def manifest(self) -> dict:
        return {"architecture": self.name, "k": int(self.plsr._pls.n_components),
                "output_transform": self.output_transform,
                "predict": "y = (X - x_mean) @ coef + y_mean"}


class LinearPLSR(BaseArch):
    name = "linear_plsr"

    def fit(self, X, y, **_):
        self.plsr = PLSRModel(k=self.k).fit(X, np.asarray(y).reshape(-1, 1))
        return self._finalize(self.plsr.to_arrays())


class LogPLSR(BaseArch):
    """Fits in ``log(NormIpeak)`` space, so it minimizes *relative* rather than absolute error."""

    name = "log_plsr"
    output_transform = "exp"

    def fit(self, X, y, **_):
        y = np.asarray(y, float)
        if np.any(y <= 0):
            raise ValueError("LogPLSR requires positive y (NormIpeak)")
        self.plsr = PLSRModel(k=self.k).fit(X, np.log(y).reshape(-1, 1))
        return self._finalize(self.plsr.to_arrays())


class WeightedPLSR(BaseArch):
    """Weighted PLSR via the sqrt-weight trick.

    Weighted-center X and y, scale rows by ``sqrt(w)``, fit standard PLSR; the
    resulting slope is the WLS estimate, and we replace scikit-learn's intercept
    with the weighted means for prediction. ``sample_weight`` defaults to uniform
    (≡ linear); the runner passes ``1/concentration`` when available.
    """

    name = "weighted_plsr"

    def fit(self, X, y, sample_weight=None, **_):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        w = w / w.sum() * len(w)                       # normalize scale
        wmean_x = (w[:, None] * X).sum(0) / w.sum()
        wmean_y = float((w * y).sum() / w.sum())
        s = np.sqrt(w)
        self.plsr = PLSRModel(k=self.k).fit(
            s[:, None] * (X - wmean_x), (s * (y - wmean_y)).reshape(-1, 1))
        arrays = self.plsr.to_arrays()
        arrays["x_mean"] = wmean_x                      # override intercept with weighted means
        arrays["y_mean"] = np.array([wmean_y])
        return self._finalize(arrays)


class OrthogonalPLSR(BaseArch):
    """Orthogonal PLSR (O-PLS).

    Removes ``n_orth`` Y-orthogonal components from X, then fits ``k`` predictive
    PLSR LVs on the deflated X. The orthogonal filter is a product of rank-1
    projections, so the whole model composes to a single linear map and serializes
    like the linear family. ``n_orth`` is a hyperparameter (default 1; sweepable).
    """

    name = "orthogonal_plsr"

    def __init__(self, k: int, n_orth: int = 1):
        super().__init__(k)
        self.n_orth = n_orth

    def fit(self, X, y, **_):
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()
        p = X.shape[1]
        x_mean = X.mean(0)
        Xc = X - x_mean
        yc = y - y.mean()

        M = np.eye(p)
        Xcur = Xc.copy()
        for _i in range(self.n_orth):
            w = Xcur.T @ yc
            nw = np.linalg.norm(w)
            if nw < 1e-12:
                break
            w /= nw
            t = Xcur @ w
            pl = Xcur.T @ t / (t @ t)
            w_o = pl - ((w @ pl) / (w @ w)) * w
            n_o = np.linalg.norm(w_o)
            if n_o < 1e-12:
                break
            w_o /= n_o
            t_o = Xcur @ w_o
            p_o = Xcur.T @ t_o / (t_o @ t_o)
            F = np.eye(p) - np.outer(w_o, p_o)          # X_i = X_{i-1} (I - w_o p_oᵀ)
            Xcur = Xcur @ F
            M = M @ F

        self.plsr = PLSRModel(k=self.k).fit(Xc @ M, y.reshape(-1, 1))
        a = self.plsr.to_arrays()
        coef = M @ a["coef"]                              # compose filter with PLS coef
        y_mean = np.atleast_1d(a["y_mean"] - a["x_mean"] @ a["coef"])
        return self._finalize({"coef": coef, "x_mean": x_mean, "y_mean": y_mean})

    def manifest(self) -> dict:
        return {**super().manifest(), "n_orth": int(self.n_orth)}


class NonlinearPLSR(BaseArch):
    """Nonlinear PLSR: polynomial inner relation between scores and response.

    Fits ``k`` PLS components, then a total-degree ``degree`` polynomial from the
    X-scores to y. Not a single linear map, so it carries its own predict + bundle
    (rotations + polynomial powers/coefficients); ``models.base`` dispatches on the
    architecture name.
    """

    name = "nonlinear_plsr"

    def __init__(self, k: int, degree: int = 2):
        super().__init__(k)
        self.degree = degree

    def fit(self, X, y, **_):
        from .base import poly_design, poly_powers
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()
        self.plsr = PLSRModel(k=self.k).fit(X, y.reshape(-1, 1))
        self._R = np.asarray(self.plsr._pls.x_rotations_)
        self._x_mean = np.asarray(self.plsr._pls._x_mean)
        T = (X - self._x_mean) @ self._R
        self._powers = poly_powers(T.shape[1], self.degree)
        self._beta, *_ = np.linalg.lstsq(poly_design(T, self._powers), y, rcond=None)
        return self

    def predict(self, X):
        from .base import poly_design
        T = (np.asarray(X, float) - self._x_mean) @ self._R
        return (poly_design(T, self._powers) @ self._beta).ravel()

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {"rotations": self._R, "x_mean": self._x_mean,
                "powers": self._powers, "beta": self._beta}

    def manifest(self) -> dict:
        return {"architecture": self.name, "k": int(self.plsr._pls.n_components),
                "degree": int(self.degree), "output_transform": "identity",
                "predict": "poly(scores)"}


ARCHITECTURES: dict[str, type[BaseArch]] = {
    "linear_plsr": LinearPLSR,
    "log_plsr": LogPLSR,
    "weighted_plsr": WeightedPLSR,
    "orthogonal_plsr": OrthogonalPLSR,
    "nonlinear_plsr": NonlinearPLSR,
}

_NOT_IMPLEMENTED = ("kernel_plsr", "multiblock_plsr", "multilevel_plsr")


def build(architecture: str, k: int, **kwargs) -> BaseArch:
    """Instantiate an architecture by name (extra kwargs: ``n_orth``, ``degree``)."""
    if architecture in _NOT_IMPLEMENTED:
        raise NotImplementedError(
            f"{architecture!r} is a registered extension point; implement it as a "
            "models.variants.BaseArch subclass and add it to ARCHITECTURES.")
    if architecture not in ARCHITECTURES:
        raise KeyError(f"unknown architecture {architecture!r}; have {list(ARCHITECTURES)}")
    return ARCHITECTURES[architecture](k=k, **kwargs)
