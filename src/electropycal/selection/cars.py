"""Competitive Adaptive Reweighted Sampling (CARS) feature selection.

Monte-Carlo feature selection wrapping PLSR: at each generation a random subset of
*samples* builds a PLS model, feature weights ``|B|`` are computed, an
exponentially decreasing function (EDF) forces the feature set to shrink toward a
minimum, and the surviving subset is scored by internal K-fold RMSECV (with a
small ``k`` sweep). The subset with the lowest RMSECV across generations wins.

Two non-determinisms to be aware of: runtime → the ``timebox_patience`` valve;
result → the RNG ``seed`` plus :func:`selection_stability` over multiple seeds.
"""

from __future__ import annotations

import numpy as np

from ..models.plsr import PLSRModel


def _kfold_rmsecv(X: np.ndarray, y: np.ndarray, k: int, n_folds: int,
                  rng: np.random.Generator) -> float:
    """K-fold RMSECV of a linear PLSR with ``k`` LVs on ``X`` (features preselected)."""
    n = len(y)
    folds = np.array_split(rng.permutation(n), min(n_folds, n))
    sse, m = 0.0, 0
    for i in range(len(folds)):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(len(folds)) if j != i])
        kk = min(k, min(len(tr) - 1, X.shape[1]))
        if kk < 1 or len(te) == 0:
            continue
        model = PLSRModel(k=kk).fit(X[tr], y[tr].reshape(-1, 1))
        pred = model.predict(X[te]).ravel()
        sse += float(((y[te] - pred) ** 2).sum())
        m += len(te)
    return float(np.sqrt(sse / m)) if m else float("inf")


def cars_select(X: np.ndarray, y: np.ndarray, k_max: int = 5,
                n_generations: int = 50, n_folds: int = 5, sample_ratio: float = 0.9,
                timebox_patience: int = 10, random_state: int | None = 0) -> np.ndarray:
    """Return the indices of the CARS-optimized feature subset (C-contiguous)."""
    X = np.ascontiguousarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, p = X.shape
    rng = np.random.default_rng(random_state)

    # EDF schedule: retained ratio r_i = a·exp(-b·i), r_1 = 1 (all) → r_N = 2/p.
    if p <= 2 or n_generations < 2:
        return np.arange(p)
    b = np.log(p / 2.0) / (n_generations - 1)
    a = np.exp(b)

    current = np.arange(p)
    best_subset, best_rmse, since_improve = current, float("inf"), 0

    for i in range(1, n_generations + 1):
        rows = rng.choice(n, size=max(2 * k_max, int(sample_ratio * n)), replace=False) \
            if int(sample_ratio * n) > 2 * k_max else np.arange(n)
        Xs, ys = X[rows][:, current], y[rows]
        kk = min(k_max, min(len(rows) - 1, len(current)))
        if kk < 1:
            break
        weights = np.abs(PLSRModel(k=kk).fit(Xs, ys.reshape(-1, 1)).to_arrays()["coef"].ravel())

        n_keep = max(2, int(round(a * np.exp(-b * i) * p)))
        n_keep = min(n_keep, len(current))
        keep_local = np.argsort(weights)[::-1][:n_keep]     # EDF forced selection by weight
        current = current[keep_local]

        rmse = _kfold_rmsecv(X[:, current], y, k_max, n_folds, rng)
        if rmse < best_rmse - 1e-9:
            best_rmse, best_subset, since_improve = rmse, current.copy(), 0
        else:
            since_improve += 1
        if since_improve >= timebox_patience or len(current) <= 2:  # time-box valve
            break

    return np.sort(best_subset)


def cars_select_multiseed(X: np.ndarray, y: np.ndarray, seeds=(0, 1, 2),
                          **kw) -> tuple[np.ndarray, np.ndarray]:
    """Run CARS across seeds; return (consensus subset, per-feature stability).

    Consensus = features selected by a majority of seeds. CARS is stochastic, so a single
    run's subset is not reproducible and should not be reported as *the* selected features;
    the majority-vote consensus is stable enough to interpret.
    """
    p = X.shape[1]
    index_sets = [cars_select(X, y, random_state=s, **kw) for s in seeds]
    freq = selection_stability(index_sets, p)
    consensus = np.where(freq >= 0.5)[0]
    return (consensus if consensus.size else np.sort(index_sets[0])), freq


def selection_stability(index_sets: list[np.ndarray], n_features: int) -> np.ndarray:
    """Per-feature selection frequency across seeds/folds → feature_stability.parquet."""
    freq = np.zeros(n_features)
    for idx in index_sets:
        freq[np.asarray(idx, dtype=int)] += 1
    return freq / max(1, len(index_sets))
