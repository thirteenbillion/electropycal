"""Minimal end-to-end pipeline slice: one Track-2 linear-PLSR condition.

This runs the implemented pieces together — forward-chained folds → in-loop
Z-scoring (train-only, frozen) → PLSR fit → pooled metrics — so the pipeline and
notebooks work before the full ``discovery.scheduler`` (CARS, nested selection,
joblib) lands. It is the "Baselines 1.1" condition of the task queue.
"""

from __future__ import annotations

import numpy as np

from ..evaluation import metrics
from ..evaluation.cv import outer_folds
from ..features.normalize import apply_zscore, fit_zscore
from ..models.plsr import PLSRModel


def run_track2_linear(
    X: np.ndarray,
    y: np.ndarray,
    channel: np.ndarray,
    timepoint: np.ndarray,
    k: int = 2,
    mode: str = "loto_c_ac",
    min_train_times: int = 3,
) -> tuple[list[metrics.FoldResult], dict]:
    """Evaluate a single linear-PLSR model under forward-chained CV (Track 2).

    Returns ``(fold_results, aggregated_metrics)``. Z-scoring statistics are fit
    on each fold's training rows only and frozen before application to the test
    rows, so no test-set statistic can leak into training.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    folds: list[metrics.FoldResult] = []

    for fold in outer_folds(channel, timepoint, mode=mode, min_train_times=min_train_times):
        X_tr, y_tr = X[fold.train_idx], y[fold.train_idx]
        X_te, y_te = X[fold.test_idx], y[fold.test_idx]
        if X_tr.shape[0] < 2 * k:  # safety valve: N_train < 2k is too few rows for k components
            continue

        mean, std = fit_zscore(X_tr)
        model = PLSRModel(k=k).fit(apply_zscore(X_tr, mean, std), y_tr)
        y_pred = model.predict(apply_zscore(X_te, mean, std)).ravel()

        folds.append(
            metrics.FoldResult.from_predictions(
                channel=fold.channel, t_test=fold.t_test,
                y_true=y_te, y_pred=y_pred, mean_train=float(y_tr.mean()),
            )
        )

    return folds, metrics.aggregate(folds)
